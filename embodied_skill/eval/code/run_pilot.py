"""Phase 4 路由阶段（方案 B：在 Windows / skill-router 环境跑，不碰仿真）。

职责：对采样 query 跑各路由方法，① 打印检索/选择指标（Hit@K / FullCoverage@K /
选择命中）；② 把"每题每方法的排序结果 + 选中技能集合"写进 selections.json，
供 WSL 侧仿真消费脚本 run_sim_from_selections.py 读它跑 SR/GC。两阶段解耦。

方法：
  bm25  : B1 词面检索，chosen = top-1
  dense : B2 稠密检索(多语向量)，chosen = top-1
  b0    : B0 LLM 全塞 prompt，chosen = LLM 选出的集合（调 Kimi，慢）

selections.json 与同一 (n, seed, pool_size) 的多次运行可增量合并（不同方法合到一起）。

用法（Windows，conda skill-router 环境）：
  python eval\\code\\run_pilot.py --n 100 --methods bm25 dense
  python eval\\code\\run_pilot.py --n 100 --methods b0        # 慢，合并进同一文件
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))                     # embodied_skill/
ROUTER_CODE = os.path.join(ROOT, "router", "code")
EVAL_DATA = os.path.join(ROOT, "eval", "datasets")
sys.path.insert(0, ROUTER_CODE)
sys.path.insert(0, HERE)
from common import load_pool, load_queries, skill_sig  # noqa: E402
import metrics as M  # noqa: E402

KS = [1, 3, 5, 10]
TOPN_STORE = 20   # selections.json 里为检索方法保存的排序长度


def q_key(q):
    return q["task_id"]


def q_meta(q):
    """写进 selections.json 的题目公共信息（供仿真使用）。"""
    return {
        "task_id": q["task_id"],
        "task_type": q["task_type"],
        "split": q.get("split"),
        "floor_plan": q["floor_plan"],
        "binding": q["binding"],
        "instruction": q["instruction"],
        "required_effects": q["required_effects"],
        "gt_ids": sorted(set(q.get("gt_composite_set", [])) | set(q.get("gt_atomic_set", []))),
    }


def eval_retriever(retriever, queries, entries, by_id):
    agg = {f"hit@{k}": 0 for k in KS}
    agg.update({f"fc@{k}": 0 for k in KS})
    agg.update({f"idhit@{k}": 0 for k in KS})
    t0 = time.time()
    for q in queries:
        ranked = retriever.rank(q["instruction"])
        req = q["required_effects"]
        binding = q.get("binding")
        gt_ids = set(q.get("gt_composite_set", [])) | set(q.get("gt_atomic_set", []))
        for k in KS:
            agg[f"hit@{k}"] += M.hit_at_k(ranked, req, k, by_id, binding)
            agg[f"fc@{k}"] += M.fullcov_at_k(ranked, req, k, by_id, binding)
            agg[f"idhit@{k}"] += M.id_hit_at_k(ranked, gt_ids, k)
        ranked_ids = [c["skill_id"] for c in ranked[:TOPN_STORE]]
        entries[q_key(q)]["methods"][retriever.name] = {
            "ranked_ids": ranked_ids,
            "chosen_ids": ranked_ids[:1],   # 检索方法：chosen = top-1
        }
    dt = time.time() - t0
    n = len(queries)
    return {k: v / n for k, v in agg.items()}, dt


def eval_selector(selector, queries, entries, by_id, retries=1, only_missing=False):
    """LLM/流水线选择方法：直接产出 chosen 集合，评 sel_hit/sel_fullcov。
    逐题容错：单题超时/异常最多重试 retries 次，仍失败则记空并继续（不拖垮整轮）。"""
    sel_hit = sel_fc = size = 0
    fails = 0
    n = len(queries)
    t0 = time.time()
    for i, q in enumerate(queries):
        # 只回填模式：已有非空 chosen 的题跳过（省去慢/贵的 LLM 重算）
        if only_missing:
            prev = entries[q_key(q)]["methods"].get(selector.name)
            if prev and prev.get("chosen_ids"):
                chosen = [selector.by_id[c] for c in prev["chosen_ids"] if c in selector.by_id] \
                    if hasattr(selector, "by_id") else []
                req = q["required_effects"]
                s = M.selection_scores(chosen, req, by_id, q.get("binding"))
                sel_hit += s["sel_hit"]; sel_fc += s["sel_fullcov"]; size += s["sel_size"]
                continue
        qt = time.time()
        chosen = None
        wants_binding = getattr(selector, "wants_binding", False)
        for attempt in range(retries + 1):
            try:
                chosen = selector.select(q["instruction"], q.get("binding")) \
                    if wants_binding else selector.select(q["instruction"])
                break
            except Exception as ex:
                if attempt < retries:
                    continue
                fails += 1
                print(f"  [{i+1}/{n}] 失败记空: {ex}", flush=True)
                chosen = []
        req = q["required_effects"]
        s = M.selection_scores(chosen, req, by_id, q.get("binding"))
        sel_hit += s["sel_hit"]
        sel_fc += s["sel_fullcov"]
        size += s["sel_size"]
        entries[q_key(q)]["methods"][selector.name] = {
            "ranked_ids": [],
            "chosen_ids": [c["skill_id"] for c in chosen],
        }
        print(f"  [{i+1}/{n}] {time.time()-qt:4.1f}s cov={'✓' if s['sel_fullcov'] else '✗'} "
              f"-> {[c['skill_id'] for c in chosen]}", flush=True)
    dt = time.time() - t0
    if fails:
        print(f"  (b0 失败 {fails}/{n} 题，已记空)")
    return {"sel_hit": sel_hit / n, "sel_fullcov": sel_fc / n, "sel_size": size / n}, dt


def load_or_init_selections(path, n, seed, pool_size, queries):
    """存在且 (n,seed,pool_size) 一致则复用（增量加方法），否则新建。"""
    entries = {}
    if os.path.exists(path):
        try:
            data = json.load(open(path, encoding="utf-8"))
            m = data.get("meta", {})
            if m.get("n") == n and m.get("seed") == seed and m.get("pool_size") == pool_size:
                for e in data.get("queries", []):
                    entries[e["task_id"]] = e
        except Exception:
            entries = {}
    for q in queries:
        k = q_key(q)
        if k not in entries:
            e = q_meta(q)
            e["methods"] = {}
            entries[k] = e
        entries[k].setdefault("methods", {})
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="采样 query 数（-1=全部）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=["bm25"],
                    choices=["bm25", "dense", "b0", "b3", "m1", "m1f", "m1h", "m1h4", "m1h4nr", "b4"])
    ap.add_argument("--out", default=os.path.join(EVAL_DATA, "selections.json"))
    ap.add_argument("--gt", default=None,
                    help="GT jsonl 文件名或绝对路径（默认 gt_kitchen.jsonl；headline 用 gt_kitchen_eval.jsonl）")
    ap.add_argument("--only-missing", action="store_true",
                    help="选择型方法只重算 chosen 为空/缺失的题（回填超时空选）")
    args = ap.parse_args()

    pool = load_pool()
    n = None if args.n < 0 else args.n
    gt_path = args.gt
    if gt_path and not os.path.isabs(gt_path):
        gt_path = os.path.join(ROOT, "gt", "datasets", gt_path)
    queries = load_queries(n=n, seed=args.seed, gt_path=gt_path)
    by_id = {c["skill_id"]: c for c in pool}
    nn = len(queries)
    print(f"技能池 {len(pool)} 条；评测 query {nn} 条；方法={args.methods}\n")

    entries = load_or_init_selections(args.out, args.n, args.seed, len(pool), queries)

    def _print_ret(r, res, bt, dt):
        print(f"== {r.name} == (构建 {bt:.1f}s, 检索 {dt:.1f}s, {dt/nn*1000:.1f}ms/题)")
        print("  Hit@K       :", "  ".join(f"@{k}={res[f'hit@{k}']:.2f}" for k in KS))
        print("  FullCov@K   :", "  ".join(f"@{k}={res[f'fc@{k}']:.2f}" for k in KS))
        print("  ID-Hit@K    :", "  ".join(f"@{k}={res[f'idhit@{k}']:.2f}" for k in KS), "\n")

    if "bm25" in args.methods:
        try:
            from retrievers import BM25Retriever
            t = time.time()
            r = BM25Retriever(pool)
            bt = time.time() - t
            res, dt = eval_retriever(r, queries, entries, by_id)
            _print_ret(r, res, bt, dt)
        except Exception as ex:
            print(f"!! bm25 跳过：{ex}\n")

    if "dense" in args.methods:
        try:
            from retrievers import DenseRetriever
            t = time.time()
            r = DenseRetriever(pool)
            bt = time.time() - t
            res, dt = eval_retriever(r, queries, entries, by_id)
            _print_ret(r, res, bt, dt)
        except Exception as ex:
            print(f"!! dense 跳过：{ex}\n")

    if "b0" in args.methods:
        try:
            from llm_baselines import B0LLMFull
            r = B0LLMFull(pool)
            res, dt = eval_selector(r, queries, entries, by_id,
                                    only_missing=args.only_missing)
            print(f"== {r.name} == (总 {dt:.1f}s, {dt/nn:.1f}s/题)")
            print(f"  Sel-Hit={res['sel_hit']:.2f}  Sel-FullCov={res['sel_fullcov']:.2f}  "
                  f"平均选中={res['sel_size']:.1f} 条\n")
        except Exception as ex:
            print(f"!! b0 跳过：{ex}\n")

    if "b3" in args.methods:
        try:
            from pipelines import B3SkillFlow
            r = B3SkillFlow(pool)
            res, dt = eval_selector(r, queries, entries, by_id, retries=0,
                                    only_missing=args.only_missing)
            print(f"== {r.name} == (总 {dt:.1f}s, {dt/nn:.1f}s/题)")
            print(f"  Sel-Hit={res['sel_hit']:.2f}  Sel-FullCov={res['sel_fullcov']:.2f}  "
                  f"平均选中={res['sel_size']:.1f} 条\n")
        except Exception as ex:
            print(f"!! b3 跳过：{ex}\n")

    if "m1" in args.methods:
        try:
            from domain_m1 import M1RuleDomain
            r = M1RuleDomain(pool)
            res, dt = eval_selector(r, queries, entries, by_id, retries=0)
            print(f"== {r.name} == (总 {dt:.1f}s, {dt/nn*1000:.1f}ms/题)")
            print(f"  Sel-Hit={res['sel_hit']:.2f}  Sel-FullCov={res['sel_fullcov']:.2f}  "
                  f"平均选中={res['sel_size']:.1f} 条\n")
        except Exception as ex:
            print(f"!! m1 跳过：{ex}\n")

    if "m1f" in args.methods:
        try:
            from domain_m1 import M1RuleDomain
            from goal_parser_llm import LLMGoalParser
            r = M1RuleDomain(pool, goal_fn=LLMGoalParser(), name="M1-Formal")
            res, dt = eval_selector(r, queries, entries, by_id, retries=0,
                                    only_missing=args.only_missing)
            print(f"== {r.name} == (总 {dt:.1f}s, {dt/nn*1000:.1f}ms/题)")
            print(f"  Sel-Hit={res['sel_hit']:.2f}  Sel-FullCov={res['sel_fullcov']:.2f}  "
                  f"平均选中={res['sel_size']:.1f} 条\n")
        except Exception as ex:
            print(f"!! m1f 跳过：{ex}\n")

    if "m1h" in args.methods:
        try:
            from domain_m1 import M1RuleDomain
            from l0_head import L0HeadParser
            r = M1RuleDomain(pool, goal_fn=L0HeadParser(), name="M1-Head")
            res, dt = eval_selector(r, queries, entries, by_id, retries=0)
            print(f"== {r.name} == (总 {dt:.1f}s, {dt/nn*1000:.1f}ms/题)")
            print(f"  Sel-Hit={res['sel_hit']:.2f}  Sel-FullCov={res['sel_fullcov']:.2f}  "
                  f"平均选中={res['sel_size']:.1f} 条\n")
        except Exception as ex:
            print(f"!! m1h 跳过：{ex}\n")

    def _resolve_l4_ckpt():
        """L4_CKPT 环境变量：绝对路径直接用；相对/纯文件名落到 router/models/ 下。"""
        ck = os.environ.get("L4_CKPT")
        if not ck:
            return None
        if not os.path.isabs(ck) and not os.path.exists(ck):
            cand = os.path.join(ROOT, "router", "models", os.path.basename(ck))
            if os.path.exists(cand):
                return cand
        return ck

    if "m1h4" in args.methods:
        try:
            from domain_m1 import M1RuleDomain
            from l0_head import L0HeadParser
            from l4_head import L4HeadScorer
            ck = _resolve_l4_ckpt()
            scorer = L4HeadScorer(ckpt=ck) if ck else L4HeadScorer()
            if ck:
                print(f"  [m1h4] L4_CKPT={ck}")
            r = M1RuleDomain(pool, goal_fn=L0HeadParser(), l4_scorer=scorer,
                             name="M1-Head+L4")
            res, dt = eval_selector(r, queries, entries, by_id, retries=0)
            print(f"== {r.name} == (总 {dt:.1f}s, {dt/nn*1000:.1f}ms/题)")
            print(f"  Sel-Hit={res['sel_hit']:.2f}  Sel-FullCov={res['sel_fullcov']:.2f}  "
                  f"平均选中={res['sel_size']:.1f} 条\n")
        except Exception as ex:
            print(f"!! m1h4 跳过：{ex}\n")

    if "m1h4nr" in args.methods:
        try:
            from domain_m1 import M1RuleDomain
            from l0_head import L0HeadParser
            from l4_head import L4HeadScorer
            ck = _resolve_l4_ckpt()
            scorer = L4HeadScorer(ckpt=ck) if ck else L4HeadScorer()
            if ck:
                print(f"  [m1h4nr] L4_CKPT={ck}")
            r = M1RuleDomain(pool, goal_fn=L0HeadParser(), l4_scorer=scorer,
                             no_legality=True, no_repair=True, name="M1-Head+L4-noRule")
            res, dt = eval_selector(r, queries, entries, by_id, retries=0)
            print(f"== {r.name} == (总 {dt:.1f}s, {dt/nn*1000:.1f}ms/题)")
            print(f"  Sel-Hit={res['sel_hit']:.2f}  Sel-FullCov={res['sel_fullcov']:.2f}  "
                  f"平均选中={res['sel_size']:.1f} 条\n")
        except Exception as ex:
            print(f"!! m1h4nr 跳过：{ex}\n")

    if "b4" in args.methods:
        try:
            from lota_b4 import B4LoTaBench
            batch = int(os.environ.get("B4_BATCH", "16"))
            r = B4LoTaBench(pool, batch_size=batch, norm="mean")
            res, dt = eval_selector(r, queries, entries, by_id, retries=0,
                                    only_missing=args.only_missing)
            print(f"== {r.name} == (总 {dt:.1f}s, {dt/nn:.1f}s/题)")
            print(f"  Sel-Hit={res['sel_hit']:.2f}  Sel-FullCov={res['sel_fullcov']:.2f}  "
                  f"平均选中={res['sel_size']:.1f} 条\n")
        except Exception as ex:
            print(f"!! b4 跳过：{ex}\n")

    os.makedirs(EVAL_DATA, exist_ok=True)
    out = {
        "meta": {"n": args.n, "seed": args.seed, "pool_size": len(pool),
                 "generated_at": datetime.now().isoformat(timespec="seconds")},
        "queries": [entries[q_key(q)] for q in queries],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    methods_present = sorted({m for e in out["queries"] for m in e["methods"]})
    print(f"已写 {args.out}\n  query {len(out['queries'])} 条；已含方法：{methods_present}")


if __name__ == "__main__":
    main()
