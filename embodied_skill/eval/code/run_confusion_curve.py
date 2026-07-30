"""WP8 产出3：混淆度曲线（Confusion-density curve）。

三档池 pool_tier_{low,mid,high} 共享同一 34 金标准基准，仅改干扰密度(1/3/6)。
为让"混淆度"成为唯一自变量，只在【答案落在该 34 基准子集内】的 query 上评测
（否则 tier 池里无正确技能，不公平）；三档共用同一答案集 = 受控对比。

指标（effect-grounded，与 WP2a 一致）：
- 选择型(m1/b0/b3)：Sel-FullCov（chosen 真实效果并集 ⊇ required_effects 的比例）
- 检索型(bm25/dense)：FullCov@1（top-1 真实效果覆盖）

用法（Windows, skill-router 环境）：
  python eval/code/run_confusion_curve.py --methods m1 bm25 dense
  python eval/code/run_confusion_curve.py --methods m1 bm25 dense b0 b3   # 含 LLM，慢
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ROUTER_CODE = os.path.join(ROOT, "router", "code")
SKILLS_DATA = os.path.join(ROOT, "skills", "datasets")
GT_DATA = os.path.join(ROOT, "gt", "datasets")
EVAL_DATA = os.path.join(ROOT, "eval", "datasets")
sys.path.insert(0, ROUTER_CODE)
sys.path.insert(0, HERE)
import common  # noqa: E402
import metrics as M  # noqa: E402

TIERS = ["low", "mid", "high"]


def balanced_subsample(queries, n, seed=0):
    """按 task_type 均衡抽 n 题（不足则全取），供 LLM 曲线控成本。"""
    import random
    if n is None or n <= 0 or n >= len(queries):
        return queries
    rng = random.Random(seed)
    by_type = {}
    for q in queries:
        by_type.setdefault(q.get("task_type", "?"), []).append(q)
    per = max(1, n // max(1, len(by_type)))
    out = []
    for lst in by_type.values():
        lst = list(lst)
        rng.shuffle(lst)
        out += lst[:per]
    rng.shuffle(out)
    return out[:n]


def load_answerable(subset_ids, queries, full_by_id):
    """query 可答 = 子集内某金标准的真实效果签名 ⊇ required_effects。"""
    subset_cards = [full_by_id[i] for i in subset_ids if i in full_by_id]
    ans = []
    for q in queries:
        req = set(q["required_effects"])
        binding = q.get("binding")
        if any(common.true_effect_sig(c, full_by_id, binding) >= req for c in subset_cards):
            ans.append(q)
    return ans


def build_method(name, pool):
    if name == "m1":
        from domain_m1 import M1RuleDomain
        return M1RuleDomain(pool), "sel"
    if name == "bm25":
        from retrievers import BM25Retriever
        return BM25Retriever(pool), "ret"
    if name == "dense":
        from retrievers import DenseRetriever
        return DenseRetriever(pool), "ret"
    if name == "b0":
        from llm_baselines import B0LLMFull
        return B0LLMFull(pool), "sel"
    if name == "b3":
        from pipelines import B3SkillFlow
        return B3SkillFlow(pool), "sel"
    if name == "b4":
        from lota_b4 import B4LoTaBench
        batch = int(os.environ.get("B4_BATCH", "16"))
        return B4LoTaBench(pool, batch_size=batch, norm="mean"), "sel"
    if name == "m1h":
        from domain_m1 import M1RuleDomain
        from l0_head import L0HeadParser
        return M1RuleDomain(pool, goal_fn=L0HeadParser(), name="M1-Head"), "sel"
    if name == "m1h4":
        from domain_m1 import M1RuleDomain
        from l0_head import L0HeadParser
        from l4_head import L4HeadScorer
        return M1RuleDomain(pool, goal_fn=L0HeadParser(), l4_scorer=L4HeadScorer(),
                            name="M1-Head+L4"), "sel"
    raise ValueError(name)


def eval_method(obj, kind, queries, by_id, mname=None, tier=None, cache=None, cache_path=None):
    """LLM 选择型(b0/b3)带持久化缓存：每题成功的 chosen 落盘，
    重跑时只对缓存缺失(即上次超时/失败)的题再调 API → 只补超时，不整跑。"""
    hit = 0.0
    n = len(queries)
    wants_binding = getattr(obj, "wants_binding", False)
    use_cache = cache is not None and kind == "sel" and mname in ("b0", "b3", "b4")
    n_reused = n_called = n_failed = 0
    for i, q in enumerate(queries):
        req = q["required_effects"]
        binding = q.get("binding")
        if kind == "ret":
            ranked = obj.rank(q["instruction"])
            hit += M.fullcov_at_k(ranked, req, 1, by_id, binding)
            continue
        ck = f"{mname}|{tier}|{q.get('task_id')}"
        if use_cache and ck in cache:
            chosen = cache[ck]
            n_reused += 1
        else:
            chosen, ok = [], False
            for attempt in range(5):  # LLM API 超时/连接错误重试
                try:
                    chosen = obj.select(q["instruction"], binding) if wants_binding else obj.select(q["instruction"])
                    ok = True
                    break
                except Exception as ex:
                    print(f"    [warn] select 重试{attempt+1}/5: {ex}", flush=True)
                    time.sleep(3 * (attempt + 1))
            if ok:
                n_called += 1
                if use_cache:
                    cache[ck] = chosen
                    if n_called % 10 == 0 and cache_path:  # 增量落盘，防中断丢进度
                        json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
            else:
                n_failed += 1
                print("    [warn] select 5 次仍失败，该题不缓存(下次重跑会补)", flush=True)
        hit += M.selection_scores(chosen, req, by_id, binding)["sel_fullcov"]
    if use_cache and cache_path:
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
    if kind == "sel" and (n_reused or n_failed):
        print(f"    [cache] 复用 {n_reused} / 新调 {n_called} / 仍失败 {n_failed}", flush=True)
    return hit / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["m1", "bm25", "dense"],
                    choices=["m1", "bm25", "dense", "b0", "b3", "b4", "m1h", "m1h4"])
    ap.add_argument("--merge", action="store_true",
                    help="载入已有 out 的 results，只重算本次 methods 并合并（保留其它方法曲线）")
    ap.add_argument("--gt", default=os.path.join(GT_DATA, "gt_kitchen_eval.jsonl"))
    ap.add_argument("--out", default=os.path.join(EVAL_DATA, "confusion_curve.json"))
    ap.add_argument("--subsample", type=int, default=0, help=">0 时按 task_type 均衡抽样控 LLM 成本")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=os.path.join(EVAL_DATA, "confusion_llm_cache.json"),
                    help="b0/b3 每题选择结果缓存；重跑只补缓存缺失(超时)的题")
    args = ap.parse_args()

    cache = {}
    if os.path.exists(args.cache):
        cache = json.load(open(args.cache, encoding="utf-8"))
        print(f"[cache] 载入 {len(cache)} 条已缓存 LLM 选择结果")

    manifest = json.load(open(os.path.join(SKILLS_DATA, "tiers_manifest.json"), encoding="utf-8"))
    subset_ids = manifest["subset_base_ids"]
    queries = common.load_queries(n=None, seed=0, gt_path=args.gt)

    # 用全池 by_id 判定"可答子集"（金标准卡与 tier 内一致）
    os.environ.pop("POOL_FILE", None)
    full_pool = common.load_pool()
    full_by_id = {c["skill_id"]: c for c in full_pool}
    answerable = load_answerable(subset_ids, queries, full_by_id)
    n_ans_full = len(answerable)
    if args.subsample and args.subsample > 0:
        answerable = balanced_subsample(answerable, args.subsample, args.seed)
    print(f"子集基准 {len(subset_ids)} 个；eval 全量 {len(queries)} 题；"
          f"可答子集 {n_ans_full} 题 → 实际评测 {len(answerable)} 题（三档共用）\n")

    results = {m: {} for m in args.methods}
    prev = {}
    if args.merge and os.path.exists(args.out):
        prev = json.load(open(args.out, encoding="utf-8")).get("results", {})
    for tier in TIERS:
        os.environ["POOL_FILE"] = f"pool_tier_{tier}.json"
        pool = common.load_pool()
        by_id = {c["skill_id"]: c for c in pool}
        ndist = sum(1 for c in pool if c.get("is_distractor"))
        print(f"==== tier {tier}：池 {len(pool)} 条（干扰 {ndist}）====")
        for m in args.methods:
            t = time.time()
            obj, kind = build_method(m, pool)
            score = eval_method(obj, kind, answerable, by_id,
                                mname=m, tier=tier, cache=cache, cache_path=args.cache)
            results[m][tier] = round(score, 4)
            print(f"  {m:5} ({kind}) FullCov = {score:.3f}   ({time.time()-t:.1f}s)")
        print()

    # 合并历史方法（--merge 保留其它方法曲线）
    merged = dict(prev)
    merged.update(results)

    # 汇总表
    print("======== 混淆度曲线（FullCov，effect-grounded） ========")
    print(f"{'method':6} {'low(1)':>8} {'mid(3)':>8} {'high(6)':>8} {'掉幅':>8}")
    for m in merged:
        r = merged[m]
        if not all(t in r for t in TIERS):
            continue
        drop = r["low"] - r["high"]
        print(f"{m:6} {r['low']:8.3f} {r['mid']:8.3f} {r['high']:8.3f} {drop:+8.3f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"n_answerable": len(answerable), "n_answerable_full": n_ans_full,
                   "subsample": args.subsample, "seed": args.seed,
                   "methods": sorted(merged), "subset_bases": len(subset_ids),
                   "tiers": {t: manifest["pools"][t] for t in TIERS},
                   "results": merged}, f, ensure_ascii=False, indent=2)
    print(f"\n已写 {args.out}")


if __name__ == "__main__":
    main()
