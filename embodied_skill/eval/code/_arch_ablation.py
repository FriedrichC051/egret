"""架构留一法消融（回应"到底是哪层/哪组件在起作用"）：
从完整旗舰 M1-Neuro 出发，每次只敲掉一个组件，在**主评测集**上看离线指标掉多少。
对比"薄学习件消融（M1-Head/M1-Rule，仅动 L0头 / L2③）"vs"规则骨架消融（L2① 效果重算 / L2② set-cover 兜底 / L1 合法）"，
用来证明：真正的大杠杆是 L2 的规则效果落地 + set-cover，而非那两个薄学习件。

离线口径：effect-grounded Sel-Hit / Sel-FullCov / 平均选中数（主池、主评测 GT）。
用法：
  python eval/code/_arch_ablation.py                     # 只算离线
  python eval/code/_arch_ablation.py --dump              # 额外把各变体 chosen 写成 selections_arch.json（供 WSL 仿真）
"""
import os
import sys
import time
import json
import argparse
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ROUTER_CODE = os.path.join(ROOT, "router", "code")
GT_DATA = os.path.join(ROOT, "gt", "datasets")
EVAL_DATA = os.path.join(ROOT, "eval", "datasets")
sys.path.insert(0, ROUTER_CODE)
sys.path.insert(0, HERE)
import common  # noqa: E402
import metrics as M  # noqa: E402


def build_variants(pool):
    from domain_m1 import M1RuleDomain
    from retrievers import DenseRetriever
    L0 = L4 = None
    try:
        from l0_head import L0HeadParser
        L0 = L0HeadParser()
    except Exception as ex:
        print(f"[warn] L0HeadParser 不可用，L0 退回规则：{ex}")
    try:
        from l4_head import L4HeadScorer
        L4 = L4HeadScorer()
    except Exception as ex:
        print(f"[warn] L4HeadScorer 不可用：{ex}")

    def m1(**kw):
        return M1RuleDomain(pool, goal_fn=L0, l4_scorer=L4, **kw)

    V = []
    # (label, obj, kind, dump_key) —— dump_key 用作 selections_arch.json 里的方法名
    # 完整旗舰
    V.append(("M1-Neuro (full)", m1(name="full"), "sel", "arch-full"))
    # —— 薄学习件消融（现有主线，改动很小）——
    V.append(("  -L2③ 判别tie-break (=M1-Head)",
              M1RuleDomain(pool, goal_fn=L0, l4_scorer=None, name="head"), "sel", "arch-head"))
    V.append(("  -L0头 (=M1-Rule, 规则解析)",
              M1RuleDomain(pool, goal_fn=None, l4_scorer=None, name="rule"), "sel", "arch-rule"))
    # —— 规则骨架消融（本次新增，验证大杠杆）——
    V.append(("  -L2① 效果重算 (信声明effects)",
              m1(use_declared=True, name="egoff"), "sel", "arch-egoff"))
    V.append(("  -L2② set-cover/兜底 (纯神经选集)",
              m1(no_repair=True, name="norepair"), "sel", "arch-norepair"))
    V.append(("  -L1 合法性过滤",
              m1(no_legality=True, name="nolegal"), "sel", "arch-nolegal"))
    # —— 全拆到无结构端点 ——
    V.append(("  -全规则 (信声明+无兜底+无合法, 近裸神经)",
              m1(use_declared=True, no_repair=True, no_legality=True, name="naked"), "sel", "arch-naked"))
    V.append(("Dense top-1 (无结构基线)", DenseRetriever(pool), "ret", "arch-dense1"))
    return V


def q_meta(q):
    """selections_arch.json 里每题的公共信息（与 run_pilot.q_meta 对齐，供仿真消费）。"""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true",
                    help="额外把各变体 chosen 写成 selections_arch.json（供 WSL 仿真）")
    ap.add_argument("--out", default=os.path.join(EVAL_DATA, "selections_arch.json"))
    args = ap.parse_args()

    os.environ.pop("POOL_FILE", None)          # 主池（pool_clean）
    pool = common.load_pool()
    by_id = {c["skill_id"]: c for c in pool}
    queries = common.load_queries(
        n=None, seed=0, gt_path=os.path.join(GT_DATA, "gt_kitchen_eval.jsonl"))
    print(f"主池 {len(pool)} 条；主评测 {len(queries)} 题\n")

    # 仿真用：每题的公共信息 + 各方法 chosen_ids
    entries = {q["task_id"]: {**q_meta(q), "methods": {}} for q in queries}

    rows = []
    for label, obj, kind, key in build_variants(pool):
        t = time.time()
        hit = fc = size = 0.0
        for q in queries:
            req = q["required_effects"]
            binding = q.get("binding")
            if kind == "ret":
                ranked = obj.rank(q["instruction"])
                chosen = ranked[:1]
            else:
                chosen = obj.select(q["instruction"], binding)
            s = M.selection_scores(chosen, req, by_id, binding)
            hit += s["sel_hit"]; fc += s["sel_fullcov"]; size += s["sel_size"]
            if args.dump:
                entries[q["task_id"]]["methods"][key] = {
                    "ranked_ids": [], "chosen_ids": [c["skill_id"] for c in chosen]}
        n = len(queries)
        rows.append((label, hit / n, fc / n, size / n))
        print(f"{label:44} Sel-Hit={hit/n:.3f}  Sel-FullCov={fc/n:.3f}  "
              f"avg选中={size/n:5.1f}  ({time.time()-t:.1f}s)")

    print("\n===== 架构留一法消融（主评测，effect-grounded 离线） =====")
    print(f"{'变体':44} {'Sel-Hit':>9} {'Sel-FullCov':>12} {'avg选中':>8}")
    base_hit = rows[0][1]; base_fc = rows[0][2]
    for label, h, f, s in rows:
        dh = h - base_hit; df = f - base_fc
        print(f"{label:44} {h:9.3f} {f:12.3f} {s:8.1f}   ΔHit={dh:+.3f} ΔFC={df:+.3f}")

    if args.dump:
        os.makedirs(EVAL_DATA, exist_ok=True)
        out = {
            "meta": {"n": -1, "seed": 0, "pool_size": len(pool),
                     "note": "arch leave-one-out ablation variants for simulation",
                     "generated_at": datetime.now().isoformat(timespec="seconds")},
            "queries": [entries[q["task_id"]] for q in queries],
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        methods = sorted({m for e in out["queries"] for m in e["methods"]})
        print(f"\n已写 {args.out}\n  query {len(out['queries'])} 条；方法：{methods}")


if __name__ == "__main__":
    main()
