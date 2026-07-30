"""WP8 产出2/产出4：按任务子集/seen-unseen 拆表 + 变体类型归因。

输入：隔离仿真结果 sim_results.json（per_query: SR/GC/chosen）
     + gt_kitchen_eval.jsonl（task_id→split）
     + pool_clean.json（skill_id→variant_type，供误选归因）
纯离线，不跑仿真/不占 GPU。

用法：python eval/code/wp8_breakdown.py
"""
import os
import sys
import json
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ROUTER_CODE = os.path.join(ROOT, "router", "code")
GT_DATA = os.path.join(ROOT, "gt", "datasets")
SKILLS_DATA = os.path.join(ROOT, "skills", "datasets")
EVAL_DATA = os.path.join(ROOT, "eval", "datasets")
sys.path.insert(0, ROUTER_CODE)
import common  # noqa: E402

# 单效果 vs 组合子集
SINGLE = {"pick_and_place_simple"}
METHOD_ORDER = ["gold", "M1-Rule", "M1-Head", "M1-Head+L4", "B0-LLM-Full",
                "B3-SkillFlow", "B2-Dense", "B4-LoTa", "B1-BM25"]


def subset_of(task_type):
    return "single" if task_type in SINGLE else "composite"


def agg(rows, keyfn):
    """rows: [(group, method, SR, GC)]；返回 {group:{method:{n,sr,gc}}}。"""
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0.0]))  # n, sr_hits, gc_sum
    for g, m, sr, gc in rows:
        a = acc[g][m]
        a[0] += 1
        a[1] += 1 if sr else 0
        a[2] += gc
    out = {}
    for g, md in acc.items():
        out[g] = {m: {"n": v[0], "sr": v[1] / v[0], "gc": v[2] / v[0]} for m, v in md.items()}
    return out


def print_table(title, table, groups):
    print(f"\n======== {title} ========")
    hdr = f"{'method':13}" + "".join(f"{g+' SR':>14}" for g in groups)
    print(hdr)
    for m in METHOD_ORDER:
        cells = []
        for g in groups:
            v = table.get(g, {}).get(m)
            cells.append(f"{v['sr']*100:6.1f}%(n{v['n']})" if v else f"{'-':>14}")
        print(f"{m:13}" + "".join(f"{c:>14}" for c in cells))
    print(f"{'':13}" + "".join(f"{'(GC)':>14}" for _ in groups))
    for m in METHOD_ORDER:
        cells = []
        for g in groups:
            v = table.get(g, {}).get(m)
            cells.append(f"{v['gc']*100:6.1f}%" if v else f"{'-':>14}")
        print(f"{m:13}" + "".join(f"{c:>14}" for c in cells))


def main():
    sim = json.load(open(os.path.join(EVAL_DATA, "sim_results.json"), encoding="utf-8"))
    pq = sim["per_query"]

    # task_id -> {split, required_effects, binding}
    split_of, req_of, bind_of = {}, {}, {}
    with open(os.path.join(GT_DATA, "gt_kitchen_eval.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            split_of[r["task_id"]] = r.get("split", "?")
            req_of[r["task_id"]] = set(r.get("required_effects", []))
            bind_of[r["task_id"]] = r.get("binding")

    # skill_id -> variant_type；全池 by_id 供真实效果判定
    pool = json.load(open(os.path.join(SKILLS_DATA, "pool_clean.json"), encoding="utf-8"))
    atomics = json.load(open(os.path.join(SKILLS_DATA, "atomic_skills.json"), encoding="utf-8"))
    by_id = {c["skill_id"]: c for c in pool}
    for c in atomics:
        by_id.setdefault(c["skill_id"], c)
    vtype = {c["skill_id"]: c.get("variant_type") for c in pool if c.get("is_distractor")}

    rows_subset, rows_split, rows_type = [], [], []
    for q in pq:
        tt = q["task_type"]
        sub = subset_of(tt)
        sp = split_of.get(q["task_id"], "?")
        for m, r in q["res"].items():
            rows_subset.append((sub, m, r["SR"], r["GC"]))
            rows_split.append((sp, m, r["SR"], r["GC"]))
            rows_type.append((tt, m, r["SR"], r["GC"]))

    print_table("产出2：按任务子集拆表（single=单效果 / composite=组合）",
                agg(rows_subset, None), ["single", "composite"])
    print_table("产出2b：按 seen/unseen 拆表",
                agg(rows_split, None), sorted({r[0] for r in rows_split}))
    print_table("附：按 task_type 细分",
                agg(rows_type, None), sorted({t for t, _, _, _ in rows_type}))

    # 产出4：误选归因——只统计"真被骗"的干扰项（其真实效果签名不覆盖任务目标）。
    # 排除"合法 reorder 变体但效果等价"这类被 M1 有意选中且实际正确的情况，避免冤枉。
    def is_lured(sid, tid):
        """选到的是干扰项，且它的真实效果签名不覆盖任务 required_effects → 真被骗。"""
        if sid not in vtype:
            return False
        card = by_id.get(sid)
        if not card:
            return True
        sig = common.true_effect_sig(card, by_id, bind_of.get(tid))
        return not (sig >= req_of.get(tid, set()))

    print("\n======== 产出4：真被骗干扰项类型归因（真实效果不覆盖目标才计）========")
    print("含义：方法被李鬼骗到（选了真实效果不达标的干扰项）按类型计数；对应 M1 哪个钩子拦住")
    print(f"{'method':13}{'reorder':>10}{'missing':>10}{'lookalike':>12}{'被骗题数':>12}{'总选干扰':>10}")
    for m in METHOD_ORDER:
        cnt = defaultdict(int)
        n_lured_q = 0
        n_any_distractor = 0
        for q in pq:
            r = q["res"].get(m)
            if not r:
                continue
            tid = q["task_id"]
            lured_here = False
            for sid in r.get("chosen", []):
                if sid in vtype:
                    n_any_distractor += 1
                if is_lured(sid, tid):
                    cnt[vtype.get(sid) or "?"] += 1
                    lured_here = True
            if lured_here:
                n_lured_q += 1
        print(f"{m:13}{cnt['reorder']:>10}{cnt['missing']:>10}{cnt['lookalike']:>12}"
              f"{n_lured_q:>12}{n_any_distractor:>10}")
    print("\n钩子对应：reorder/missing←合法性过滤（顺序/前置条件）；lookalike←真实效果覆盖（换错动作/容器）")
    print("说明：M1 选到的合法 reorder 变体若真实效果等价则不计入（不算被骗）。")

    # 存 JSON 供画图/写作
    out = {
        "by_subset": agg(rows_subset, None),
        "by_split": agg(rows_split, None),
        "by_task_type": agg(rows_type, None),
    }
    with open(os.path.join(EVAL_DATA, "wp8_breakdown.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已写 {os.path.join(EVAL_DATA, 'wp8_breakdown.json')}")


if __name__ == "__main__":
    main()
