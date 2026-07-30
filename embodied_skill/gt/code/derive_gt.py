"""Phase 2：从 ALFRED 任务派生路由 Ground-Truth（金标准技能集 + 效果签名）。

对每道题：
  1. 由 task_type + pddl_params 推出"必须达成的目标效果" required_effects（角色化：obj/recep）。
  2. 在技能池上做 set-cover，给出两种等价的最简金标准答案：
       - gt_composite_set：允许用复合技能（通常 1~2 个技能）
       - gt_atomic_set   ：只用原子技能（多技能）
  3. 评测时判正确的准则：路由器选中技能的目标效果并集 ⊇ required_effects（等价即算对）。

用法：
  python gt/code/derive_gt.py                       # 全量 1149 题
  python gt/code/derive_gt.py --tasks tasks_kitchen_sample.jsonl --example
输出：embodied_skill/gt/datasets/gt_kitchen.jsonl
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))          # gt/code/
ROOT = os.path.dirname(os.path.dirname(HERE))              # embodied_skill/
SKILLS_CODE = os.path.join(ROOT, "skills", "code")         # 可导入模块
SKILLS_DATA = os.path.join(ROOT, "skills", "datasets")     # 技能卡 JSON
TASKS_DATA = os.path.join(ROOT, "data", "datasets")        # 任务 jsonl
GT_DATA = os.path.join(HERE, os.pardir, "datasets")        # gt/datasets/ 输出
sys.path.insert(0, SKILLS_CODE)
from skill_effects import goal_signature  # noqa: E402


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_pool():
    atoms = load_json(os.path.join(SKILLS_DATA, "atomic_skills.json"))
    comps = load_json(os.path.join(SKILLS_DATA, "composite_skills_seed.json"))
    # 追加已通过校验的 LLM 生成卡（若存在）
    clean = os.path.join(SKILLS_DATA, "pool_clean.json")
    extra = load_json(clean) if os.path.exists(clean) else []
    return atoms, comps, extra


def task_required_effects(task):
    """由 task_type + pddl_params 推出目标效果 token 集合 + 角色绑定。"""
    tt = task["task_type"]
    p = task.get("pddl_params", {})
    eff = {"at_receptacle"}  # 都要把目标物放进 parent_target
    if "heat" in tt:
        eff.add("isHot")
    if "cool" in tt:
        eff.add("isCold")
    if "clean" in tt:
        eff.add("isClean")
    if p.get("object_sliced"):
        eff.add("isSliced")
    binding = {"obj": p.get("object_target"), "receptacle": p.get("parent_target")}
    return eff, binding


def greedy_cover(required, candidates):
    """candidates: list of (skill_id, goal_sig set)。贪心最小覆盖。
    偏好：单步覆盖越多越好；平局时技能数更少。返回 (chosen_ids, covered?, covered_set)。"""
    remaining = set(required)
    chosen = []
    # 只考虑有目标效果、且与 required 有交集的技能
    cands = [(sid, sig) for sid, sig in candidates if sig & required]
    while remaining:
        best = None
        best_gain = 0
        for sid, sig in cands:
            gain = len(sig & remaining)
            if gain > best_gain:
                best_gain = gain
                best = (sid, sig)
        if best is None:
            break
        chosen.append(best[0])
        remaining -= best[1]
    return chosen, (len(remaining) == 0), (set(required) - remaining)


def build_candidates(skills):
    return [(s["skill_id"], goal_signature(s.get("effects", []))) for s in skills]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tasks_kitchen.jsonl")
    ap.add_argument("--out", default=os.path.join(GT_DATA, "gt_kitchen.jsonl"))
    ap.add_argument("--example", action="store_true", help="额外详细打印一道加热切片题")
    args = ap.parse_args()

    atoms, comps, extra = load_pool()
    # GT 金标准只能由"正例技能"覆盖，绝不能选中干扰项（否则金标准被李鬼污染，破坏 GT 可信度）。
    gold_extra = [s for s in extra if not s.get("is_distractor")]
    atomic_cands = build_candidates(atoms)
    full_cands = build_candidates(atoms + comps + gold_extra)
    comp_index = {s["skill_id"]: s for s in (comps + gold_extra)}

    tasks_path = args.tasks if os.path.isabs(args.tasks) else os.path.join(TASKS_DATA, args.tasks)
    rows = []
    incomplete = 0
    example_row = None
    with open(tasks_path, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            req, binding = task_required_effects(t)
            gt_comp, ok_c, _ = greedy_cover(req, full_cands)
            gt_atom, ok_a, _ = greedy_cover(req, atomic_cands)
            if not (ok_c and ok_a):
                incomplete += 1
            row = {
                "task_id": t["task_id"],
                "task_type": t["task_type"],
                "split": t.get("split"),
                "floor_plan": t.get("floor_plan"),
                "instruction": t.get("instruction"),
                "binding": binding,
                "required_effects": sorted(req),
                "gt_composite_set": gt_comp,
                "gt_atomic_set": gt_atom,
                "cover_ok": bool(ok_c and ok_a),
            }
            rows.append(row)
            if args.example and example_row is None and "isSliced" in req and "isHot" in req:
                example_row = (row, t)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计
    from collections import Counter
    comp_sizes = Counter(len(r["gt_composite_set"]) for r in rows)
    atom_sizes = Counter(len(r["gt_atomic_set"]) for r in rows)
    print(f"== Phase2 GT 派生完成：{len(rows)} 题 -> {args.out} ==")
    print("cover 不完整的题:", incomplete)
    print("gt_composite_set 大小分布:", dict(sorted(comp_sizes.items())))
    print("gt_atomic_set 大小分布:", dict(sorted(atom_sizes.items())))

    if args.example and example_row:
        r, t = example_row
        print("\n================ 一道真实题的完整 GT 例子 ================")
        print("task_id:", r["task_id"], "| 场景:", r["floor_plan"])
        print("任务指令:", t["instruction"])
        print("task_type:", r["task_type"], "| pddl_params:", t["pddl_params"])
        seq = " -> ".join(a["action"] for a in t["high_pddl_actions"])
        print("ALFRED 子目标序列:", seq)
        print("→ 派生的 required_effects(必须达成):", r["required_effects"])
        print("  绑定 obj=%s, receptacle=%s" % (r["binding"]["obj"], r["binding"]["receptacle"]))
        print("→ 金标准(复合技能版) gt_composite_set:", r["gt_composite_set"])
        for sid in r["gt_composite_set"]:
            if sid in comp_index:
                print("     -", sid, "effects=", comp_index[sid]["effects"])
        print("→ 金标准(原子技能版) gt_atomic_set:", r["gt_atomic_set"])
        print("  两者都覆盖 required_effects → set-cover 意义下等价，路由器给出任一个都算对。")


if __name__ == "__main__":
    main()
