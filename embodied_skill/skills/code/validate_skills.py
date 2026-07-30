"""技能池校验器：静态校验（+可选仿真校验）复合/变体技能卡。

静态检查：
  1. 结构字段齐全（skill_id 唯一、有 sub_skills 等）。
  2. sub_skills 里引用的 skill 必须是已有的原子技能 skill_id。
  3. 复合技能 effects 的"目标签名"必须 == 其 sub_skills 原子效果的并集（effects 自洽）。
可选仿真检查（--sim）：对可执行卡，在 FloorPlan1 上跑一遍看是否产生声明的目标效果。

用法：
  python validate_skills.py                     # 校验 seed 池（atomic+composite），静态
  python validate_skills.py --extra composite_skills_generated.json confusing_variants_generated.json
  python validate_skills.py --sim --n 10        # 额外抽样跑仿真校验
输出：pool_clean.json（通过校验的复合/变体卡）+ 一份报告到 stdout。
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))          # skills/code/
DATA = os.path.join(HERE, os.pardir, "datasets")           # skills/datasets/
sys.path.insert(0, HERE)
from skill_effects import goal_signature, combine_sequential  # noqa: E402

ATOMIC_FILE = os.path.join(DATA, "atomic_skills.json")
SEED_FILE = os.path.join(DATA, "composite_skills_seed.json")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_atomic_index():
    atoms = load_json(ATOMIC_FILE)
    return {a["skill_id"]: a for a in atoms}


def static_check(card, atomic_index, seen_ids):
    """返回 (ok, problems:list)。"""
    problems = []
    sid = card.get("skill_id")
    if not sid:
        return False, ["缺 skill_id"]
    if sid in seen_ids:
        problems.append(f"skill_id 重复: {sid}")
    for field in ["name", "description", "type", "sub_skills", "effects"]:
        if field not in card:
            problems.append(f"缺字段: {field}")
    subs = card.get("sub_skills", [])
    if not subs:
        problems.append("sub_skills 为空")
    # 2. 子技能引用必须存在于原子技能；按执行顺序收集各步目标 token（热态覆盖）
    step_sigs = []
    for st in subs:
        sk = st.get("skill")
        if sk not in atomic_index:
            problems.append(f"引用了不存在的原子技能: {sk}")
        else:
            step_sigs.append(goal_signature(atomic_index[sk]["effects"]))
    union_goal = combine_sequential(step_sigs)
    # 干扰项(变体)：本就故意不自洽(错序/缺步/近义)，跳过 effects 自洽检查，
    # 但必须带齐干扰项元数据；其 effects 反映"实际能达成什么"，不做 ==union 约束。
    if card.get("is_distractor"):
        for f in ["variant_of", "variant_type", "why_wrong"]:
            if f not in card:
                problems.append(f"干扰项缺字段: {f}")
        return (len(problems) == 0), problems
    # 3. 金标准/正例：effects 自洽，声明的目标签名 == 子技能并集
    declared = goal_signature(card.get("effects", []))
    if declared != union_goal:
        problems.append(f"effects 不自洽: 声明目标={sorted(declared)} vs 子技能并集={sorted(union_goal)}")
    return (len(problems) == 0), problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="额外要校验的卡文件(LLM生成的)")
    ap.add_argument("--sim", action="store_true", help="对可执行卡额外做仿真校验")
    ap.add_argument("--n", type=int, default=8, help="仿真抽样数量")
    ap.add_argument("--out", default=os.path.join(DATA, "pool_clean.json"))
    args = ap.parse_args()

    atomic_index = build_atomic_index()
    print(f"原子技能 {len(atomic_index)} 个（作为 sub_skills 合法引用集）")

    cards = load_json(SEED_FILE)
    for extra in args.extra:
        p = extra if os.path.isabs(extra) else os.path.join(DATA, extra)
        if os.path.exists(p):
            cards += load_json(p)
            print(f"载入额外卡: {p}")
        else:
            print(f"[跳过] 找不到 {p}")

    seen = set()
    passed, failed = [], []
    for c in cards:
        ok, probs = static_check(c, atomic_index, seen)
        seen.add(c.get("skill_id"))
        if ok:
            passed.append(c)
        else:
            failed.append((c.get("skill_id"), probs))

    print(f"\n== 静态校验：通过 {len(passed)} / 失败 {len(failed)} ==")
    for sid, probs in failed:
        print(f"  ✗ {sid}: {probs}")

    # 可选仿真校验
    if args.sim:
        sim_check(passed[: args.n])

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(passed, f, ensure_ascii=False, indent=2)
    print(f"\n通过校验的卡已写出: {args.out}（{len(passed)} 条）")


def sim_check(cards):
    """对可执行卡在 FloorPlan1 上跑一遍，看是否产生声明的目标效果。"""
    import composite_bodies as cb
    import atomic_bodies as ab
    from ai2thor.controller import Controller
    from ai2thor.platform import Linux64

    print(f"\n== 仿真校验（{len(cards)} 条，FloorPlan1） ==")
    ctrl = Controller(platform=Linux64, scene="FloorPlan1", width=300, height=300, x_display="0")
    n_pass = n_fail = 0
    for c in cards:
        if not c.get("is_executable", False):
            print(f"  - {c['skill_id']}: 非可执行，跳过")
            continue
        bindings = _pick_bindings(c)
        ctrl.reset("FloorPlan1")
        _, ok, log = cb.execute_composite(ctrl, c, bindings)
        n_pass += int(ok)
        n_fail += int(not ok)
        print(f"  {'✓' if ok else '✗'} {c['skill_id']}: exec_ok={ok} bind={bindings}"
              + ("" if ok else f" 失败步={[s for s in log if s[2] is False]}"))
    print(f"== 仿真汇总：通过 {n_pass} / 失败 {n_fail} ==")
    ctrl.stop()


def _pick_bindings(card):
    """按技能实际用到的子技能，智能选择绑定对象，避免用错物体类型导致'假失败'。
    - 灌液/倒液(fill/empty) → 需可盛液体的物体(Mug)
    - 默认 → 可拿可切的食材(Apple)；receptacle 用中性的 CounterTop(温度可保持)
    """
    subs = {st.get("skill") for st in card.get("sub_skills", [])}
    if subs & {"fill_with_liquid", "empty_liquid"}:
        obj = "Mug"
    else:
        obj = "Apple"
    # liquid 占位符绑定成合法液体类型；receptacle 用中性 CounterTop(温度可保持)
    return {"obj": obj, "receptacle": "CounterTop", "liquid": "water"}


if __name__ == "__main__":
    main()
