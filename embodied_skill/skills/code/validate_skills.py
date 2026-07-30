"""Skill-pool validator: static validation (+ optional simulation validation) of
composite/variant skill cards.

Static checks:
  1. Structural fields are complete (skill_id unique, has sub_skills, etc.).
  2. Every skill referenced in sub_skills must be an existing atomic skill_id.
  3. A composite skill's effects "goal signature" must == the union of the atomic
     effects of its sub_skills (effects self-consistency).
Optional simulation check (--sim): for executable cards, run once on FloorPlan1
to see whether the declared goal effects are produced.

Usage:
  python validate_skills.py                     # validate the seed pool (atomic+composite), static
  python validate_skills.py --extra composite_skills_generated.json confusing_variants_generated.json
  python validate_skills.py --sim --n 10        # additionally run simulation validation on a sample
Output: pool_clean.json (composite/variant cards that pass) + a report to stdout.
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
    """Return (ok, problems:list)."""
    problems = []
    sid = card.get("skill_id")
    if not sid:
        return False, ["missing skill_id"]
    if sid in seen_ids:
        problems.append(f"duplicate skill_id: {sid}")
    for field in ["name", "description", "type", "sub_skills", "effects"]:
        if field not in card:
            problems.append(f"missing field: {field}")
    subs = card.get("sub_skills", [])
    if not subs:
        problems.append("sub_skills is empty")
    # 2. Sub-skill references must exist in the atomic skills; collect each step's
    #    goal tokens in execution order (thermal override).
    step_sigs = []
    for st in subs:
        sk = st.get("skill")
        if sk not in atomic_index:
            problems.append(f"references a nonexistent atomic skill: {sk}")
        else:
            step_sigs.append(goal_signature(atomic_index[sk]["effects"]))
    union_goal = combine_sequential(step_sigs)
    # Distractors (variants): intentionally not self-consistent (reorder/missing/
    # lookalike), so skip the effects self-consistency check, but they must carry
    # full distractor metadata; their effects reflect "what can actually be
    # achieved" and are not constrained to == union.
    if card.get("is_distractor"):
        for f in ["variant_of", "variant_type", "why_wrong"]:
            if f not in card:
                problems.append(f"distractor missing field: {f}")
        return (len(problems) == 0), problems
    # 3. Gold/positive: effects self-consistent, declared goal signature == union of sub-skills
    declared = goal_signature(card.get("effects", []))
    if declared != union_goal:
        problems.append(f"effects not self-consistent: declared goal={sorted(declared)} vs sub-skill union={sorted(union_goal)}")
    return (len(problems) == 0), problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="extra card files to validate (LLM-generated)")
    ap.add_argument("--sim", action="store_true", help="additionally run simulation validation on executable cards")
    ap.add_argument("--n", type=int, default=8, help="simulation sample size")
    ap.add_argument("--out", default=os.path.join(DATA, "pool_clean.json"))
    args = ap.parse_args()

    atomic_index = build_atomic_index()
    print(f"{len(atomic_index)} atomic skills (the legal reference set for sub_skills)")

    cards = load_json(SEED_FILE)
    for extra in args.extra:
        p = extra if os.path.isabs(extra) else os.path.join(DATA, extra)
        if os.path.exists(p):
            cards += load_json(p)
            print(f"loaded extra cards: {p}")
        else:
            print(f"[skip] not found {p}")

    seen = set()
    passed, failed = [], []
    for c in cards:
        ok, probs = static_check(c, atomic_index, seen)
        seen.add(c.get("skill_id"))
        if ok:
            passed.append(c)
        else:
            failed.append((c.get("skill_id"), probs))

    print(f"\n== Static validation: passed {len(passed)} / failed {len(failed)} ==")
    for sid, probs in failed:
        print(f"  ✗ {sid}: {probs}")

    # Optional simulation validation
    if args.sim:
        sim_check(passed[: args.n])

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(passed, f, ensure_ascii=False, indent=2)
    print(f"\nValidated cards written to: {args.out} ({len(passed)} cards)")


def sim_check(cards):
    """Run executable cards once on FloorPlan1 to see whether the declared goal effects are produced."""
    import composite_bodies as cb
    import atomic_bodies as ab
    from ai2thor.controller import Controller
    from ai2thor.platform import Linux64

    print(f"\n== Simulation validation ({len(cards)} cards, FloorPlan1) ==")
    ctrl = Controller(platform=Linux64, scene="FloorPlan1", width=300, height=300, x_display="0")
    n_pass = n_fail = 0
    for c in cards:
        if not c.get("is_executable", False):
            print(f"  - {c['skill_id']}: not executable, skipped")
            continue
        bindings = _pick_bindings(c)
        ctrl.reset("FloorPlan1")
        _, ok, log = cb.execute_composite(ctrl, c, bindings)
        n_pass += int(ok)
        n_fail += int(not ok)
        print(f"  {'✓' if ok else '✗'} {c['skill_id']}: exec_ok={ok} bind={bindings}"
              + ("" if ok else f" failed_steps={[s for s in log if s[2] is False]}"))
    print(f"== Simulation summary: passed {n_pass} / failed {n_fail} ==")
    ctrl.stop()


def _pick_bindings(card):
    """Smartly choose binding objects based on the sub-skills actually used, to
    avoid 'false failures' from using the wrong object type.
    - fill/empty liquid -> needs an object that can hold liquid (Mug)
    - default -> a pickupable, sliceable food (Apple); receptacle uses a neutral
      CounterTop (temperature is preserved)
    """
    subs = {st.get("skill") for st in card.get("sub_skills", [])}
    if subs & {"fill_with_liquid", "empty_liquid"}:
        obj = "Mug"
    else:
        obj = "Apple"
    # Bind the liquid placeholder to a valid liquid type; receptacle uses a neutral CounterTop (temperature preserved)
    return {"obj": obj, "receptacle": "CounterTop", "liquid": "water"}


if __name__ == "__main__":
    main()
