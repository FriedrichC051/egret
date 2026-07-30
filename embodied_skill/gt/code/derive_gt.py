"""Derive routing ground truth (gold-standard skill set + effect signatures) from ALFRED tasks.

For each task:
  1. From task_type + pddl_params, derive the "target effects that must be achieved",
     required_effects (role-based: obj/recep).
  2. Run set-cover over the skill pool to give two equivalent minimal gold answers:
       - gt_composite_set: composite skills allowed (usually 1~2 skills)
       - gt_atomic_set   : atomic skills only (more skills)
  3. Correctness criterion at eval time: the union of the selected skills' target effects
     must be a superset of required_effects (equivalent coverage counts as correct).

Usage:
  python gt/code/derive_gt.py                       # full set
  python gt/code/derive_gt.py --tasks tasks_kitchen_sample.jsonl --example
Output: embodied_skill/gt/datasets/gt_kitchen.jsonl
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))          # gt/code/
ROOT = os.path.dirname(os.path.dirname(HERE))              # embodied_skill/
SKILLS_CODE = os.path.join(ROOT, "skills", "code")         # importable modules
SKILLS_DATA = os.path.join(ROOT, "skills", "datasets")     # skill-card JSON
TASKS_DATA = os.path.join(ROOT, "data", "datasets")        # task jsonl
GT_DATA = os.path.join(HERE, os.pardir, "datasets")        # gt/datasets/ output
sys.path.insert(0, SKILLS_CODE)
from skill_effects import goal_signature  # noqa: E402


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_pool():
    atoms = load_json(os.path.join(SKILLS_DATA, "atomic_skills.json"))
    comps = load_json(os.path.join(SKILLS_DATA, "composite_skills_seed.json"))
    # Append validated LLM-generated cards (if present)
    clean = os.path.join(SKILLS_DATA, "pool_clean.json")
    extra = load_json(clean) if os.path.exists(clean) else []
    return atoms, comps, extra


def task_required_effects(task):
    """Derive the target-effect token set + role binding from task_type + pddl_params."""
    tt = task["task_type"]
    p = task.get("pddl_params", {})
    eff = {"at_receptacle"}  # always place the target object into parent_target
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
    """candidates: list of (skill_id, goal_sig set). Greedy minimal cover.
    Preference: cover more per step; break ties toward fewer skills. Returns (chosen_ids, covered?, covered_set)."""
    remaining = set(required)
    chosen = []
    # only consider skills that have target effects and intersect required
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
    ap.add_argument("--example", action="store_true", help="also print one heat+slice task in detail")
    args = ap.parse_args()

    atoms, comps, extra = load_pool()
    # The GT gold standard may only be covered by positive skills, never distractors
    # (otherwise the gold set gets contaminated and GT credibility breaks).
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

    # Statistics
    from collections import Counter
    comp_sizes = Counter(len(r["gt_composite_set"]) for r in rows)
    atom_sizes = Counter(len(r["gt_atomic_set"]) for r in rows)
    print(f"== GT derivation done: {len(rows)} tasks -> {args.out} ==")
    print("tasks with incomplete cover:", incomplete)
    print("gt_composite_set size distribution:", dict(sorted(comp_sizes.items())))
    print("gt_atomic_set size distribution:", dict(sorted(atom_sizes.items())))

    if args.example and example_row:
        r, t = example_row
        print("\n================ Full GT example for one real task ================")
        print("task_id:", r["task_id"], "| scene:", r["floor_plan"])
        print("instruction:", t["instruction"])
        print("task_type:", r["task_type"], "| pddl_params:", t["pddl_params"])
        seq = " -> ".join(a["action"] for a in t["high_pddl_actions"])
        print("ALFRED subgoal sequence:", seq)
        print("-> derived required_effects (must achieve):", r["required_effects"])
        print("  binding obj=%s, receptacle=%s" % (r["binding"]["obj"], r["binding"]["receptacle"]))
        print("-> gold standard (composite) gt_composite_set:", r["gt_composite_set"])
        for sid in r["gt_composite_set"]:
            if sid in comp_index:
                print("     -", sid, "effects=", comp_index[sid]["effects"])
        print("-> gold standard (atomic) gt_atomic_set:", r["gt_atomic_set"])
        print("  Both cover required_effects -> equivalent under set-cover; the router giving either counts as correct.")


if __name__ == "__main__":
    main()
