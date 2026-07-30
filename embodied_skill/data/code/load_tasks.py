"""Data loading: filter "kitchen + selected task types" from ALFRED-lite trajectory JSON,
distill the fields we need (including the high_pddl subgoal sequence = raw material for GT),
and write them out as compact jsonl.

Usage (run under WSL; download ALFRED first, see README):
    python embodied_skill/data/code/load_tasks.py

Input:  ~/alfred_data/json_2.1.0/{train,valid_seen,valid_unseen}/.../traj_data.json
Output: embodied_skill/data/datasets/tasks_kitchen.jsonl        (all matching tasks)
        embodied_skill/data/datasets/tasks_kitchen_sample.jsonl (balanced sample ~100 tasks)
"""
import os
import json
import glob
import random
from collections import Counter, defaultdict

ALFRED_ROOT = os.path.expanduser("~/alfred_data/json_2.1.0")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
# Task types kept, all within the effect-signature GT's reliable support:
#   pick_and_place_simple          -> at_receptacle
#   pick_heat_then_place_in_recep  -> isHot + at_receptacle
#   pick_cool_then_place_in_recep  -> isCold + at_receptacle
#   pick_clean_then_place_in_recep -> isClean + at_receptacle
# (slice is a modifier: pddl_params.object_sliced triggers isSliced, stackable on the above)
# Explicitly excluded (not covered by the GT framework):
#   pick_two_obj_and_place          -> needs object-instance identification/counting (LoTa-Bench also excludes it)
#   pick_and_place_with_movable_recep -> "put into a movable receptacle then place the whole thing" double placement; at_receptacle cannot faithfully describe it
#   look_at_obj_in_light            -> examine has no corresponding effect token (and does not occur in kitchen scenes)
WANT_TYPES = {
    "pick_and_place_simple",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
}
# AI2-THOR kitchen scenes are numbered 1~30
def is_kitchen(scene_num):
    return scene_num is not None and 1 <= int(scene_num) <= 30

SAMPLE_PER_TYPE = 50  # how many per type (for the quick sample set)


def load_one(traj_path):
    try:
        with open(traj_path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        return None
    tt = d.get("task_type")
    scene = d.get("scene", {}).get("scene_num")
    if tt not in WANT_TYPES or not is_kitchen(scene):
        return None
    high = []
    for h in d.get("plan", {}).get("high_pddl", []):
        act = h.get("discrete_action", {})
        high.append({"action": act.get("action"), "args": act.get("args", [])})
    anns = d.get("turk_annotations", {}).get("anns", [])
    instruction = anns[0].get("task_desc", "").strip() if anns else ""
    step_instr = anns[0].get("high_descs", []) if anns else []
    return {
        "task_id": d.get("task_id"),
        "task_type": tt,
        "scene_num": int(scene),
        "floor_plan": d.get("scene", {}).get("floor_plan"),
        "pddl_params": d.get("pddl_params", {}),
        "high_pddl_actions": high,     # subgoal action sequence (raw material for GT)
        "instruction": instruction,     # task-level natural language (the router's query)
        "step_instructions": step_instr,
        "source_path": traj_path,
    }


def main():
    if not os.path.isdir(ALFRED_ROOT):
        raise SystemExit(f"Data directory {ALFRED_ROOT} not found; run fetch_alfred_lite.sh to download and extract it first.")
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    for split in ["train", "valid_seen", "valid_unseen"]:
        pattern = os.path.join(ALFRED_ROOT, split, "*", "*", "traj_data.json")
        for tj in glob.glob(pattern):
            r = load_one(tj)
            if r:
                r["split"] = split
                rows.append(r)

    # Statistics
    by_type = Counter(r["task_type"] for r in rows)
    by_split = Counter(r["split"] for r in rows)
    print(f"== Matched task stats (kitchen + {len(WANT_TYPES)} types) ==")
    print("total:", len(rows))
    print("by type:", dict(by_type))
    print("by split:", dict(by_split))

    def dump(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path

    # Write the full set (with the split field)
    full_path = dump(os.path.join(OUT_DIR, "tasks_kitchen.jsonl"), rows)
    print("wrote full set:", full_path, f"({len(rows)} tasks)")

    # Headline eval set = valid_seen + valid_unseen (ALFRED official split, a natural generalization axis)
    eval_rows = [r for r in rows if r["split"] in ("valid_seen", "valid_unseen")]
    eval_path = dump(os.path.join(OUT_DIR, "tasks_kitchen_eval.jsonl"), eval_rows)
    ev_split = Counter(r["split"] for r in eval_rows)
    ev_type = Counter(r["task_type"] for r in eval_rows)
    print(f"wrote eval set (valid_seen+unseen, {len(eval_rows)} tasks):", eval_path)
    print("  eval by split:", dict(ev_split), "| by type:", dict(ev_type))

    # Training pool = train split (for L0/L4 training/synthesis, decoupled from eval scenes)
    train_rows = [r for r in rows if r["split"] == "train"]
    train_path = dump(os.path.join(OUT_DIR, "tasks_kitchen_train.jsonl"), train_rows)
    print(f"wrote train set (train, {len(train_rows)} tasks):", train_path)

    # Balanced sample (SAMPLE_PER_TYPE per type, for quick smoke tests)
    random.seed(42)
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["task_type"]].append(r)
    sample = []
    for tt, items in buckets.items():
        random.shuffle(items)
        sample.extend(items[:SAMPLE_PER_TYPE])
    sample_path = dump(os.path.join(OUT_DIR, "tasks_kitchen_sample.jsonl"), sample)
    print(f"wrote sample ({len(sample)} tasks):", sample_path)

    # Print a few examples to eyeball the high_pddl subgoal sequences
    print("\n== Examples (one per type) ==")
    shown = set()
    for r in rows:
        if r["task_type"] in shown:
            continue
        shown.add(r["task_type"])
        seq = " -> ".join(a["action"] for a in r["high_pddl_actions"])
        print(f"\n[{r['task_type']}] scene={r['floor_plan']} id={r['task_id']}")
        print("  instruction:", r["instruction"][:90])
        print("  subgoal sequence:", seq)
        print("  pddl_params:", r["pddl_params"])
        if len(shown) >= len(WANT_TYPES):
            break


if __name__ == "__main__":
    main()
