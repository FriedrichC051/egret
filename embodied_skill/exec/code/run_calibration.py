"""Phase 3: executor + success checker + "gold standard vs wrong skill" calibration comparison.

Purpose (a key pilot calibration, not routing itself):
  Verify that the causal chain "give the right skills -> simulation succeeds; give wrong/insufficient skills ->
  simulation fails" holds, thereby proving that judging the router by SR/GC later is trustworthy.

Approach: for several sampled tasks, execute three skill sets in their corresponding scenes and compare SR/GC:
  - gold : the task's gold-standard composite skill (gt_composite_set[0], a fully executable sub_skills sequence)
  - naive: a "seemingly reasonable but insufficient" skill (pick_and_place, only places, no heat/slice/clean)
  - empty: does nothing (lower-bound reference)

Success check (ALFRED style):
  - at_receptacle: the final-state obj's parentReceptacles contains the target receptacle (placement persists, read final state)
  - isHot/isCold : during execution the obj temperature "ever reached" Hot/Cold (peak tracking, since later steps change temperature)
  - isSliced     : during execution the obj "was ever sliced"
  - isClean      : the cleaning sub-skill succeeded (isDirty went from true to false)
  SR = all required_effects achieved; GC = fraction achieved.

Usage (WSL, requires GPU display environment):
  DISPLAY=:0 GALLIUM_DRIVER=d3d12 python exec/code/run_calibration.py --n 20
"""
import os
import sys
import json
import argparse
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))              # exec/code/
ROOT = os.path.dirname(os.path.dirname(HERE))                 # embodied_skill/
SKILLS_CODE = os.path.join(ROOT, "skills", "code")            # importable modules
SKILLS_DATA = os.path.join(ROOT, "skills", "datasets")        # skill card JSON
GT_DATA = os.path.join(ROOT, "gt", "datasets")               # gt_kitchen.jsonl
sys.path.insert(0, SKILLS_CODE)
import atomic_bodies as ab            # noqa: E402
from atomic_bodies import _norm       # noqa: E402
from skill_effects import goal_signature  # noqa: E402

# 5 gold-standard seeds (prefer these as executable gold; their naming is the most standardized)
SEED_IDS = ["pick_and_place", "heat_then_place", "cool_then_place",
            "clean_then_place", "slice_object_with_knife", "slice_then_place",
            "heat_then_slice_then_place"]


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_pool_index():
    """skill_id -> card (including sub_skills). Merge atomic + pool_clean (composite + distractors)."""
    idx = {}
    for c in load_json(os.path.join(SKILLS_DATA, "atomic_skills.json")):
        idx[c["skill_id"]] = c
    for c in load_json(os.path.join(SKILLS_DATA, "pool_clean.json")):
        idx[c["skill_id"]] = c
    return idx


def build_exec_gold_index(pool):
    """effect signature -> executable composite gold-standard card.

    When gt_composite_set's set-cover ties, it may pick a bare atomic (e.g. put_object) -- which lacks goto/pickup
    glue steps and cannot execute independently. Phase 3 needs a "truly runnable" composite card, so it picks, by the
    effect signature of required_effects, an equivalent and fully executable one from the gold composite cards to run.
    Seed cards are preferred (most standardized naming).
    """
    def rank(card):
        # Card selection priority: seed > labeled gold > other executable composite
        return (card["skill_id"] in SEED_IDS, bool(card.get("gold")))

    sig2card = {}
    for c in pool.values():
        # Include all "executable composites" (including LLM-generated slice_then_place, etc.), no longer limited to seed gold,
        # otherwise slice-type signatures would have no match -> gold falls back to bare atomic -> the ceiling is wrongly pulled to 0.
        if c.get("type") != "composite" or not c.get("is_executable"):
            continue
        if c.get("is_distractor") or not c.get("sub_skills"):
            continue
        sig = frozenset(goal_signature(c.get("effects", [])))
        prev = sig2card.get(sig)
        if prev is None or rank(c) > rank(prev):
            sig2card[sig] = c
    return sig2card


def pick_exec_gold(task, exec_gold_index, pool):
    """Pick an executable gold-standard composite card for the task: match by the required_effects effect signature."""
    sig = frozenset(task["required_effects"])
    card = exec_gold_index.get(sig)
    if card is not None:
        return card
    # Fallback: use gt_composite_set[0] (possibly a bare atomic, only when there is no matching composite)
    gid = task["gt_composite_set"][0] if task["gt_composite_set"] else None
    return pool.get(gid)


def find_target_objs(ev, base_type):
    """Match target objects: after normalization, objectType starts with base (compatible with post-slice 'AppleSliced')."""
    b = _norm(base_type)
    out = []
    for o in ev.metadata["objects"]:
        t = _norm(o["objectType"])
        if t.startswith(b) or b.startswith(t):
            out.append(o)
    return out


# sub-skill -> the effect token it is responsible for achieving (used to judge "this action was actually performed during execution")
ACTION_TOKEN = {
    "heat_object_microwave": "isHot",
    "cool_object_fridge": "isCold",
    "slice_object": "isSliced",
    "clean_object_sink": "isClean",
}
EMPTY_FLAGS = {"isHot": False, "isCold": False, "isSliced": False, "isClean": False}


def read_state_flags(ctrl, task):
    """Read the target object's effect flags in the current world state (used to record the initial state, for "subtracting the initial state")."""
    obj_type = task["binding"]["obj"]
    recep = task["binding"]["receptacle"]
    f = {"isHot": False, "isCold": False, "isSliced": False, "isClean": False, "at_receptacle": False}
    rb = _norm(recep) if recep else None
    for o in find_target_objs(ctrl.last_event, obj_type):
        if o.get("temperature") == "Hot":
            f["isHot"] = True
        if o.get("temperature") == "Cold":
            f["isCold"] = True
        if o.get("isSliced") or "sliced" in _norm(o["objectType"]):
            f["isSliced"] = True
        if not o.get("isDirty", False):
            f["isClean"] = True
        if rb:
            for p in (o.get("parentReceptacles") or []):
                if rb in _norm(p):
                    f["at_receptacle"] = True
    return f


def execute_and_track(ctrl, card, bindings):
    """Execute sub_skills, returning (flags peak-achieved, did whether the corresponding action was performed, placed_to_target, log)."""
    flags = dict(EMPTY_FLAGS)
    did = dict(EMPTY_FLAGS)
    placed_to_target = False
    log = []
    obj_type = bindings["obj"]
    for st in card.get("sub_skills", []):
        skill = st["skill"]
        raw_args = st.get("args", [])
        fn = ab.REGISTRY.get(skill)
        if fn is None:
            log.append((skill, False, "unknown skill"))
            continue
        args = [bindings.get(a, a) for a in raw_args]
        _, ok, info = fn(ctrl, *args)
        log.append((skill, ok, info))
        if ok and skill in ACTION_TOKEN:
            did[ACTION_TOKEN[skill]] = True
        # placed into the "target receptacle" (the original placeholder receptacle, not an intermediate surface) and succeeded
        if ok and skill == "put_object" and len(raw_args) >= 2 and raw_args[1] == "receptacle":
            placed_to_target = True
        if skill == "clean_object_sink" and ok:
            flags["isClean"] = True
        # peak-track the target object's temperature/slice state
        for o in find_target_objs(ctrl.last_event, obj_type):
            if o.get("temperature") == "Hot":
                flags["isHot"] = True
            if o.get("temperature") == "Cold":
                flags["isCold"] = True
            if o.get("isSliced") or "sliced" in _norm(o["objectType"]):
                flags["isSliced"] = True
    return flags, did, placed_to_target, log


def check_success(ctrl, task, init, flags, did, placed_to_target):
    """Achievement judgment (subtracting the initial state): token achieved = finally satisfied AND (not initially satisfied OR the corresponding action was performed).
    This way empty (doing nothing) cannot get by on "the object was already in the target receptacle / already hot initially",
    while not wronging a truly-executing gold (even if already satisfied initially, as long as the corresponding action was performed it counts as achieved)."""
    req = set(task["required_effects"])
    obj_type = task["binding"]["obj"]
    recep = task["binding"]["receptacle"]

    final_at = False
    if recep:
        rb = _norm(recep)
        for o in find_target_objs(ctrl.last_event, obj_type):
            for p in (o.get("parentReceptacles") or []):
                if rb in _norm(p):
                    final_at = True

    achieved = set()
    for tok in ["isHot", "isCold", "isSliced", "isClean"]:
        if tok in req and flags.get(tok) and (not init.get(tok) or did.get(tok)):
            achieved.add(tok)
    if "at_receptacle" in req and final_at and (not init.get("at_receptacle") or placed_to_target):
        achieved.add("at_receptacle")

    SR = (achieved == req)
    GC = len(achieved) / max(1, len(req))
    return SR, GC, achieved


def run_condition(ctrl, task, card, bindings):
    ctrl.reset(task["floor_plan"])
    init = read_state_flags(ctrl, task)      # record initial state before execution
    if card is None:                         # empty reference
        flags, did, placed, log = dict(EMPTY_FLAGS), dict(EMPTY_FLAGS), False, []
    else:
        flags, did, placed, log = execute_and_track(ctrl, card, bindings)
    SR, GC, achieved = check_success(ctrl, task, init, flags, did, placed)
    return SR, GC, achieved, log, init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=os.path.join(GT_DATA, "gt_kitchen.jsonl"))
    ap.add_argument("--n", type=int, default=20, help="number of sampled tasks")
    ap.add_argument("--seed", type=int, default=0, help="random sampling seed")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    pool = load_pool_index()
    exec_gold_index = build_exec_gold_index(pool)
    naive_card = pool.get("pick_and_place")

    # Read GT, sampling the two task types in a balanced way
    rows = []
    with open(args.gt, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    import random
    rng = random.Random(args.seed)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["task_type"]].append(r)
    sample = []
    per = max(1, args.n // max(1, len(by_type)))
    for tt, lst in by_type.items():
        lst = list(lst)
        rng.shuffle(lst)
        sample += lst[:per]
    rng.shuffle(sample)
    sample = sample[: args.n]

    from ai2thor.controller import Controller
    from ai2thor.platform import Linux64
    ctrl = Controller(platform=Linux64, scene="FloorPlan1", width=300, height=300, x_display="0")

    agg = {m: {"SR": 0, "GC": 0.0} for m in ["gold", "naive", "empty"]}
    dirty_init = 0    # number of queries whose initial state already satisfies some required effects (a measure of unclean initial data state)
    print(f"== Phase3 calibration: {len(sample)} queries, running the three conditions gold/naive/empty per query ==\n")
    for r in sample:
        binding = dict(r["binding"])
        binding.setdefault("liquid", "water")
        gold_card = pick_exec_gold(r, exec_gold_index, pool)
        gold_id = gold_card["skill_id"] if gold_card else None
        conds = {"gold": gold_card, "naive": naive_card, "empty": None}
        line = f"[{r['task_type'][:16]:16}] {r['task_id'][:22]:22} req={sorted(r['required_effects'])}"
        res = {}
        init_seen = None
        for m, card in conds.items():
            SR, GC, ach, log, init = run_condition(ctrl, r, card, binding)
            init_seen = init
            agg[m]["SR"] += int(SR)
            agg[m]["GC"] += GC
            res[m] = (SR, GC, ach)
            if args.verbose and m == "gold" and not SR:
                fails = [s for s in log if s[1] is False]
                line += f"\n      gold failed steps={fails}"
        pre = {t for t in r["required_effects"] if init_seen and init_seen.get(t)}
        if pre:
            dirty_init += 1
            line += f"\n      [initially satisfied {sorted(pre)}, subtracted]"
        print(line)
        print(f"      gold SR={res['gold'][0]} GC={res['gold'][1]:.2f} achieved{sorted(res['gold'][2])}"
              f" | naive SR={res['naive'][0]} GC={res['naive'][1]:.2f}"
              f" | empty GC={res['empty'][1]:.2f}  (gold={gold_id})")
    ctrl.stop()

    n = len(sample)
    print("\n================ calibration summary ================")
    for m in ["gold", "naive", "empty"]:
        print(f"  {m:6}: SR={agg[m]['SR']}/{n}={agg[m]['SR']/n:.1%}  GC avg={agg[m]['GC']/n:.1%}")
    print(f"  queries with unclean initial state (subtracted): {dirty_init}/{n}")
    print("\nConclusion criterion: gold's SR should be significantly higher than naive/empty -> the causal chain holds, SR/GC is trustworthy.")


if __name__ == "__main__":
    main()
