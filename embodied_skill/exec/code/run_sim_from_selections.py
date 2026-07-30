"""Phase 4 simulation stage (option B: run in WSL, consuming selections.json produced by the Windows routing stage).

For each query and each routing method's chosen_ids in selections.json, execute that set of skills
in the corresponding scene and compute SR/GC with the same success checker used in Phase 3; additionally
run gold (the executable gold standard) as a ceiling reference. Finally print a "method x SR/GC" comparison
table and write sim_results.json.

Execution rules: the cards in chosen_ids are executed in order -- a composite card runs its sub_skills, an
atomic card runs itself (skill_id + params); effect flags are accumulated as peaks throughout, and success is
finally judged uniformly by "subtracting the initial state".

Usage (WSL, GPU display environment):
  DISPLAY=:0 GALLIUM_DRIVER=d3d12 python exec/code/run_sim_from_selections.py \
      --selections eval/datasets/selections.json --limit 30
"""
import os
import sys
import json
import argparse
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))              # exec/code/
ROOT = os.path.dirname(os.path.dirname(HERE))                 # embodied_skill/
EVAL_DATA = os.path.join(ROOT, "eval", "datasets")
SKILLS_CODE = os.path.join(ROOT, "skills", "code")
sys.path.insert(0, HERE)
sys.path.insert(0, SKILLS_CODE)
import atomic_bodies as ab                                     # noqa: E402
# Reuse Phase 3's executor / checker components
from run_calibration import (                                  # noqa: E402
    load_pool_index, find_target_objs, read_state_flags, check_success,
    build_exec_gold_index, pick_exec_gold,
    ACTION_TOKEN, EMPTY_FLAGS,
)


def card_to_steps(card):
    """Break a skill card into a sequence of [(skill, arg_names)] steps.
    Composite card -> sub_skills; atomic card -> single step (its own skill_id + params)."""
    subs = card.get("sub_skills")
    if subs:
        return [(st["skill"], st.get("args", [])) for st in subs]
    return [(card["skill_id"], card.get("params", []))]


def execute_chosen(ctrl, cards, bindings):
    """Execute a set of skill cards in order, accumulating effect flags as peaks, returning (flags, did, placed, log)."""
    flags = dict(EMPTY_FLAGS)
    did = dict(EMPTY_FLAGS)
    placed_to_target = False
    log = []
    obj_type = bindings["obj"]
    for card in cards:
        for skill, raw_args in card_to_steps(card):
            fn = ab.REGISTRY.get(skill)
            if fn is None:
                log.append((skill, False, "unknown skill"))
                continue
            args = [bindings.get(a, a) for a in raw_args]
            _, ok, info = fn(ctrl, *args)
            log.append((skill, ok, info))
            if ok and skill in ACTION_TOKEN:
                did[ACTION_TOKEN[skill]] = True
            if ok and skill == "put_object" and len(raw_args) >= 2 and raw_args[1] == "receptacle":
                placed_to_target = True
            if skill == "clean_object_sink" and ok:
                flags["isClean"] = True
            for o in find_target_objs(ctrl.last_event, obj_type):
                if o.get("temperature") == "Hot":
                    flags["isHot"] = True
                if o.get("temperature") == "Cold":
                    flags["isCold"] = True
                if o.get("isSliced") or "sliced" in ab._norm(o["objectType"]):
                    flags["isSliced"] = True
    return flags, did, placed_to_target, log


def make_controller():
    from ai2thor.controller import Controller
    from ai2thor.platform import Linux64
    return Controller(platform=Linux64, scene="FloorPlan1",
                      width=300, height=300, x_display="0")


def run_one(ctrl, task, cards, bindings):
    ctrl.reset(task["floor_plan"])
    init = read_state_flags(ctrl, task)
    if not cards:
        flags, did, placed = dict(EMPTY_FLAGS), dict(EMPTY_FLAGS), False
    else:
        flags, did, placed, _ = execute_chosen(ctrl, cards, bindings)
    return check_success(ctrl, task, init, flags, did, placed)   # (SR, GC, achieved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selections", default=os.path.join(EVAL_DATA, "selections.json"))
    ap.add_argument("--methods", nargs="+", default=None,
                    help="method names to simulate (default: all that appear in selections)")
    ap.add_argument("--gold", action="store_true", default=True,
                    help="additionally run the executable gold standard as a ceiling reference (on by default)")
    ap.add_argument("--no-gold", dest="gold", action="store_false")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N queries (0=all)")
    ap.add_argument("--merge", action="store_true",
                    help="only recompute this run's conds and merge into existing sim_results (keep other methods' results)")
    ap.add_argument("--out", default=os.path.join(EVAL_DATA, "sim_results.json"))
    args = ap.parse_args()

    data = json.load(open(args.selections, encoding="utf-8"))
    queries = data["queries"]
    if args.limit > 0:
        queries = queries[:args.limit]

    present = sorted({m for e in queries for m in e.get("methods", {})})
    methods = args.methods or present   # "gold" can also be passed as a method, handled specially in the loop
    pool = load_pool_index()
    exec_gold_index = build_exec_gold_index(pool)

    prev_res = {}
    if args.merge and os.path.exists(args.out):
        for r in json.load(open(args.out, encoding="utf-8")).get("per_query", []):
            prev_res[r["task_id"]] = r.get("res", {})

    ctx = {"ctrl": make_controller()}

    def safe_run(task, cards, binding):
        """Execute one condition; if the THOR backend times out/crashes, restart the controller and retry once, still failing records SR=False."""
        for attempt in range(2):
            try:
                return run_one(ctx["ctrl"], task, cards, binding)
            except Exception as ex:
                print(f"    !! THOR error ({ex.__class__.__name__}), restarting controller...", flush=True)
                try:
                    ctx["ctrl"].stop()
                except Exception:
                    pass
                ctx["ctrl"] = make_controller()
        return False, 0.0, set()

    run_conds = list(dict.fromkeys(list(methods) + (["gold"] if args.gold else [])))
    per_query = []
    print(f"== Phase4 simulation: {len(queries)} queries x this run {run_conds}"
          f"{' (merge existing)' if args.merge else ''} ==\n", flush=True)

    def dump_summary():
        # Summary is computed live from per_query, covering all methods seen (including historical methods in merge mode)
        all_m = list(dict.fromkeys([m for r in per_query for m in r["res"]]))
        agg = {m: [0, 0.0, 0] for m in all_m}
        for r in per_query:
            for m, v in r["res"].items():
                agg[m][0] += int(v["SR"]); agg[m][1] += v["GC"]; agg[m][2] += 1
        done = len(per_query)
        print(f"\n============ Phase4 simulation summary ({done}/{len(queries)} queries) ============")
        print(f"{'method':14} {'SR':>12} {'GC':>8}")
        for m in all_m:
            a = agg[m]; nn = max(1, a[2])
            print(f"{m:14} {a[0]}/{a[2]}={a[0]/nn:6.1%} {a[1]/nn:7.1%}")
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"meta": {"n": done, "methods": all_m},
                       "summary": {m: {"SR": agg[m][0] / max(1, agg[m][2]),
                                       "GC": agg[m][1] / max(1, agg[m][2])} for m in all_m},
                       "per_query": per_query}, f, ensure_ascii=False, indent=2)
        print(f"wrote {args.out}", flush=True)

    try:
        for e in queries:
            task = {"floor_plan": e["floor_plan"], "binding": e["binding"],
                    "required_effects": e["required_effects"],
                    "gt_composite_set": e.get("gt_ids", [])}
            binding = dict(e["binding"])
            binding.setdefault("liquid", "water")
            binding.setdefault("target", binding.get("obj"))
            rec = {"task_id": e["task_id"], "task_type": e["task_type"],
                   "res": dict(prev_res.get(e["task_id"], {}))}   # merge mode keeps historical methods
            line = f"[{e['task_type'][:16]:16}] {e['task_id'][:20]:20} req={sorted(e['required_effects'])}"
            for m in run_conds:
                if m == "gold":
                    card = pick_exec_gold(task, exec_gold_index, pool)
                    cards = [card] if card else []
                else:
                    ids = e["methods"].get(m, {}).get("chosen_ids", [])
                    cards = [pool[i] for i in ids if i in pool]
                SR, GC, ach = safe_run(task, cards, binding)
                rec["res"][m] = {"SR": bool(SR), "GC": round(GC, 3),
                                 "chosen": [c["skill_id"] for c in cards]}
            per_query.append(rec)
            print(line, flush=True)
            print("   " + " | ".join(
                f"{m}:SR={rec['res'][m]['SR']!s:5} GC={rec['res'][m]['GC']:.2f}" for m in run_conds),
                flush=True)
            if len(per_query) % 20 == 0:
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump({"meta": {"n": len(per_query), "partial": True},
                               "per_query": per_query}, f, ensure_ascii=False, indent=2)
    finally:
        try:
            ctx["ctrl"].stop()
        except Exception:
            pass
        dump_summary()


if __name__ == "__main__":
    main()
