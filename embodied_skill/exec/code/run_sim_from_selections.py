"""Phase 4 仿真阶段（方案 B：在 WSL 跑，消费 Windows 路由阶段产出的 selections.json）。

对 selections.json 里每题、每个路由方法的 chosen_ids，在其所属场景执行这组技能，
用与 Phase 3 相同的成功检查器算 SR/GC；额外跑 gold（可执行金标准）作天花板对照。
最后打印"方法 × SR/GC"对比表并写 sim_results.json。

执行规则：chosen_ids 里的卡按序执行——复合卡执行其 sub_skills，原子卡执行自身
(skill_id + params)；效果 flag 全程峰值累积，最后统一按"扣除初始态"判定达成。

用法（WSL，GPU 显示环境）：
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
# 复用 Phase 3 的执行/检查器件
from run_calibration import (                                  # noqa: E402
    load_pool_index, find_target_objs, read_state_flags, check_success,
    build_exec_gold_index, pick_exec_gold,
    ACTION_TOKEN, EMPTY_FLAGS,
)


def card_to_steps(card):
    """把一张技能卡拆成 [(skill, arg_names)] 步骤序列。
    复合卡 -> sub_skills；原子卡 -> 单步(自身 skill_id + params)。"""
    subs = card.get("sub_skills")
    if subs:
        return [(st["skill"], st.get("args", [])) for st in subs]
    return [(card["skill_id"], card.get("params", []))]


def execute_chosen(ctrl, cards, bindings):
    """按序执行一组技能卡，峰值累积效果 flag，返回 (flags, did, placed, log)。"""
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
                    help="要仿真的方法名（默认 selections 里出现的全部）")
    ap.add_argument("--gold", action="store_true", default=True,
                    help="附带跑可执行金标准作天花板对照（默认开）")
    ap.add_argument("--no-gold", dest="gold", action="store_false")
    ap.add_argument("--limit", type=int, default=0, help="仅跑前 N 题（0=全部）")
    ap.add_argument("--merge", action="store_true",
                    help="只重算本次 conds 并合并进已有 sim_results（保留其它方法结果）")
    ap.add_argument("--out", default=os.path.join(EVAL_DATA, "sim_results.json"))
    args = ap.parse_args()

    data = json.load(open(args.selections, encoding="utf-8"))
    queries = data["queries"]
    if args.limit > 0:
        queries = queries[:args.limit]

    present = sorted({m for e in queries for m in e.get("methods", {})})
    methods = args.methods or present   # "gold" 亦可作为 methods 传入，循环里特殊处理
    pool = load_pool_index()
    exec_gold_index = build_exec_gold_index(pool)

    prev_res = {}
    if args.merge and os.path.exists(args.out):
        for r in json.load(open(args.out, encoding="utf-8")).get("per_query", []):
            prev_res[r["task_id"]] = r.get("res", {})

    ctx = {"ctrl": make_controller()}

    def safe_run(task, cards, binding):
        """执行一条件；THOR 后端超时/崩溃则重启 controller 后重试一次，仍失败记 SR=False。"""
        for attempt in range(2):
            try:
                return run_one(ctx["ctrl"], task, cards, binding)
            except Exception as ex:
                print(f"    !! THOR 异常({ex.__class__.__name__})，重启 controller…", flush=True)
                try:
                    ctx["ctrl"].stop()
                except Exception:
                    pass
                ctx["ctrl"] = make_controller()
        return False, 0.0, set()

    run_conds = list(dict.fromkeys(list(methods) + (["gold"] if args.gold else [])))
    per_query = []
    print(f"== Phase4 仿真：{len(queries)} 题 × 本次跑 {run_conds}"
          f"{'（合并已有）' if args.merge else ''} ==\n", flush=True)

    def dump_summary():
        # 汇总从 per_query 现算，覆盖所有出现过的方法（合并模式下含历史方法）
        all_m = list(dict.fromkeys([m for r in per_query for m in r["res"]]))
        agg = {m: [0, 0.0, 0] for m in all_m}
        for r in per_query:
            for m, v in r["res"].items():
                agg[m][0] += int(v["SR"]); agg[m][1] += v["GC"]; agg[m][2] += 1
        done = len(per_query)
        print(f"\n============ Phase4 仿真汇总（{done}/{len(queries)} 题）============")
        print(f"{'method':14} {'SR':>12} {'GC':>8}")
        for m in all_m:
            a = agg[m]; nn = max(1, a[2])
            print(f"{m:14} {a[0]}/{a[2]}={a[0]/nn:6.1%} {a[1]/nn:7.1%}")
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"meta": {"n": done, "methods": all_m},
                       "summary": {m: {"SR": agg[m][0] / max(1, agg[m][2]),
                                       "GC": agg[m][1] / max(1, agg[m][2])} for m in all_m},
                       "per_query": per_query}, f, ensure_ascii=False, indent=2)
        print(f"已写 {args.out}", flush=True)

    try:
        for e in queries:
            task = {"floor_plan": e["floor_plan"], "binding": e["binding"],
                    "required_effects": e["required_effects"],
                    "gt_composite_set": e.get("gt_ids", [])}
            binding = dict(e["binding"])
            binding.setdefault("liquid", "water")
            binding.setdefault("target", binding.get("obj"))
            rec = {"task_id": e["task_id"], "task_type": e["task_type"],
                   "res": dict(prev_res.get(e["task_id"], {}))}   # 合并模式保留历史方法
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
