"""Phase 1 复合技能执行冒烟：加载金标准复合技能卡，在 FloorPlan1 上执行并验证最终世界状态。

运行： wsl -d Ubuntu -- bash embodied_skill/skills/scripts/wsl_smoke_composite.sh
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))          # skills/code/
DATA = os.path.join(HERE, os.pardir, "datasets")           # skills/datasets/
sys.path.insert(0, HERE)
import atomic_bodies as ab  # noqa: E402
import composite_bodies as cb  # noqa: E402
from ai2thor.controller import Controller  # noqa: E402
from ai2thor.platform import Linux64  # noqa: E402


def load_seed():
    with open(os.path.join(DATA, "composite_skills_seed.json"), encoding="utf-8") as f:
        cards = json.load(f)
    return {c["skill_id"]: c for c in cards}


def obj_of(ev, t):
    for o in ev.metadata["objects"]:
        if ab._norm(o["objectType"]) == ab._norm(t):
            return o
    return None


def run_case(ctrl, cards, skill_id, bindings, checks):
    ctrl.reset("FloorPlan1")
    card = cards[skill_id]
    _, ok, log = cb.execute_composite(ctrl, card, bindings)
    print(f"\n=== {skill_id}  bindings={bindings}  exec_ok={ok} ===")
    for step in log:
        print("   ", step[0], step[1], "->", step[2], (step[3] if step[2] is False else ""))
    # 结果校验
    results = {}
    obj = obj_of(ctrl.last_event, bindings.get("obj", ""))
    for name, fn in checks.items():
        results[name] = bool(fn(obj))
    print("   校验:", results)
    return ok and all(results.values())


def main():
    cards = load_seed()
    ctrl = Controller(platform=Linux64, scene="FloorPlan1", width=300, height=300, x_display="0")

    # 打印刀具/可切物体情况，便于确认场景
    ev = ctrl.last_event
    types = sorted({o["objectType"] for o in ev.metadata["objects"]})
    knives = [t for t in types if "knife" in t.lower()]
    sliceables = [o["objectType"] for o in ev.metadata["objects"] if o.get("sliceable")]
    print("场景刀具:", knives, "| 可切物体样例:", sorted(set(sliceables))[:6])

    passed = {}
    # 放到中性容器 CounterTop（温度保持）；若放冰箱会被物理性重新冷却，那是 Phase3 成功判据要处理的语义
    passed["heat_then_place"] = run_case(
        ctrl, cards, "heat_then_place", {"obj": "Apple", "receptacle": "CounterTop"},
        {"isHot": lambda o: o and o.get("temperature") == "Hot",
         "on_counter": lambda o: o and o.get("parentReceptacles") and any("CounterTop" in p for p in o["parentReceptacles"])},
    )
    passed["pick_and_place"] = run_case(
        ctrl, cards, "pick_and_place", {"obj": "Tomato", "receptacle": "SinkBasin"},
        {"in_sink": lambda o: o and o.get("parentReceptacles") and any("Sink" in p for p in o["parentReceptacles"])},
    )
    passed["slice_object_with_knife"] = run_case(
        ctrl, cards, "slice_object_with_knife", {"obj": "Bread"},
        {"isSliced": lambda o: o and o.get("isSliced")},
    )

    print("\n==== 汇总 ====")
    for k, v in passed.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print("SMOKE_COMPOSITE_%s" % ("OK" if all(passed.values()) else "PARTIAL"))
    ctrl.stop()


if __name__ == "__main__":
    main()
