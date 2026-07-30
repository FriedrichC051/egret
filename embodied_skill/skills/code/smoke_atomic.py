"""Phase 1 原子技能执行冒烟：在 FloorPlan1 上跑 goto->pickup->heat->put 核心链路，
验证原子技能 body 能真实驱动 AI2-THOR 并产生预期状态（物体变热、放置成功）。

运行： wsl -d Ubuntu -- bash embodied_skill/skills/scripts/wsl_smoke_atomic.sh
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atomic_bodies as ab  # noqa: E402
from ai2thor.controller import Controller  # noqa: E402
from ai2thor.platform import Linux64  # noqa: E402


def obj_by_type(ev, t):
    for o in ev.metadata["objects"]:
        if ab._norm(o["objectType"]) == ab._norm(t):
            return o
    return None


def main():
    ctrl = Controller(platform=Linux64, scene="FloorPlan1", width=300, height=300, x_display="0")
    ev = ctrl.last_event

    # 选一个存在且可抓取的食材
    food = None
    for cand in ["Apple", "Potato", "Tomato", "Egg", "Bread"]:
        o = obj_by_type(ev, cand)
        if o and o.get("pickupable"):
            food = cand
            break
    if food is None:
        print("FAIL 场景里没有可抓取食材"); ctrl.stop(); return
    print("选用食材:", food)

    # 1) goto + pickup
    _, ok, info = ab.goto_object(ctrl, food); print(f"[goto {food}] ok={ok} {info}")
    _, ok, info = ab.pickup_object(ctrl, food); print(f"[pickup {food}] ok={ok} {info}")
    held = ab._held(ctrl.last_event)
    print("  持有:", held["objectType"] if held else None)

    # 2) goto microwave + heat
    _, ok, info = ab.goto_object(ctrl, "Microwave"); print(f"[goto Microwave] ok={ok} {info}")
    _, ok, info = ab.heat_object_microwave(ctrl, food); print(f"[heat {food}] ok={ok} {info}")
    o = obj_by_type(ctrl.last_event, food)
    print("  加热后温度:", o.get("temperature") if o else None)

    # 3) goto countertop + put
    _, ok, info = ab.goto_object(ctrl, "CounterTop"); print(f"[goto CounterTop] ok={ok} {info}")
    _, ok, info = ab.put_object(ctrl, food, "CounterTop"); print(f"[put {food} on CounterTop] ok={ok} {info}")
    o = obj_by_type(ctrl.last_event, food)
    parents = o.get("parentReceptacles") if o else None
    print("  放置后 parentReceptacles:", parents)

    # 结论
    hot_ok = o and o.get("temperature") == "Hot"
    put_ok = parents and any("CounterTop" in p for p in parents)
    print("\nRESULT heat_ok=%s put_ok=%s" % (bool(hot_ok), bool(put_ok)))
    print("SMOKE_ATOMIC_%s" % ("OK" if (hot_ok and put_ok) else "PARTIAL"))
    ctrl.stop()


if __name__ == "__main__":
    main()
