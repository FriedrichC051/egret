"""复合技能执行器（解释器）。

核心思想：复合技能的 body 不是手写代码，而是一串"子技能序列"(sub_skills)。
本模块提供统一解释器 execute_sub_skills，把 sub_skills 里的占位参数(obj/receptacle)
绑定成实际物体类型，再依次调用 atomic_bodies 里的原子技能。

好处：
- LLM 批量合成时只需产出 sub_skills 列表(JSON)，无需生成可执行代码 → 安全；
- 易混淆变体(错序/缺步)只是对 sub_skills 做置换/删除 → 天然可造；
- 效果签名可从 sub_skills 的原子 effects 组合推导。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atomic_bodies as ab  # noqa: E402


def _bind(arg, bindings):
    """占位符(obj/receptacle/...)替换成实际值；非占位符按字面物体类型处理(如 Microwave)。"""
    return bindings.get(arg, arg)


def execute_sub_skills(ctrl, sub_skills, bindings, stop_on_fail=True):
    """依次执行子技能序列。返回 (event, ok, log)。log 记录每步 (skill, args, ok, info)。"""
    log = []
    ev = ctrl.last_event
    for st in sub_skills:
        skill = st["skill"]
        fn = ab.REGISTRY.get(skill)
        if fn is None:
            log.append((skill, st.get("args", []), False, "unknown skill"))
            if stop_on_fail:
                return ev, False, log
            continue
        args = [_bind(a, bindings) for a in st.get("args", [])]
        ev, ok, info = fn(ctrl, *args)
        log.append((skill, args, ok, info))
        if not ok and stop_on_fail:
            return ev, False, log
    return ev, True, log


def execute_composite(ctrl, card, bindings, stop_on_fail=True):
    """执行一张复合技能卡（读取其 sub_skills）。"""
    return execute_sub_skills(ctrl, card.get("sub_skills", []), bindings, stop_on_fail)
