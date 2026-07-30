"""Composite skill executor (interpreter).

Core idea: the body of a composite skill is not hand-written code but a
"sub-skill sequence" (sub_skills). This module provides a unified interpreter,
execute_sub_skills, which binds the placeholder args (obj/receptacle) in
sub_skills to actual object types, then calls the atomic skills in atomic_bodies
in order.

Benefits:
- When batch-synthesizing with an LLM, we only need to produce a sub_skills list
  (JSON), no executable code -> safe;
- Confusing variants (reorder/missing steps) are just permutations/deletions of
  sub_skills -> naturally easy to construct;
- The effect signature can be derived by combining the atomic effects of the
  sub_skills.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atomic_bodies as ab  # noqa: E402


def _bind(arg, bindings):
    """Replace placeholders (obj/receptacle/...) with actual values; non-placeholders
    are treated as literal object types (e.g. Microwave)."""
    return bindings.get(arg, arg)


def execute_sub_skills(ctrl, sub_skills, bindings, stop_on_fail=True):
    """Execute the sub-skill sequence in order. Returns (event, ok, log). `log`
    records each step as (skill, args, ok, info)."""
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
    """Execute one composite skill card (reads its sub_skills)."""
    return execute_sub_skills(ctrl, card.get("sub_skills", []), bindings, stop_on_fail)
