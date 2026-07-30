"""Effect-signature utilities: normalize the effects of a skill card/task into a
"goal-effect token set", used by (a) the validator to check whether a composite
skill's effects are self-consistent, and (b) the set-cover equivalence check of
the Phase 2 GT.

Only "persistent goal-type effects" are kept (isHot/isCold/isClean/isSliced/
isFilled/at_receptacle); "means/enabling" effects are discarded (agent.holding,
hand_empty, isOpen, isToggled and any `not ...`). This is because routing
correctness depends on whether the "task goal is covered", not on intermediate
means.
"""

GOAL_TOKENS = ["isHot", "isCold", "isClean", "isSliced", "isFilledWithLiquid", "at_receptacle"]

# Thermal states are mutually exclusive: isHot / isCold cannot coexist; by
# execution order "the latter overrides the former" (cool-then-heat = hot,
# heat-then-cool = cold).
THERMAL_TOKENS = {"isHot", "isCold"}


def combine_sequential(token_sets):
    """Merge the goal-token sets of multiple sub-skills in execution order to get
    the composite skill's "final state" goal signature.
    Difference from a plain set union: thermal states (isHot/isCold) are mutually
    exclusive; when a step introduces a thermal state, the opposite thermal state
    is cleared first (the latter overrides).
    E.g. cool->heat yields {isHot}; heat->cool yields {isCold}, not both.
    """
    acc = set()
    for toks in token_sets:
        toks = set(toks)
        if toks & THERMAL_TOKENS:
            acc -= THERMAL_TOKENS
        acc |= toks
    return acc


def canon_goal_effect(e):
    """Map a single effect string to one goal token; return None for non-goal effects."""
    s = e.strip()
    if s.lower().startswith("not "):
        return None  # undo-type effects don't count as goals
    if s.startswith("agent."):
        return None  # agent state (position/holding) is a means, not a task goal
    if "isHot" in s:
        return "isHot"
    if "isCold" in s:
        return "isCold"
    if "isClean" in s:
        return "isClean"
    if "isSliced" in s:
        return "isSliced"
    if "isFilledWithLiquid" in s:
        return "isFilledWithLiquid"
    if ".at(" in s or s.startswith("at(") or ".at_" in s:
        return "at_receptacle"
    return None


def goal_signature(effects):
    """A skill card's effects list -> goal token set."""
    sig = set()
    for e in effects or []:
        t = canon_goal_effect(e)
        if t:
            sig.add(t)
    return sig
