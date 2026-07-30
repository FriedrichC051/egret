"""效果签名工具：把技能卡/任务的 effects 归一化成"目标效果 token 集合"，
供 (a) 校验器判断复合技能 effects 是否自洽，(b) Phase 2 GT 的 set-cover 等价判定。

只保留"持久的目标类效果"（isHot/isCold/isClean/isSliced/isFilled/at_receptacle），
丢弃"手段类/使能类"效果（agent.holding、hand_empty、isOpen、isToggled 及一切 not ...）。
因为路由的正确性看的是"任务目标是否被覆盖"，而非中间手段。
"""

GOAL_TOKENS = ["isHot", "isCold", "isClean", "isSliced", "isFilledWithLiquid", "at_receptacle"]

# 热态互斥：isHot / isCold 不能共存，按执行顺序"后者覆盖前者"（先冷后热=热，先热后冷=冷）。
THERMAL_TOKENS = {"isHot", "isCold"}


def combine_sequential(token_sets):
    """按执行顺序合并多个子技能的目标 token 集合，得到复合技能"最终态"目标签名。
    与集合并集的区别：热态(isHot/isCold)互斥，某步引入热态时先清除相反热态（后者覆盖）。
    例：cool→heat 得 {isHot}；heat→cool 得 {isCold}，而非同时含两者。
    """
    acc = set()
    for toks in token_sets:
        toks = set(toks)
        if toks & THERMAL_TOKENS:
            acc -= THERMAL_TOKENS
        acc |= toks
    return acc


def canon_goal_effect(e):
    """把单条 effect 字符串映射成一个目标 token；非目标类返回 None。"""
    s = e.strip()
    if s.lower().startswith("not "):
        return None  # 撤销类效果不算目标
    if s.startswith("agent."):
        return None  # 智能体状态(位置/手持)是手段，不是任务目标
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
    """一个技能卡 effects 列表 -> 目标 token 集合。"""
    sig = set()
    for e in effects or []:
        t = canon_goal_effect(e)
        if t:
            sig.add(t)
    return sig
