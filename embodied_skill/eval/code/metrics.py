"""Phase 4 检索/选择指标：一律按**效果覆盖**判正确（不看 ID 是否完全一致）。

WP2a（effect-grounded coverage）：效果签名一律从 **sub_skills 重算的真实效果**
（common.true_effect_sig，含任务容器判定）得到，而非卡上**声明**的 effects。
干扰项会谎报 effects（lookalike 换动作 / missing 少步仍写全），若用声明值会奖励
"被李鬼骗到"的方法、冤枉"看穿真实结构"的方法，导致纸面指标与仿真反着来。
统一用真实效果后，所有方法（检索/LLM/规则）在同一把尺子上比较。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # embodied_skill/
ROUTER_CODE = os.path.join(ROOT, "router", "code")
if ROUTER_CODE not in sys.path:
    sys.path.insert(0, ROUTER_CODE)
from common import true_effect_sig  # noqa: E402


def _sig(card, by_id, binding):
    return true_effect_sig(card, by_id, binding)


def _union_sig(cards, by_id, binding):
    u = set()
    for c in cards:
        u |= _sig(c, by_id, binding)
    return u


def hit_at_k(ranked, required, k, by_id, binding=None):
    """top-K 里存在某条技能其**真实**效果签名 ⊇ required_effects（单技能命中）。"""
    req = set(required)
    return int(any(_sig(c, by_id, binding) >= req for c in ranked[:k]))


def fullcov_at_k(ranked, required, k, by_id, binding=None):
    """top-K 技能**真实**效果签名并集 ⊇ required_effects（组合式完整覆盖）。"""
    req = set(required)
    return int(_union_sig(ranked[:k], by_id, binding) >= req)


def id_hit_at_k(ranked, gt_ids, k):
    """ID 级命中（次要参考）：top-K 里出现任一金标准技能 ID。"""
    gt = set(gt_ids)
    return int(any(c.get("skill_id") in gt for c in ranked[:k]))


def selection_scores(chosen, required, by_id, binding=None):
    """对"选出的技能集合"（LLM/流水线方法）评：是否单技能命中 / 是否完整覆盖 / 选了几条。
    效果一律用真实签名（effect-grounded），干扰项谎报的声明 effects 不再蒙混过关。"""
    req = set(required)
    return {
        "sel_hit": int(any(_sig(c, by_id, binding) >= req for c in chosen)),
        "sel_fullcov": int(_union_sig(chosen, by_id, binding) >= req),
        "sel_size": len(chosen),
    }
