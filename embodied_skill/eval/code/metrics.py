"""Phase 4 retrieval/selection metrics: correctness is always judged by **effect coverage**
(not by whether the IDs match exactly).

WP2a (effect-grounded coverage): effect signatures are always derived from the **true effects
recomputed from sub_skills** (common.true_effect_sig, including task-container judgment), rather
than the effects **declared** on the card. Distractors misreport their effects (lookalike swaps
the action / missing drops steps but still writes the full set). Using the declared values would
reward methods that "get fooled by the impostor" and penalize methods that "see through the true
structure", making the paper metrics contradict the simulation. Once true effects are used
uniformly, all methods (retrieval/LLM/rule) are compared on the same yardstick.
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
    """Some skill in the top-K has a **true** effect signature ⊇ required_effects (single-skill hit)."""
    req = set(required)
    return int(any(_sig(c, by_id, binding) >= req for c in ranked[:k]))


def fullcov_at_k(ranked, required, k, by_id, binding=None):
    """Union of the top-K skills' **true** effect signatures ⊇ required_effects (compositional full coverage)."""
    req = set(required)
    return int(_union_sig(ranked[:k], by_id, binding) >= req)


def id_hit_at_k(ranked, gt_ids, k):
    """ID-level hit (secondary reference): any gold-standard skill ID appears in the top-K."""
    gt = set(gt_ids)
    return int(any(c.get("skill_id") in gt for c in ranked[:k]))


def selection_scores(chosen, required, by_id, binding=None):
    """Score the "selected skill set" (LLM/pipeline methods): single-skill hit? / full coverage? / how many selected.
    Effects always use the true signature (effect-grounded), so distractors' misreported declared effects no longer slip through."""
    req = set(required)
    return {
        "sel_hit": int(any(_sig(c, by_id, binding) >= req for c in chosen)),
        "sel_fullcov": int(_union_sig(chosen, by_id, binding) >= req),
        "sel_size": len(chosen),
    }
