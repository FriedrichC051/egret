"""Shared data layer for Phase 4 routing evaluation: skill-pool loading,
retrieval-document construction, effect signatures, and query loading.

Key conventions (see plan section 7, "effect signature + set-cover equivalence check"):
- Correctness is judged not by exact skill-ID match but by **target effect coverage**:
  - Single-skill hit (Hit): a skill's effect signature superset-of required_effects (one is enough).
  - Full coverage (FullCoverage): the union of a skill set's effect signatures superset-of required_effects.
- Skill name/effects/description are all in English (description already batch-translated by englishify),
  and ALFRED queries are English too; retrieval doc = name + tokenized skill_id + effects + description.
"""
import os
import sys
import json
import re
import random

HERE = os.path.dirname(os.path.abspath(__file__))                 # router/code/
ROOT = os.path.dirname(os.path.dirname(HERE))                     # embodied_skill/
SKILLS_CODE = os.path.join(ROOT, "skills", "code")
SKILLS_DATA = os.path.join(ROOT, "skills", "datasets")
GT_DATA = os.path.join(ROOT, "gt", "datasets")
sys.path.insert(0, SKILLS_CODE)
from skill_effects import goal_signature, combine_sequential  # noqa: E402


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_pool():
    """Full skill pool: 13 atomic + pool_clean (seed composites + generated composites +
    distractor variants). Deduplicated by skill_id.
    The POOL_FILE env var can override pool_clean (used for confusion-tier pools
    pool_tier_{low,mid,high}.json); either an absolute path or a filename under
    skills/datasets is accepted."""
    pool_file = os.environ.get("POOL_FILE") or os.path.join(SKILLS_DATA, "pool_clean.json")
    if not os.path.isabs(pool_file):
        pool_file = os.path.join(SKILLS_DATA, pool_file)
    pool = []
    seen = set()
    for path in [os.path.join(SKILLS_DATA, "atomic_skills.json"), pool_file]:
        for c in load_json(path):
            sid = c.get("skill_id")
            if sid and sid not in seen:
                pool.append(c)
                seen.add(sid)
    return pool


def _split_id(sid):
    return " ".join(str(sid).split("_"))


def skill_doc(card):
    """Retrieval document text: name + tokenized skill_id + effect tokens + description."""
    effs = " ".join(sorted(skill_sig(card)))
    return f"{card.get('name', '')}. {_split_id(card.get('skill_id', ''))}. " \
           f"effects: {effs}. {card.get('description', '')}"


def skill_sig(card):
    """The skill's **declared** target-effect signature (token set). Note: distractors
    **misreport** their effects, so this is only for retrieval docs/display; for correctness
    use true_effect_sig below (recomputed from sub_skills)."""
    return frozenset(goal_signature(card.get("effects", [])))


# ---------- effect-grounded coverage (WP2a): recompute true effects from sub_skills ----------
# A distractor's declared effects may lie (lookalike writes heat as cool; missing drops a
# step but still lists everything). Correctness must be based on the "state changes actually
# produced", i.e. the effect signature synthesized in sub_skills order + whether the task
# object is placed into the task receptacle (at_receptacle). All methods use this so the
# reported metrics stay aligned with simulation.

def flatten_atomic(card, by_id, depth=0):
    """Expand into an atomic (skill_id, args) sequence; recurse if sub_skills reference a
    composite. An atomic card maps to itself."""
    if depth > 6:
        return []
    subs = card.get("sub_skills")
    if not subs:
        return [(card.get("skill_id"), card.get("params", []))]
    seq = []
    for st in subs:
        sid, args = st.get("skill"), st.get("args", [])
        child = by_id.get(sid)
        if child and child.get("sub_skills"):
            seq += flatten_atomic(child, by_id, depth + 1)
        else:
            seq.append((sid, args))
    return seq


def true_transform_sig(card, by_id):
    """The card's true "transform-type" target signature (isHot/isCold/isClean/isSliced/isFilled,
    with thermal states mutually exclusive/overriding).
    **Excludes at_receptacle**: whether placement hits must be judged against the task
    receptacle (see places_at_target)."""
    subs = card.get("sub_skills")
    if not subs:
        toks = set(goal_signature(card.get("effects", [])))
        toks.discard("at_receptacle")
        return toks
    token_sets = []
    for sid, _ in flatten_atomic(card, by_id):
        child = by_id.get(sid)
        if child:
            token_sets.append(goal_signature(child.get("effects", [])))
    sig = set(combine_sequential(token_sets))
    sig.discard("at_receptacle")
    return sig


def places_at_target(card, by_id, binding):
    """Whether the card places the "task object" into the "task receptacle": take the last
    put_object; its receptacle must be the placeholder `receptacle` (which binds to the task
    receptacle) or a hardcoded value exactly equal to the task receptacle. Blocks the
    "wrong receptacle" lookalike."""
    binding = binding or {}
    obj, recep = binding.get("obj"), binding.get("receptacle")
    last_put = None
    for sid, args in flatten_atomic(card, by_id):
        if sid == "put_object":
            last_put = args
    if not last_put:
        return False
    put_obj = last_put[0] if len(last_put) > 0 else None
    put_rec = last_put[1] if len(last_put) > 1 else None
    obj_ok = put_obj in ("obj", "target") or (obj and put_obj == obj)
    rec_ok = put_rec == "receptacle" or (recep and put_rec == recep)
    return bool(obj_ok and rec_ok)


def true_effect_sig(card, by_id, binding=None):
    """The **true** effective coverage set for a given task binding: transform signature +
    (at_receptacle if the task receptacle is hit). For distractors it faithfully reflects
    "what can actually be achieved"; this is the primitive for effect-grounded coverage."""
    sig = set(true_transform_sig(card, by_id))
    if places_at_target(card, by_id, binding):
        sig.add("at_receptacle")
    return sig


# ---------- instruction -> target-effect parsing (same source as M1 and every baseline:
# consumes only the instruction, never reads GT/labels) ----------
# Placed in common so M1 and the WP2b verification scripts share one rule set (single source of truth).
_GOAL_KEYWORDS = [
    # Note: "microwave" as a "receptacle" (put ... in the microwave) does not heat; only as a
    # **verb** (microwave the/a/it ...) does it trigger isHot; otherwise "putting into the
    # microwave" would be misread as heating.
    ("isHot", r"\bheat|\bhot\b|warm|\bcook|boil|roast|microwaved|\bmicrowave\s+(a|an|the|some|this|it|your)\b"),
    # fridge/refrigerator is often a "receptacle" (place in fridge) and does not trigger isCold;
    # only explicit cooling verbs count (including the refrigerate/refrigerated verb, but the
    # noun "refrigerator" does not match -- different word form).
    ("isCold", r"\bcool|\bchill|\bcold\b|frozen|freeze|refrigerate"),
    ("isClean", r"clean|wash|rins"),
    ("isSliced", r"slice|\bcut\b|chop|\bpiece|\bslices?\b"),
    ("isFilledWithLiquid", r"\bfill|filled|pour"),
]
_GOAL_RE = [(tok, re.compile(pat)) for tok, pat in _GOAL_KEYWORDS]


def parse_goal(instruction, binding=None):
    """Parse the target-effect token set from the instruction text. A receptacle binding or
    placement verb -> at_receptacle."""
    s = (instruction or "").lower()
    goal = set()
    for tok, rx in _GOAL_RE:
        if rx.search(s):
            goal.add(tok)
    binding = binding or {}
    if binding.get("receptacle") or re.search(r"\bput|place|\bon\b|\binto\b|\bin\b", s):
        goal.add("at_receptacle")
    return goal


_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text):
    """Tokenize English by word and CJK by single character (used by BM25)."""
    return _TOKEN_RE.findall(str(text).lower())


def load_queries(n=None, seed=0, gt_path=None):
    """Balanced sampling of queries from gt_kitchen.jsonl. Each item has instruction +
    required_effects + gold-standard set (id level) + binding/floor_plan (for simulation).
    n=None returns all."""
    gt_path = gt_path or os.path.join(GT_DATA, "gt_kitchen.jsonl")
    rows = []
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if n is None:
        return rows
    rng = random.Random(seed)
    by_type = {}
    for r in rows:
        by_type.setdefault(r["task_type"], []).append(r)
    per = max(1, n // max(1, len(by_type)))
    sample = []
    for _, lst in by_type.items():
        lst = list(lst)
        rng.shuffle(lst)
        sample += lst[:per]
    rng.shuffle(sample)
    return sample[:n]
