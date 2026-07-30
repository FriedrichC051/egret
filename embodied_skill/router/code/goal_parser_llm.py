"""L0-zero: LLM few-shot target-effect parser (zero-training version, the L0 of the production M1 pipeline).

**Same signature and same output** as the rule-based parse_goal -- parses from the instruction a
fixed 6-word effect set:
  {isHot, isCold, isClean, isSliced, isFilledWithLiquid, at_receptacle}
So it can seamlessly replace parse_goal in domain_m1, leaving downstream L2 legality filter /
L4 set-cover / L5 repair untouched.

Design points:
  - Closed-set ontology: the prompt enumerates the 6 legal tokens and explains each one,
    forbidding inventing new words (first gate against hallucination).
  - few-shot picks hard cases: microwave/fridge "as receptacle vs as verb" ambiguity,
    synonym paraphrases, multi-step conjunctions.
  - Structured output + post-validation: keep only tokens in the 6-word vocabulary; if JSON
    parsing fails -> fall back to the rule-based parse_goal.
  - at_receptacle prefers to trust the binding (the receptacle instance comes from the task;
    don't let the LLM guess).
  - temperature=0 + per-instruction disk cache (each of the 211 items is called once, with zero
    duplicate cost across methods/reruns).
Fairness red line: consumes only instruction (+ binding receptacle name), **never reads
required_effects (GT)**, same source as M1/baselines.

Standalone self-check (on the 211-item eval set, reporting LLM vs rule vs GT exact-match):
  python router/code/goal_parser_llm.py --n 30
"""
import os
import re
import sys
import json
import time
import hashlib
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import parse_goal  # noqa: E402  rule fallback

VOCAB = {"isHot", "isCold", "isClean", "isSliced", "isFilledWithLiquid", "at_receptacle"}
_CACHE_PATH = os.path.join(HERE, os.pardir, os.pardir, "eval", "datasets", "goal_llm_cache.json")

SYSTEM = (
    "You are a target-effect parser for embodied household tasks. Given a natural-language "
    "instruction, decide which final-state effects the task requires the object to reach. "
    "Choose only from the following fixed 6 labels; inventing new labels is forbidden:\n"
    "- isHot              object is heated (heat/cook/microwave as a verb/nuke/warm/boil/roast/bake)\n"
    "- isCold             object is cooled (cool/chill/freeze/refrigerate as a verb)\n"
    "- isClean            object is cleaned (clean/wash/rinse/scrub)\n"
    "- isSliced           object is sliced (slice/cut/chop/section)\n"
    "- isFilledWithLiquid container is filled with liquid (fill/pour water)\n"
    "- at_receptacle      object is placed at a specified receptacle/surface (put/place/on/into)\n"
    "Note: \"put it IN the microwave/fridge\" treats the microwave/fridge as a **receptacle** "
    "(counts as at_receptacle only), not heating/cooling; only \"microwave/heat the X\" or "
    "\"cool/chill the X\" as a **verb** counts as a state change.\n"
    "Key: a state adjective (clean/chilled/heated/sliced/washed modifying the object, e.g. "
    "\"clean lettuce\", \"chilled apple\", \"a slice of X\") means **the robot must achieve that "
    "state** (go wash/cool/slice it), not that the object already is that way -- always count it "
    "as the corresponding effect.\n"
    "Output only a JSON array of strings, e.g. [\"isHot\",\"at_receptacle\"], with no explanation."
)

FEWSHOT = (
    "Instruction: microwave the potato and put it on the countertop\nOutput: [\"isHot\",\"at_receptacle\"]\n\n"
    "Instruction: put the mug in the microwave\nOutput: [\"at_receptacle\"]\n\n"
    "Instruction: rinse the lettuce then chill it in the fridge\nOutput: [\"isClean\",\"isCold\",\"at_receptacle\"]\n\n"
    "Instruction: slice the apple\nOutput: [\"isSliced\"]\n\n"
    "Instruction: cut a slice of tomato, heat it, and place it on the table\n"
    "Output: [\"isSliced\",\"isHot\",\"at_receptacle\"]\n\n"
    "Instruction: place the spatula on the table\nOutput: [\"at_receptacle\"]\n\n"
    "Instruction: place clean lettuce inside of the kitchen fridge\nOutput: [\"isClean\",\"at_receptacle\"]\n\n"
    "Instruction: put a chilled pan on the counter\nOutput: [\"isCold\",\"at_receptacle\"]\n\n"
    "Instruction: put a slice of apple on the table\nOutput: [\"isSliced\",\"at_receptacle\"]\n\n"
)


def _parse_arr(text):
    """Extract the JSON array from the LLM output and keep only legal tokens; return None on failure."""
    if not text:
        return None
    s, e = text.find("["), text.rfind("]")
    if s == -1 or e == -1:
        return None
    try:
        arr = json.loads(text[s:e + 1])
    except Exception:
        return None
    return {x for x in arr if isinstance(x, str) and x in VOCAB}


class LLMGoalParser:
    """LLM few-shot goal parser with disk cache + rule fallback. Callable: parse(instruction, binding)."""

    def __init__(self, cache_path=_CACHE_PATH, temperature=1, verbose=False):
        # Note: Kimi highspeed only allows temperature=1; stability/reproducibility relies on few-shot + closed set + disk cache
        self.cache_path = cache_path
        self.temperature = temperature
        self.verbose = verbose
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            try:
                self.cache = json.load(open(cache_path, encoding="utf-8"))
            except Exception:
                self.cache = {}
        self.n_call = self.n_hit = self.n_fallback = 0

    def _flush(self):
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            json.dump(self.cache, open(self.cache_path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=0)

    def parse(self, instruction, binding=None):
        key = hashlib.md5((instruction or "").encode("utf-8")).hexdigest()
        if key in self.cache:
            self.n_hit += 1
            goal = set(self.cache[key])
        else:
            from llm_baselines import llm_call
            goal = None
            try:
                self.n_call += 1
                user = f"{FEWSHOT}Instruction: {instruction}\nOutput:"
                text = llm_call(SYSTEM, user, temperature=self.temperature)
                goal = _parse_arr(text)
            except Exception as ex:
                if self.verbose:
                    print(f"  [L0] LLM call error, rule fallback: {ex}", flush=True)
                goal = None
            if goal is None:                          # parsing failed -> rule fallback
                self.n_fallback += 1
                goal = parse_goal(instruction, binding)
            self.cache[key] = sorted(goal)
            self._flush()
        # at_receptacle prefers to trust the binding (the receptacle instance comes from the task; don't let the LLM guess)
        binding = binding or {}
        if binding.get("receptacle"):
            goal.add("at_receptacle")
        return goal

    # same signature as parse_goal for a drop-in replacement
    def __call__(self, instruction, binding=None):
        return self.parse(instruction, binding)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    from common import load_queries
    queries = load_queries(
        n=(None if args.n < 0 else args.n), seed=0,
        gt_path=os.path.join(HERE, os.pardir, os.pardir, "gt", "datasets", "gt_kitchen_eval.jsonl"))
    llm = LLMGoalParser(verbose=True)
    t0 = time.time()
    llm_ok = rule_ok = 0
    mism = []
    for q in queries:
        req = set(q["required_effects"])
        g_llm = llm.parse(q["instruction"], q.get("binding"))
        g_rule = parse_goal(q["instruction"], q.get("binding"))
        llm_ok += int(g_llm == req)
        rule_ok += int(g_rule == req)
        if g_llm != req and len(mism) < 15:
            mism.append((q["instruction"][:70], sorted(g_llm), sorted(g_rule), sorted(req)))
    n = len(queries)
    print(f"\nN={n}  LLM calls {llm.n_call} (cache hits {llm.n_hit}, fallbacks {llm.n_fallback})"
          f"  time {time.time()-t0:.1f}s")
    print(f"exact-match: LLM = {llm_ok}/{n} = {llm_ok/n:.1%}   rule = {rule_ok}/{n} = {rule_ok/n:.1%}")
    print("\nExamples the LLM still parses wrong (instruction | LLM | rule | GT):")
    for ins, l, r, g in mism:
        print(f"  - {ins!r}\n    LLM={l}  rule={r}  GT={g}")


if __name__ == "__main__":
    main()
