"""Phase 4 domain method M1 (pilot, training-free): B3 skeleton + two "domain hooks".

Replace SkillFlow (B3)'s last two layers with domain-aware modules that leverage the
**structured information** of skill cards -- things a generic text pipeline (B0/B3) cannot use:

  (1) Retrieval layer : Dense top-K (kept, same as B3)
  (2) Rerank layer -> domain filter: run a "hand-state symbolic simulation" on candidates and
              drop physically illegal chains (put/heat/cool/clean/slice without first holding
              the corresponding object/knife -> illegal). This cuts all reorder / missing-pickup
              distractors and also filters out inconsistent bare atomics.
  (3) Selection layer -> combined coverage: use combine_sequential to **recompute the true
              effect signature** from sub_skills (a lookalike swapping heat for cool -> the true
              signature becomes isCold; missing drops a token), then greedy set-cover over the
              "goal parsed from the instruction", picking the minimal exact covering set.

Fairness: the goal is parsed only from instruction keywords (like B0-B3, it consumes only the
instruction), **does not read** the is_distractor/why_wrong/variant_type labels, and **does not
use** the GT required_effects.

Standalone self-check (with goal-parsing accuracy diagnostics):
  python router/code/domain_m1.py --n 20
"""
import os
import re
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (load_pool, load_queries, skill_sig, parse_goal,  # noqa: E402
                    flatten_atomic, true_transform_sig, places_at_target, true_effect_sig)
from retrievers import DenseRetriever                  # noqa: E402

SKILLS_CODE = os.path.join(HERE, os.pardir, os.pardir, "skills", "code")
sys.path.insert(0, SKILLS_CODE)
from skill_effects import goal_signature, combine_sequential  # noqa: E402

# ---- sub-skill categories (for hand-state simulation + effect derivation) ----
_HOLD_OPS = {"heat_object_microwave", "cool_object_fridge",
             "clean_object_sink", "fill_with_liquid", "empty_liquid"}
_PASSIVE = {"open_receptacle", "close_receptacle", "toggle_on", "toggle_off",
            "goto_object"}


def is_legal(card, by_id):
    """Hand-state symbolic simulation: starting empty-handed, step through and check whether
    preconditions are satisfied.
    Catches: reorder (put before pick), missing pickup, and bare put/heat/slice-style
    inconsistent cards.
    (Effect recomputation transform_sig/places_at_target/flatten has been factored into common,
    single source of truth.)"""
    hand = None  # None = empty hand; otherwise the name of the held object
    for sid, args in flatten_atomic(card, by_id):
        a0 = args[0] if args else None
        if sid == "pickup_object":
            if hand is not None:            # hand not empty but trying to pick up again
                return False
            hand = a0
        elif sid == "put_object":
            if hand is None or hand != a0:  # not holding the thing to be put down
                return False
            hand = None
        elif sid in _HOLD_OPS:
            if hand is None or hand != a0:  # must hold the object before heat/cool/clean/fill
                return False
        elif sid == "slice_object":
            if hand is None or "knife" not in str(hand).lower():  # slicing requires holding a knife
                return False
        elif sid in _PASSIVE:
            pass
        # unknown skill: pass leniently (avoid false kills)
    return True


class M1RuleDomain:
    """Rule-based M1: Dense top-K -> legality filter -> goal set-cover (no LLM).

    Coverage judgment: transform-type effects are recomputed from sub_skills; at_receptacle is
    judged per-task against the task receptacle."""
    name = "M1-Rule"
    wants_binding = True   # run_pilot passes each query's binding into select

    def __init__(self, pool, k=20, retriever=None, goal_fn=None, name=None,
                 l4_scorer=None, use_retrieval=False,
                 no_legality=False, no_repair=False, use_declared=False):
        self.pool = pool
        self.by_id = {c["skill_id"]: c for c in pool}
        # WP5 axis-5 anti-overfitting ablation: remove symbolic scaffolding to test whether a
        #   pure-neural L4 overfits/over-selects.
        #   no_legality=True -> disable the L2 legality hard mask (reorder/missing-pickup/bare
        #     atomics also enter candidates);
        #   no_repair=True   -> disable "coverage-first set-cover + fallback", switching to a
        #     pure-L4 selection of candidates with retention score > 0.
        self.no_legality = no_legality
        self.no_repair = no_repair
        # Anti-confusion layer-wise attribution ablation (WP-attr): disable L4's "effect
        #   recomputation grounding" and instead run set-cover on the card's **declared** effects
        #   (distractors misreport effects -> should be fooled). Used to prove "effect
        #   recomputation" is the source of confusion resistance.
        self.use_declared = use_declared
        # Drop L1 (default): no retrieval; all legal L2 candidates in the full pool go straight
        #   into set-cover (for a realistic 10^1-10^2 small library retrieval is unnecessary, and
        #   K truncation instead drops the correct skill; see stage2_execution_plan
        #   "zero-training probe - L1 decision").
        #   use_retrieval=True only for ablation / to reuse retrieval when the library scales to 10^4+.
        self.use_retrieval = use_retrieval
        self.k = k if use_retrieval else len(pool)
        self.retriever = (retriever or DenseRetriever(pool)) if use_retrieval else None
        # L0 goal parser: defaults to the rule-based parse_goal; the production version can pass
        # LLMGoalParser (same signature)
        self.goal_fn = goal_fn or parse_goal
        # L4 discriminative retention scorer (optional): given (goal, legal candidates) returns a
        # per-candidate retention score, used as a selection key only after "coverage-first";
        # the main system uses None and takes the pure-rule set-cover path.
        self.l4_scorer = l4_scorer
        if name:
            self.name = name
        # query-independent quantities can be cached
        self._tsig = {c["skill_id"]: true_transform_sig(c, self.by_id) for c in pool}
        self._legal = {c["skill_id"]: is_legal(c, self.by_id) for c in pool}

    def eff_sig(self, card, binding):
        """The effective coverage set for a given task binding: transform signature +
        (at_receptacle if the task receptacle is hit).
        When use_declared=True (ablation), use the card's **declared** effects instead (no
        recomputation, no grounding) -- a distractor's misreported declared values slip through,
        used to isolate the contribution of "effect recomputation grounding" to confusion
        resistance."""
        if self.use_declared:
            return set(skill_sig(card))
        sig = set(self._tsig[card["skill_id"]])
        if places_at_target(card, self.by_id, binding):
            sig.add("at_receptacle")
        return sig

    def _cover(self, cands, goal, scores=None):
        """Greedy set-cover: each round pick (max new coverage -> [high L4 learned retention
        score] -> min extra effects -> earlier rank).
        scores: optional {skill_id: float} learned score; **always coverage-first (structural
        lower bound unchanged)**, the learned score only breaks ties among candidates with
        "equal new coverage" -- exactly where lookalike/variant disambiguation happens."""
        remaining = set(goal)
        chosen = []
        pool = list(cands)  # (rank, card, sig)
        while remaining:
            best, best_key = None, None
            for rank, card, sig in pool:
                new = len(sig & remaining)
                if new == 0:
                    continue
                extra = len(sig - goal)          # fewer is better: don't do unrequested transforms
                sc = -scores.get(card["skill_id"], 0.0) if scores else 0.0  # higher learned score first
                key = (-new, sc, extra, rank)     # coverage-first -> learned score -> less extra -> earlier
                if best_key is None or key < best_key:
                    best, best_key = (card, sig), key
            if best is None:
                break                             # no one can cover more -> stop
            card, sig = best
            chosen.append(card)
            remaining -= sig
            pool = [(r, c, s) for (r, c, s) in pool if c["skill_id"] != card["skill_id"]]
        return chosen

    def legal_candidates(self, query, binding=None):
        """[L1 dropped] full pool in original order -> L2 legality filter; (L1 retrieval top-K
        only when use_retrieval=True).
        Returns [(rank, card, eff_sig)]. Shared by L4 training/inference."""
        cand = self.retriever.rank(query)[:self.k] if self.use_retrieval else self.pool
        return [(i, c, self.eff_sig(c, binding))
                for i, c in enumerate(cand)
                if self.no_legality or self._legal[c["skill_id"]]]

    def select(self, query, binding=None):
        goal = self.goal_fn(query, binding)
        legal = self.legal_candidates(query, binding)
        if not goal:                              # goal not parsed -> degrade to top-1 legal
            return [legal[0][1]] if legal else []
        scores = None
        if self.l4_scorer is not None and legal:
            scores = self.l4_scorer.score(goal, legal)
        # Axis-5 pure-neural ablation: drop coverage/fallback, keep only candidates with L4
        # retention score > 0 (to examine over-selection/overfitting).
        if self.no_repair and scores is not None:
            chosen = [c for (_, c, _) in legal if scores.get(c["skill_id"], -1e9) > 0.0]
            if not chosen and legal:              # all negative still needs non-empty output -> take the top-scoring one
                best = max(legal, key=lambda t: scores.get(t[1]["skill_id"], -1e9))
                chosen = [best[1]]
            return chosen
        chosen = self._cover(legal, goal, scores)
        if not chosen and legal:                  # fallback: even with no coverage, return the earliest legal card
            chosen = [legal[0][1]]
        return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()
    pool = load_pool()
    queries = load_queries(n=args.n, seed=0)
    m1 = M1RuleDomain(pool, k=args.k)
    n_legal = sum(m1._legal.values())
    print(f"Skill pool: {len(pool)} cards (legal {n_legal} / illegal {len(pool)-n_legal}); "
          f"M1 retrieval top{args.k} -> legality filter -> goal set-cover\n")

    cov_ok = goal_ok = 0
    for q in queries:
        t = time.time()
        chosen = m1.select(q["instruction"], q.get("binding"))
        dt = time.time() - t
        req = set(q["required_effects"])
        parsed = parse_goal(q["instruction"], q.get("binding"))
        union = set()
        for c in chosen:
            union |= m1.eff_sig(c, q.get("binding"))
        cov = union >= req
        gok = parsed == req
        cov_ok += cov
        goal_ok += gok
        print(f"[{dt*1000:5.0f}ms] parsed={sorted(parsed)} {'OK' if gok else '!=GT '+str(sorted(req))}")
        print(f"          -> {[c['skill_id'] for c in chosen]} covers_GT={'OK' if cov else 'X'}")
    n = len(queries)
    print(f"\ngoal parse == GT: {goal_ok}/{n}={goal_ok/n:.0%}; "
          f"selected set covers GT: {cov_ok}/{n}={cov_ok/n:.0%}")


if __name__ == "__main__":
    main()
