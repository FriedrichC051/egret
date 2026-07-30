"""WP8 output 3: confusion-density curve.

The three-tier pools pool_tier_{low,mid,high} share the same 34 gold-standard baseline, changing
only the distractor density (1/3/6). To make "confusion" the sole independent variable, we only
evaluate on queries [whose answer lies within the 34-baseline subset] (otherwise a tier pool has
no correct skill, which is unfair); all three tiers share the same answer set = a controlled comparison.

Metrics (effect-grounded, consistent with WP2a):
- selection (m1/b0/b3): Sel-FullCov (fraction where the union of chosen true effects ⊇ required_effects)
- retrieval (bm25/dense): FullCov@1 (top-1 true-effect coverage)

Usage (Windows, skill-router environment):
  python eval/code/run_confusion_curve.py --methods m1 bm25 dense
  python eval/code/run_confusion_curve.py --methods m1 bm25 dense b0 b3   # includes LLM, slow
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ROUTER_CODE = os.path.join(ROOT, "router", "code")
SKILLS_DATA = os.path.join(ROOT, "skills", "datasets")
GT_DATA = os.path.join(ROOT, "gt", "datasets")
EVAL_DATA = os.path.join(ROOT, "eval", "datasets")
sys.path.insert(0, ROUTER_CODE)
sys.path.insert(0, HERE)
import common  # noqa: E402
import metrics as M  # noqa: E402

TIERS = ["low", "mid", "high"]


def balanced_subsample(queries, n, seed=0):
    """Sample n queries balanced by task_type (take all if fewer), to cap LLM-curve cost."""
    import random
    if n is None or n <= 0 or n >= len(queries):
        return queries
    rng = random.Random(seed)
    by_type = {}
    for q in queries:
        by_type.setdefault(q.get("task_type", "?"), []).append(q)
    per = max(1, n // max(1, len(by_type)))
    out = []
    for lst in by_type.values():
        lst = list(lst)
        rng.shuffle(lst)
        out += lst[:per]
    rng.shuffle(out)
    return out[:n]


def load_answerable(subset_ids, queries, full_by_id):
    """A query is answerable = some gold standard in the subset has true effect signature ⊇ required_effects."""
    subset_cards = [full_by_id[i] for i in subset_ids if i in full_by_id]
    ans = []
    for q in queries:
        req = set(q["required_effects"])
        binding = q.get("binding")
        if any(common.true_effect_sig(c, full_by_id, binding) >= req for c in subset_cards):
            ans.append(q)
    return ans


def build_method(name, pool):
    if name == "m1":
        from domain_m1 import M1RuleDomain
        return M1RuleDomain(pool), "sel"
    if name == "bm25":
        from retrievers import BM25Retriever
        return BM25Retriever(pool), "ret"
    if name == "dense":
        from retrievers import DenseRetriever
        return DenseRetriever(pool), "ret"
    if name == "b0":
        from llm_baselines import B0LLMFull
        return B0LLMFull(pool), "sel"
    if name == "b3":
        from pipelines import B3SkillFlow
        return B3SkillFlow(pool), "sel"
    if name == "b4":
        from lota_b4 import B4LoTaBench
        batch = int(os.environ.get("B4_BATCH", "16"))
        return B4LoTaBench(pool, batch_size=batch, norm="mean"), "sel"
    if name == "m1h":
        from domain_m1 import M1RuleDomain
        from l0_head import L0HeadParser
        return M1RuleDomain(pool, goal_fn=L0HeadParser(), name="M1-Head"), "sel"
    if name == "m1h4":
        from domain_m1 import M1RuleDomain
        from l0_head import L0HeadParser
        from l4_head import L4HeadScorer
        return M1RuleDomain(pool, goal_fn=L0HeadParser(), l4_scorer=L4HeadScorer(),
                            name="M1-Head+L4"), "sel"
    raise ValueError(name)


def eval_method(obj, kind, queries, by_id, mname=None, tier=None, cache=None, cache_path=None):
    """LLM selection methods (b0/b3) with a persistent cache: each query's successful chosen is written to disk,
    so on a rerun only queries missing from the cache (i.e. previously timed out/failed) call the API again ->
    only backfill timeouts, do not rerun everything."""
    hit = 0.0
    n = len(queries)
    wants_binding = getattr(obj, "wants_binding", False)
    use_cache = cache is not None and kind == "sel" and mname in ("b0", "b3", "b4")
    n_reused = n_called = n_failed = 0
    for i, q in enumerate(queries):
        req = q["required_effects"]
        binding = q.get("binding")
        if kind == "ret":
            ranked = obj.rank(q["instruction"])
            hit += M.fullcov_at_k(ranked, req, 1, by_id, binding)
            continue
        ck = f"{mname}|{tier}|{q.get('task_id')}"
        if use_cache and ck in cache:
            chosen = cache[ck]
            n_reused += 1
        else:
            chosen, ok = [], False
            for attempt in range(5):  # retry on LLM API timeout/connection errors
                try:
                    chosen = obj.select(q["instruction"], binding) if wants_binding else obj.select(q["instruction"])
                    ok = True
                    break
                except Exception as ex:
                    print(f"    [warn] select retry {attempt+1}/5: {ex}", flush=True)
                    time.sleep(3 * (attempt + 1))
            if ok:
                n_called += 1
                if use_cache:
                    cache[ck] = chosen
                    if n_called % 10 == 0 and cache_path:  # incremental flush to disk, avoid losing progress on interruption
                        json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
            else:
                n_failed += 1
                print("    [warn] select still failing after 5 tries; this query is not cached (will be retried next run)", flush=True)
        hit += M.selection_scores(chosen, req, by_id, binding)["sel_fullcov"]
    if use_cache and cache_path:
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
    if kind == "sel" and (n_reused or n_failed):
        print(f"    [cache] reused {n_reused} / new calls {n_called} / still failed {n_failed}", flush=True)
    return hit / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["m1", "bm25", "dense"],
                    choices=["m1", "bm25", "dense", "b0", "b3", "b4", "m1h", "m1h4"])
    ap.add_argument("--merge", action="store_true",
                    help="load results from an existing out file, recompute only this run's methods and merge (keep other methods' curves)")
    ap.add_argument("--gt", default=os.path.join(GT_DATA, "gt_kitchen_eval.jsonl"))
    ap.add_argument("--out", default=os.path.join(EVAL_DATA, "confusion_curve.json"))
    ap.add_argument("--subsample", type=int, default=0, help="if >0, subsample balanced by task_type to cap LLM cost")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=os.path.join(EVAL_DATA, "confusion_llm_cache.json"),
                    help="per-query selection cache for b0/b3; a rerun only backfills queries missing from the cache (timed out)")
    args = ap.parse_args()

    cache = {}
    if os.path.exists(args.cache):
        cache = json.load(open(args.cache, encoding="utf-8"))
        print(f"[cache] loaded {len(cache)} cached LLM selection results")

    manifest = json.load(open(os.path.join(SKILLS_DATA, "tiers_manifest.json"), encoding="utf-8"))
    subset_ids = manifest["subset_base_ids"]
    queries = common.load_queries(n=None, seed=0, gt_path=args.gt)

    # Use the full-pool by_id to determine the "answerable subset" (gold-standard cards match those inside each tier)
    os.environ.pop("POOL_FILE", None)
    full_pool = common.load_pool()
    full_by_id = {c["skill_id"]: c for c in full_pool}
    answerable = load_answerable(subset_ids, queries, full_by_id)
    n_ans_full = len(answerable)
    if args.subsample and args.subsample > 0:
        answerable = balanced_subsample(answerable, args.subsample, args.seed)
    print(f"subset baselines {len(subset_ids)}; full eval {len(queries)} queries; "
          f"answerable subset {n_ans_full} queries -> actually evaluated {len(answerable)} queries (shared across three tiers)\n")

    results = {m: {} for m in args.methods}
    prev = {}
    if args.merge and os.path.exists(args.out):
        prev = json.load(open(args.out, encoding="utf-8")).get("results", {})
    for tier in TIERS:
        os.environ["POOL_FILE"] = f"pool_tier_{tier}.json"
        pool = common.load_pool()
        by_id = {c["skill_id"]: c for c in pool}
        ndist = sum(1 for c in pool if c.get("is_distractor"))
        print(f"==== tier {tier}: pool {len(pool)} entries (distractors {ndist}) ====")
        for m in args.methods:
            t = time.time()
            obj, kind = build_method(m, pool)
            score = eval_method(obj, kind, answerable, by_id,
                                mname=m, tier=tier, cache=cache, cache_path=args.cache)
            results[m][tier] = round(score, 4)
            print(f"  {m:5} ({kind}) FullCov = {score:.3f}   ({time.time()-t:.1f}s)")
        print()

    # Merge historical methods (--merge keeps other methods' curves)
    merged = dict(prev)
    merged.update(results)

    # Summary table
    print("======== confusion-density curve (FullCov, effect-grounded) ========")
    print(f"{'method':6} {'low(1)':>8} {'mid(3)':>8} {'high(6)':>8} {'drop':>8}")
    for m in merged:
        r = merged[m]
        if not all(t in r for t in TIERS):
            continue
        drop = r["low"] - r["high"]
        print(f"{m:6} {r['low']:8.3f} {r['mid']:8.3f} {r['high']:8.3f} {drop:+8.3f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"n_answerable": len(answerable), "n_answerable_full": n_ans_full,
                   "subsample": args.subsample, "seed": args.seed,
                   "methods": sorted(merged), "subset_bases": len(subset_ids),
                   "tiers": {t: manifest["pools"][t] for t in TIERS},
                   "results": merged}, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
