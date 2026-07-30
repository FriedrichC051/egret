"""WP1: Build "confusion-tier" skill pools from pool_clean.json, used for the
SR-vs-confusion curve in WP8.

The confusion curve requires a "controlled, uniform" density axis: the three
tier pools share the same set of gold-standard bases and only change "how many
distractors each gold base gets" (low=1 / mid=3 / high=6). To keep it uniform,
only gold bases with "available variants >= the highest tier k" are chosen as
the shared base subset; each tier samples as evenly as possible across the
reorder/missing/lookalike types.

Output (under skills/datasets/, all in the drop-in format directly consumable by
load_pool):
  pool_tier_low.json / pool_tier_mid.json / pool_tier_high.json
  tiers_manifest.json (records the base subset, per-tier density, and total pool
  size, for paper/reproduction traceability)

Usage:
  python build_tiers.py                      # default tiers 1/3/6
  python build_tiers.py --tiers 1 3 6 --seed 0
"""
import os
import sys
import json
import random
import argparse
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "datasets")


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def dump_json(p, obj):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def balanced_pick(variants, k, rng):
    """Pick k variants from one base, balanced across variant_type as much as possible."""
    by_type = defaultdict(list)
    for v in variants:
        by_type[v.get("variant_type", "unknown")].append(v)
    for t in by_type:
        rng.shuffle(by_type[t])
    picked, types = [], list(by_type.keys())
    rng.shuffle(types)
    # Round-robin one per type until k are picked or all are exhausted
    ti = 0
    while len(picked) < k and any(by_type[t] for t in types):
        t = types[ti % len(types)]
        if by_type[t]:
            picked.append(by_type[t].pop())
        ti += 1
    return picked[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", type=int, nargs="+", default=[1, 3, 6])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", default=os.path.join(DATA, "pool_clean.json"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    kmax = max(args.tiers)

    atomics = load_json(os.path.join(DATA, "atomic_skills.json"))
    pool = load_json(args.pool)
    golds = [c for c in pool if not c.get("is_distractor")]
    variants = [c for c in pool if c.get("is_distractor")]
    gold_ids = {c["skill_id"] for c in golds}

    # Group variants by variant_of (keep only variants pointing to an existing gold)
    by_base = defaultdict(list)
    orphan = 0
    for v in variants:
        b = v.get("variant_of")
        if b in gold_ids:
            by_base[b].append(v)
        else:
            orphan += 1

    # Shared base subset: golds with available variants >= kmax
    subset_bases = sorted([b for b, vs in by_base.items() if len(vs) >= kmax])
    gold_by_id = {c["skill_id"]: c for c in golds}
    subset_golds = [gold_by_id[b] for b in subset_bases]

    print(f"{len(golds)} golds; {len(variants)} distractor variants ({orphan} orphans)")
    print(f"{len(by_base)} bases with variants; {len(subset_bases)} bases with variants >= kmax({kmax}) -> used as shared subset")
    dist = Counter(len(vs) for vs in by_base.values())
    print("variants-per-base distribution:", dict(sorted(dist.items())))

    manifest = {"seed": args.seed, "tiers": args.tiers, "kmax": kmax,
                "n_gold_total": len(golds), "n_variant_total": len(variants),
                "subset_base_ids": subset_bases, "n_subset_bases": len(subset_bases),
                "atomic": len(atomics), "pools": {}}

    for k in args.tiers:
        tier_pool = list(subset_golds)  # gold subset (same across all three tiers)
        picked_types = Counter()
        for b in subset_bases:
            picks = balanced_pick(by_base[b], k, rng)
            tier_pool.extend(picks)
            for v in picks:
                picked_types[v.get("variant_type", "unknown")] += 1
        name = {1: "low", 3: "mid", 6: "high"}.get(k, f"k{k}")
        out = os.path.join(DATA, f"pool_tier_{name}.json")
        dump_json(out, tier_pool)
        total_with_atomic = len(atomics) + len(tier_pool)
        manifest["pools"][name] = {
            "k_per_base": k, "n_distractors": sum(picked_types.values()),
            "n_gold": len(subset_golds), "pool_clean_size": len(tier_pool),
            "total_routing_pool": total_with_atomic,
            "variant_type_breakdown": dict(picked_types),
        }
        print(f"  tier {name}(k={k}): gold {len(subset_golds)} + distractors {sum(picked_types.values())}"
              f" = pool_clean {len(tier_pool)}; +13 atomic = total pool {total_with_atomic}"
              f"; types {dict(picked_types)} -> {out}")

    dump_json(os.path.join(DATA, "tiers_manifest.json"), manifest)
    print("wrote tiers_manifest.json")


if __name__ == "__main__":
    main()
