"""WP1：从 pool_clean.json 构建"混淆度分档"技能池，用于 WP8 的 SR-vs-混淆度曲线。

混淆曲线要求一个"受控、均匀"的密度轴：三档池共享同一组金标准基准，仅改变
"每个金标准基准配几个干扰项"（低=1 / 中=3 / 高=6）。为保证均匀，只选取
"可用变体数 >= 最高档 k"的金标准基准作为共享基准子集；每档在 reorder/missing/
lookalike 三型间尽量均衡采样。

产出（skills/datasets/ 下，均为 load_pool 可直接消费的 drop-in 格式）：
  pool_tier_low.json / pool_tier_mid.json / pool_tier_high.json
  tiers_manifest.json （记录基准子集、各档密度、总池规模，供论文/复现追溯）

用法：
  python build_tiers.py                      # 默认档位 1/3/6
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
    """从一个基准的变体里挑 k 个，尽量在 variant_type 间均衡。"""
    by_type = defaultdict(list)
    for v in variants:
        by_type[v.get("variant_type", "unknown")].append(v)
    for t in by_type:
        rng.shuffle(by_type[t])
    picked, types = [], list(by_type.keys())
    rng.shuffle(types)
    # 轮转各类型取一个，直到取满 k 或取尽
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

    # 变体按 variant_of 归组（只保留指向现存金标准的变体）
    by_base = defaultdict(list)
    orphan = 0
    for v in variants:
        b = v.get("variant_of")
        if b in gold_ids:
            by_base[b].append(v)
        else:
            orphan += 1

    # 共享基准子集：可用变体数 >= kmax 的金标准
    subset_bases = sorted([b for b, vs in by_base.items() if len(vs) >= kmax])
    gold_by_id = {c["skill_id"]: c for c in golds}
    subset_golds = [gold_by_id[b] for b in subset_bases]

    print(f"金标准 {len(golds)} 条；干扰变体 {len(variants)} 条（孤儿 {orphan}）")
    print(f"有变体的基准 {len(by_base)} 个；变体数>= kmax({kmax}) 的基准 {len(subset_bases)} 个 -> 作共享子集")
    dist = Counter(len(vs) for vs in by_base.values())
    print("每基准变体数分布:", dict(sorted(dist.items())))

    manifest = {"seed": args.seed, "tiers": args.tiers, "kmax": kmax,
                "n_gold_total": len(golds), "n_variant_total": len(variants),
                "subset_base_ids": subset_bases, "n_subset_bases": len(subset_bases),
                "atomic": len(atomics), "pools": {}}

    for k in args.tiers:
        tier_pool = list(subset_golds)  # 金标准子集（三档一致）
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
        print(f"  tier {name}(k={k}): 金标准 {len(subset_golds)} + 干扰 {sum(picked_types.values())}"
              f" = pool_clean {len(tier_pool)}；+13 原子 = 总池 {total_with_atomic}"
              f"；类型 {dict(picked_types)} -> {out}")

    dump_json(os.path.join(DATA, "tiers_manifest.json"), manifest)
    print("已写 tiers_manifest.json")


if __name__ == "__main__":
    main()
