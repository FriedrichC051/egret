"""Phase 4 routing stage (Plan B: run in the Windows / skill-router environment, no simulation).

Responsibilities: run each routing method over the sampled queries. (1) Print retrieval/selection
metrics (Hit@K / FullCoverage@K / selection hit). (2) Write "the ranked result and the selected
skill set for each query and method" into selections.json, so the WSL-side simulation consumer
script run_sim_from_selections.py can read it and run SR/GC. The two stages are decoupled.

Methods:
  bm25  : B1 lexical retrieval, chosen = top-1
  dense : B2 dense retrieval (multilingual vectors), chosen = top-1
  b0    : B0 LLM stuffs the full prompt, chosen = the set selected by the LLM (calls Kimi, slow)

selections.json can be incrementally merged across multiple runs with the same (n, seed, pool_size)
(different methods are merged together).

Usage (Windows, conda skill-router environment):
  python eval\\code\\run_pilot.py --n 100 --methods bm25 dense
  python eval\\code\\run_pilot.py --n 100 --methods b0        # slow, merged into the same file
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))                     # embodied_skill/
ROUTER_CODE = os.path.join(ROOT, "router", "code")
EVAL_DATA = os.path.join(ROOT, "eval", "datasets")
sys.path.insert(0, ROUTER_CODE)
sys.path.insert(0, HERE)
from common import load_pool, load_queries, skill_sig  # noqa: E402
import metrics as M  # noqa: E402

KS = [1, 3, 5, 10]
TOPN_STORE = 20   # ranking length stored for retrieval methods in selections.json


def q_key(q):
    return q["task_id"]


def q_meta(q):
    """Common per-query info written into selections.json (used by the simulation)."""
    return {
        "task_id": q["task_id"],
        "task_type": q["task_type"],
        "split": q.get("split"),
        "floor_plan": q["floor_plan"],
        "binding": q["binding"],
        "instruction": q["instruction"],
        "required_effects": q["required_effects"],
        "gt_ids": sorted(set(q.get("gt_composite_set", [])) | set(q.get("gt_atomic_set", []))),
    }


def eval_retriever(retriever, queries, entries, by_id):
    agg = {f"hit@{k}": 0 for k in KS}
    agg.update({f"fc@{k}": 0 for k in KS})
    agg.update({f"idhit@{k}": 0 for k in KS})
    t0 = time.time()
    for q in queries:
        ranked = retriever.rank(q["instruction"])
        req = q["required_effects"]
        binding = q.get("binding")
        gt_ids = set(q.get("gt_composite_set", [])) | set(q.get("gt_atomic_set", []))
        for k in KS:
            agg[f"hit@{k}"] += M.hit_at_k(ranked, req, k, by_id, binding)
            agg[f"fc@{k}"] += M.fullcov_at_k(ranked, req, k, by_id, binding)
            agg[f"idhit@{k}"] += M.id_hit_at_k(ranked, gt_ids, k)
        ranked_ids = [c["skill_id"] for c in ranked[:TOPN_STORE]]
        entries[q_key(q)]["methods"][retriever.name] = {
            "ranked_ids": ranked_ids,
            "chosen_ids": ranked_ids[:1],   # retrieval method: chosen = top-1
        }
    dt = time.time() - t0
    n = len(queries)
    return {k: v / n for k, v in agg.items()}, dt


def dump_selections(path, entries, queries, n, seed, pool_size):
    """Write selections.json through a temp file + os.replace, so an interrupted run never
    truncates it. Called after every query for the API-backed methods, which makes a
    half-finished run resumable via --only-missing instead of restarting from scratch."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = {
        "meta": {"n": n, "seed": seed, "pool_size": pool_size,
                 "generated_at": datetime.now().isoformat(timespec="seconds")},
        "queries": [entries[q_key(q)] for q in queries],
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return out


def eval_selector(selector, queries, entries, by_id, retries=1, only_missing=False,
                  flush=None):
    """LLM/pipeline selection methods: directly produce the chosen set, scored on sel_hit/sel_fullcov.
    Per-query fault tolerance: on timeout/exception, retry a single query up to `retries` times; if it
    still fails, record it as empty and continue (so one query does not drag down the whole run)."""
    sel_hit = sel_fc = size = 0
    fails = 0
    n = len(queries)
    t0 = time.time()
    for i, q in enumerate(queries):
        # Backfill-only mode: skip queries that already have a non-empty chosen (avoid slow/expensive LLM recompute)
        if only_missing:
            prev = entries[q_key(q)]["methods"].get(selector.name)
            if prev and prev.get("chosen_ids"):
                chosen = [selector.by_id[c] for c in prev["chosen_ids"] if c in selector.by_id] \
                    if hasattr(selector, "by_id") else []
                req = q["required_effects"]
                s = M.selection_scores(chosen, req, by_id, q.get("binding"))
                sel_hit += s["sel_hit"]; sel_fc += s["sel_fullcov"]; size += s["sel_size"]
                continue
        qt = time.time()
        chosen = None
        wants_binding = getattr(selector, "wants_binding", False)
        for attempt in range(retries + 1):
            try:
                chosen = selector.select(q["instruction"], q.get("binding")) \
                    if wants_binding else selector.select(q["instruction"])
                break
            except Exception as ex:
                if attempt < retries:
                    continue
                fails += 1
                print(f"  [{i+1}/{n}] failed, recorded empty: {ex}", flush=True)
                chosen = []
        req = q["required_effects"]
        s = M.selection_scores(chosen, req, by_id, q.get("binding"))
        sel_hit += s["sel_hit"]
        sel_fc += s["sel_fullcov"]
        size += s["sel_size"]
        entries[q_key(q)]["methods"][selector.name] = {
            "ranked_ids": [],
            "chosen_ids": [c["skill_id"] for c in chosen],
        }
        print(f"  [{i+1}/{n}] {time.time()-qt:4.1f}s cov={'✓' if s['sel_fullcov'] else '✗'} "
              f"-> {[c['skill_id'] for c in chosen]}", flush=True)
        if flush is not None:
            flush()
    dt = time.time() - t0
    if fails:
        print(f"  (b0 failed on {fails}/{n} queries, recorded empty)")
    return {"sel_hit": sel_hit / n, "sel_fullcov": sel_fc / n, "sel_size": size / n}, dt


def load_or_init_selections(path, n, seed, pool_size, queries):
    """Reuse the file if it exists and (n, seed, pool_size) match (incrementally add methods); otherwise create a new one."""
    entries = {}
    if os.path.exists(path):
        try:
            data = json.load(open(path, encoding="utf-8"))
            m = data.get("meta", {})
            if m.get("n") == n and m.get("seed") == seed and m.get("pool_size") == pool_size:
                for e in data.get("queries", []):
                    entries[e["task_id"]] = e
        except Exception:
            entries = {}
    for q in queries:
        k = q_key(q)
        if k not in entries:
            e = q_meta(q)
            e["methods"] = {}
            entries[k] = e
        entries[k].setdefault("methods", {})
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of sampled queries (-1 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=["bm25"],
                    choices=["bm25", "dense", "b0", "b3", "m1", "m1f", "m1h", "m1h4", "m1h4nr", "b4",
                             # information-matched baselines: same per-skill fields as the router
                             "bm25c", "densec", "b0c", "b3c"])
    ap.add_argument("--out", default=os.path.join(EVAL_DATA, "selections.json"))
    ap.add_argument("--gt", default=None,
                    help="GT jsonl filename or absolute path (default gt_kitchen.jsonl; headline uses gt_kitchen_eval.jsonl)")
    ap.add_argument("--only-missing", action="store_true",
                    help="selection methods only recompute queries whose chosen is empty/missing (backfill timed-out empty selections)")
    args = ap.parse_args()

    pool = load_pool()
    n = None if args.n < 0 else args.n
    gt_path = args.gt
    if gt_path and not os.path.isabs(gt_path):
        gt_path = os.path.join(ROOT, "gt", "datasets", gt_path)
    queries = load_queries(n=n, seed=args.seed, gt_path=gt_path)
    by_id = {c["skill_id"]: c for c in pool}
    nn = len(queries)
    print(f"skill pool {len(pool)} entries; eval queries {nn}; methods={args.methods}\n")

    entries = load_or_init_selections(args.out, args.n, args.seed, len(pool), queries)

    def _flush():
        dump_selections(args.out, entries, queries, args.n, args.seed, len(pool))

    def _print_sel(r, res, dt):
        print(f"== {r.name} == (total {dt:.1f}s, {dt/nn:.1f}s/query)")
        print(f"  Sel-Hit={res['sel_hit']:.2f}  Sel-FullCov={res['sel_fullcov']:.2f}  "
              f"avg selected={res['sel_size']:.1f}\n")

    def _print_ret(r, res, bt, dt):
        print(f"== {r.name} == (build {bt:.1f}s, retrieval {dt:.1f}s, {dt/nn*1000:.1f}ms/query)")
        print("  Hit@K       :", "  ".join(f"@{k}={res[f'hit@{k}']:.2f}" for k in KS))
        print("  FullCov@K   :", "  ".join(f"@{k}={res[f'fc@{k}']:.2f}" for k in KS))
        print("  ID-Hit@K    :", "  ".join(f"@{k}={res[f'idhit@{k}']:.2f}" for k in KS), "\n")

    if "bm25" in args.methods:
        try:
            from retrievers import BM25Retriever
            t = time.time()
            r = BM25Retriever(pool)
            bt = time.time() - t
            res, dt = eval_retriever(r, queries, entries, by_id)
            _print_ret(r, res, bt, dt)
        except Exception as ex:
            print(f"!! bm25 skipped: {ex}\n")

    if "dense" in args.methods:
        try:
            from retrievers import DenseRetriever
            t = time.time()
            r = DenseRetriever(pool)
            bt = time.time() - t
            res, dt = eval_retriever(r, queries, entries, by_id)
            _print_ret(r, res, bt, dt)
        except Exception as ex:
            print(f"!! dense skipped: {ex}\n")

    if "b0" in args.methods:
        try:
            from llm_baselines import B0LLMFull
            r = B0LLMFull(pool)
            res, dt = eval_selector(r, queries, entries, by_id,
                                    only_missing=args.only_missing, flush=_flush)
            _print_sel(r, res, dt)
        except Exception as ex:
            print(f"!! b0 skipped: {ex}\n")

    if "b3" in args.methods:
        try:
            from pipelines import B3SkillFlow
            r = B3SkillFlow(pool)
            # retries=1: a single retry, otherwise transient API timeouts are recorded as empty
            # selections and the baseline is reported lower than it actually is.
            res, dt = eval_selector(r, queries, entries, by_id, retries=1,
                                    only_missing=args.only_missing, flush=_flush)
            _print_sel(r, res, dt)
        except Exception as ex:
            print(f"!! b3 skipped: {ex}\n")

    if "bm25c" in args.methods:
        try:
            from retrievers import BM25Retriever
            t = time.time()
            r = BM25Retriever(pool, full_card=True)
            bt = time.time() - t
            res, dt = eval_retriever(r, queries, entries, by_id)
            _print_ret(r, res, bt, dt)
        except Exception as ex:
            print(f"!! bm25c skipped: {ex}\n")

    if "densec" in args.methods:
        try:
            from retrievers import DenseRetriever
            t = time.time()
            r = DenseRetriever(pool, full_card=True)
            bt = time.time() - t
            res, dt = eval_retriever(r, queries, entries, by_id)
            _print_ret(r, res, bt, dt)
        except Exception as ex:
            print(f"!! densec skipped: {ex}\n")

    if "b0c" in args.methods:
        try:
            from llm_baselines import B0LLMFull
            r = B0LLMFull(pool, full_card=True)
            res, dt = eval_selector(r, queries, entries, by_id,
                                    only_missing=args.only_missing, flush=_flush)
            _print_sel(r, res, dt)
        except Exception as ex:
            print(f"!! b0c skipped: {ex}\n")

    if "b3c" in args.methods:
        try:
            from pipelines import B3SkillFlow
            r = B3SkillFlow(pool, full_card=True)
            res, dt = eval_selector(r, queries, entries, by_id, retries=1,
                                    only_missing=args.only_missing, flush=_flush)
            _print_sel(r, res, dt)
        except Exception as ex:
            print(f"!! b3c skipped: {ex}\n")

    if "m1" in args.methods:
        try:
            from domain_m1 import M1RuleDomain
            r = M1RuleDomain(pool)
            res, dt = eval_selector(r, queries, entries, by_id, retries=0)
            print(f"== {r.name} == (total {dt:.1f}s, {dt/nn*1000:.1f}ms/query)")
            print(f"  Sel-Hit={res['sel_hit']:.2f}  Sel-FullCov={res['sel_fullcov']:.2f}  "
                  f"avg selected={res['sel_size']:.1f}\n")
        except Exception as ex:
            print(f"!! m1 skipped: {ex}\n")

    if "m1f" in args.methods:
        try:
            from domain_m1 import M1RuleDomain
            from goal_parser_llm import LLMGoalParser
            r = M1RuleDomain(pool, goal_fn=LLMGoalParser(), name="M1-Formal")
            res, dt = eval_selector(r, queries, entries, by_id, retries=0,
                                    only_missing=args.only_missing)
            print(f"== {r.name} == (total {dt:.1f}s, {dt/nn*1000:.1f}ms/query)")
            print(f"  Sel-Hit={res['sel_hit']:.2f}  Sel-FullCov={res['sel_fullcov']:.2f}  "
                  f"avg selected={res['sel_size']:.1f}\n")
        except Exception as ex:
            print(f"!! m1f skipped: {ex}\n")

    if "m1h" in args.methods:
        try:
            from domain_m1 import M1RuleDomain
            from l0_head import L0HeadParser
            r = M1RuleDomain(pool, goal_fn=L0HeadParser(), name="M1-Head")
            res, dt = eval_selector(r, queries, entries, by_id, retries=0)
            print(f"== {r.name} == (total {dt:.1f}s, {dt/nn*1000:.1f}ms/query)")
            print(f"  Sel-Hit={res['sel_hit']:.2f}  Sel-FullCov={res['sel_fullcov']:.2f}  "
                  f"avg selected={res['sel_size']:.1f}\n")
        except Exception as ex:
            print(f"!! m1h skipped: {ex}\n")

    def _resolve_l4_ckpt():
        """L4_CKPT env var: use an absolute path as-is; a relative/bare filename resolves under router/models/."""
        ck = os.environ.get("L4_CKPT")
        if not ck:
            return None
        if not os.path.isabs(ck) and not os.path.exists(ck):
            cand = os.path.join(ROOT, "router", "models", os.path.basename(ck))
            if os.path.exists(cand):
                return cand
        return ck

    if "m1h4" in args.methods:
        try:
            from domain_m1 import M1RuleDomain
            from l0_head import L0HeadParser
            from l4_head import L4HeadScorer
            ck = _resolve_l4_ckpt()
            scorer = L4HeadScorer(ckpt=ck) if ck else L4HeadScorer()
            if ck:
                print(f"  [m1h4] L4_CKPT={ck}")
            r = M1RuleDomain(pool, goal_fn=L0HeadParser(), l4_scorer=scorer,
                             name="M1-Head+L4")
            res, dt = eval_selector(r, queries, entries, by_id, retries=0)
            print(f"== {r.name} == (total {dt:.1f}s, {dt/nn*1000:.1f}ms/query)")
            print(f"  Sel-Hit={res['sel_hit']:.2f}  Sel-FullCov={res['sel_fullcov']:.2f}  "
                  f"avg selected={res['sel_size']:.1f}\n")
        except Exception as ex:
            print(f"!! m1h4 skipped: {ex}\n")

    if "m1h4nr" in args.methods:
        try:
            from domain_m1 import M1RuleDomain
            from l0_head import L0HeadParser
            from l4_head import L4HeadScorer
            ck = _resolve_l4_ckpt()
            scorer = L4HeadScorer(ckpt=ck) if ck else L4HeadScorer()
            if ck:
                print(f"  [m1h4nr] L4_CKPT={ck}")
            r = M1RuleDomain(pool, goal_fn=L0HeadParser(), l4_scorer=scorer,
                             no_legality=True, no_repair=True, name="M1-Head+L4-noRule")
            res, dt = eval_selector(r, queries, entries, by_id, retries=0)
            print(f"== {r.name} == (total {dt:.1f}s, {dt/nn*1000:.1f}ms/query)")
            print(f"  Sel-Hit={res['sel_hit']:.2f}  Sel-FullCov={res['sel_fullcov']:.2f}  "
                  f"avg selected={res['sel_size']:.1f}\n")
        except Exception as ex:
            print(f"!! m1h4nr skipped: {ex}\n")

    if "b4" in args.methods:
        try:
            from lota_b4 import B4LoTaBench
            batch = int(os.environ.get("B4_BATCH", "16"))
            r = B4LoTaBench(pool, batch_size=batch, norm="mean")
            res, dt = eval_selector(r, queries, entries, by_id, retries=0,
                                    only_missing=args.only_missing)
            print(f"== {r.name} == (total {dt:.1f}s, {dt/nn:.1f}s/query)")
            print(f"  Sel-Hit={res['sel_hit']:.2f}  Sel-FullCov={res['sel_fullcov']:.2f}  "
                  f"avg selected={res['sel_size']:.1f}\n")
        except Exception as ex:
            print(f"!! b4 skipped: {ex}\n")

    os.makedirs(EVAL_DATA, exist_ok=True)
    out = dump_selections(args.out, entries, queries, args.n, args.seed, len(pool))
    methods_present = sorted({m for e in out["queries"] for m in e["methods"]})
    print(f"wrote {args.out}\n  {len(out['queries'])} queries; methods included: {methods_present}")


if __name__ == "__main__":
    main()
