"""Phase 4 pipeline methods:
- B3: a slimmed-down SkillFlow (training-free three-stage) -- Dense retrieval top-K ->
      cross-encoder rerank top-M -> LLM picks the minimal skill set from the M candidates.
      Compared with B0 (which sees all 150 cards) the LLM only sees M; compared with B2
      (pure-vector top-1) it adds reranking + an LLM decision. This represents the
      "generic strong baseline".

Standalone self-check:
  python router/code/pipelines.py --n 3
"""
import os
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load_pool, load_queries, skill_sig, skill_doc  # noqa: E402
from retrievers import DenseRetriever                             # noqa: E402
from llm_baselines import llm_call, SYSTEM, parse_id_array        # noqa: E402

_LOCAL_RERANKER = os.path.join(HERE, os.pardir, "models", "bge-reranker-base")


def _resolve_reranker():
    env = os.environ.get("RERANKER_DIR")
    if env and os.path.isdir(env):
        return env
    if os.path.isdir(_LOCAL_RERANKER):
        return os.path.abspath(_LOCAL_RERANKER)
    return "BAAI/bge-reranker-base"


def _catalog(cards):
    lines = []
    for c in cards:
        sig = ",".join(sorted(skill_sig(c))) or "-"
        lines.append(f"- {c['skill_id']} | {c.get('name', '')} | effects: {sig}")
    return "\n".join(lines)


class B3SkillFlow:
    name = "B3-SkillFlow"

    def __init__(self, pool, k=20, m=8, retriever=None):
        self.pool = pool
        self.by_id = {c["skill_id"]: c for c in pool}
        self.k = k
        self.m = m
        self.retriever = retriever or DenseRetriever(pool)
        from sentence_transformers import CrossEncoder
        self.reranker = CrossEncoder(_resolve_reranker())

    def select(self, query):
        # (1) Dense retrieval top-K
        cand = self.retriever.rank(query)[:self.k]
        # (2) cross-encoder rerank -> top-M
        pairs = [(query, skill_doc(c)) for c in cand]
        scores = self.reranker.predict(pairs)
        order = sorted(range(len(cand)), key=lambda i: -scores[i])
        short = [cand[i] for i in order[:self.m]]
        # (3) LLM picks the minimal set from the M candidates
        by_id = {c["skill_id"]: c for c in short}
        user = (f"Skill candidates ({len(short)} total, sorted by relevance):\n{_catalog(short)}\n\n"
                f"Task: {query}\n\nFrom the candidates above, select the minimal skill set that "
                f"completes the task (a JSON array of skill_id, choosing only from the candidates).")
        text = llm_call(SYSTEM, user, timeout=120.0)  # aligned with b0: 60s would mass-timeout during API peaks
        ids = parse_id_array(text, set(by_id))
        return [by_id[i] for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--m", type=int, default=8)
    args = ap.parse_args()
    pool = load_pool()
    queries = load_queries(n=args.n, seed=0)
    b3 = B3SkillFlow(pool, k=args.k, m=args.m)
    print(f"Skill pool: {len(pool)} cards; B3 retrieval top{args.k} -> rerank top{args.m} -> LLM select\n")
    for q in queries:
        t = time.time()
        chosen = b3.select(q["instruction"])
        dt = time.time() - t
        req = set(q["required_effects"])
        union = set()
        for c in chosen:
            union |= skill_sig(c)
        print(f"[{dt:5.1f}s] req={sorted(req)} -> {[c['skill_id'] for c in chosen]} "
              f"covered={'OK' if union >= req else 'X'}")


if __name__ == "__main__":
    main()
