"""Phase 4 流水线方法：
- B3：SkillFlow 缩水版（免训练三阶段）——Dense 检索 top-K → 交叉编码器重排 top-M
      → LLM 从 M 条候选里选最小技能集合。相比 B0（看全 150 条）只让 LLM 看 M 条，
      相比 B2（纯向量 top-1）多了重排 + LLM 决策。这是"通用强 baseline"的代表。

单跑自检：
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
        # ① Dense 检索 top-K
        cand = self.retriever.rank(query)[:self.k]
        # ② 交叉编码器重排 → top-M
        pairs = [(query, skill_doc(c)) for c in cand]
        scores = self.reranker.predict(pairs)
        order = sorted(range(len(cand)), key=lambda i: -scores[i])
        short = [cand[i] for i in order[:self.m]]
        # ③ LLM 从 M 条候选里选最小集合
        by_id = {c["skill_id"]: c for c in short}
        user = (f"技能候选（共 {len(short)} 条，已按相关度排序）：\n{_catalog(short)}\n\n"
                f"任务：{query}\n\n请从上面候选中选出能完成该任务的最小技能集合"
                f"（skill_id 的 JSON 数组，只能选候选里的）。")
        text = llm_call(SYSTEM, user, timeout=120.0)  # 对齐 b0：60s 在 API 高峰会大面积超时
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
    print(f"技能池 {len(pool)} 条；B3 检索top{args.k}→重排top{args.m}→LLM选\n")
    for q in queries:
        t = time.time()
        chosen = b3.select(q["instruction"])
        dt = time.time() - t
        req = set(q["required_effects"])
        union = set()
        for c in chosen:
            union |= skill_sig(c)
        print(f"[{dt:5.1f}s] req={sorted(req)} -> {[c['skill_id'] for c in chosen]} "
              f"覆盖={'✓' if union >= req else '✗'}")


if __name__ == "__main__":
    main()
