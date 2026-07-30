"""Phase 4 检索器：B1 BM25（纯词面）、B2 Dense（稠密向量，稍后接入）。

统一接口：
  retriever.rank(query_text) -> 按分数降序的 card 列表（全池排序）。
"""
import os
from common import skill_doc, tokenize

_HERE = os.path.dirname(os.path.abspath(__file__))                 # router/code/
_LOCAL_BGE = os.path.join(_HERE, os.pardir, "models", "bge-small-en-v1.5")


def _resolve_bge():
    """优先用本地已下载目录，避免联网/缓存符号链接问题；否则回退 hub id。"""
    env = os.environ.get("BGE_MODEL_DIR")
    if env and os.path.isdir(env):
        return env
    if os.path.isdir(_LOCAL_BGE):
        return os.path.abspath(_LOCAL_BGE)
    return "BAAI/bge-small-en-v1.5"


class BM25Retriever:
    name = "B1-BM25"

    def __init__(self, pool):
        from rank_bm25 import BM25Okapi
        self.pool = pool
        self._docs = [tokenize(skill_doc(c)) for c in pool]
        self.bm25 = BM25Okapi(self._docs)

    def rank(self, query):
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.pool)), key=lambda i: -scores[i])
        return [self.pool[i] for i in order]


class DenseRetriever:
    """B2 稠密检索：BGE 英文向量模型（技能卡/query 均已英文，无需多语模型）。

    bge-small-en-v1.5 约定：passage 直接编码，query 前加检索指令前缀。"""
    name = "B2-Dense"
    Q_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, pool, model_name=None):
        model_name = model_name or _resolve_bge()
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self.np = np
        self.pool = pool
        self.model = SentenceTransformer(model_name)
        docs = [skill_doc(c) for c in pool]
        self.doc_emb = self.model.encode(docs, normalize_embeddings=True,
                                         convert_to_numpy=True)

    def rank(self, query):
        q = self.model.encode([self.Q_PREFIX + query], normalize_embeddings=True,
                              convert_to_numpy=True)[0]
        scores = self.doc_emb @ q
        order = self.np.argsort(-scores)
        return [self.pool[i] for i in order]
