"""Phase 4 retrievers: B1 BM25 (pure lexical), B2 Dense (dense vectors, wired in later).

Unified interface:
  retriever.rank(query_text) -> list of cards sorted by descending score (full-pool ranking).
"""
import os
from common import skill_doc, skill_doc_full, tokenize


def _docs_of(pool, full_card):
    """Retrieval documents: skill_doc by default; with full_card=True the symbolic
    preconditions and the atomic-expanded sub-skill sequence are appended, so that the
    baseline reads the same fields as the router (see common.skill_struct_text)."""
    if not full_card:
        return [skill_doc(c) for c in pool]
    by_id = {c["skill_id"]: c for c in pool}
    return [skill_doc_full(c, by_id) for c in pool]


_HERE = os.path.dirname(os.path.abspath(__file__))                 # router/code/
_LOCAL_BGE = os.path.join(_HERE, os.pardir, "models", "bge-small-en-v1.5")


def _resolve_bge():
    """Prefer a locally downloaded directory to avoid network/cache-symlink issues;
    otherwise fall back to the hub id."""
    env = os.environ.get("BGE_MODEL_DIR")
    if env and os.path.isdir(env):
        return env
    if os.path.isdir(_LOCAL_BGE):
        return os.path.abspath(_LOCAL_BGE)
    return "BAAI/bge-small-en-v1.5"


class BM25Retriever:
    name = "B1-BM25"

    def __init__(self, pool, full_card=False):
        from rank_bm25 import BM25Okapi
        self.pool = pool
        self.full_card = full_card
        if full_card:
            self.name = "B1-BM25-Card"
        self._docs = [tokenize(d) for d in _docs_of(pool, full_card)]
        self.bm25 = BM25Okapi(self._docs)

    def rank(self, query):
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.pool)), key=lambda i: -scores[i])
        return [self.pool[i] for i in order]


class DenseRetriever:
    """B2 dense retrieval: BGE English vector model (skill cards/queries are already English,
    so no multilingual model is needed).

    bge-small-en-v1.5 convention: passages are encoded directly; queries are prefixed with a
    retrieval-instruction prefix."""
    name = "B2-Dense"
    Q_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, pool, model_name=None, full_card=False):
        model_name = model_name or _resolve_bge()
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self.np = np
        self.pool = pool
        self.full_card = full_card
        if full_card:
            self.name = "B2-Dense-Card"
        self.model = SentenceTransformer(model_name)
        docs = _docs_of(pool, full_card)
        self.doc_emb = self.model.encode(docs, normalize_embeddings=True,
                                         convert_to_numpy=True)

    def rank(self, query):
        q = self.model.encode([self.Q_PREFIX + query], normalize_embeddings=True,
                              convert_to_numpy=True)[0]
        scores = self.doc_emb @ q
        order = self.np.argsort(-scores)
        return [self.pool[i] for i in order]
