"""L4 combination selector, discriminative trainable version (WP4 ablation: "can a trained L4 beat the rule set-cover").

The main system's L4 = rule-based greedy set-cover (`domain_m1._cover`, provably minimal-coverage
optimal when the goal + effects are correct).
This module provides a **learned counterpart**: for each L2 legal candidate, read [goal g,
candidate true effect signature, coverage/extra/missing counts, retrieval rank, candidate-card
text embedding] and discriminatively assign a "retention score".

How it connects (see domain_m1): **always coverage-first** (structural lower bound unchanged),
the learned score only breaks ties among candidates with "equal new coverage" -- exactly where
lookalikes / near-synonym variants compete and the rule can only blindly pick by rank. If the
learned score brings no gain on a clean small pool, that is WP4's expected "training ~= no
training" finding; if it disambiguates variants correctly, that is a positive gain.

Fairness red line: uses only candidate-card text (name/id/effects/desc) + parsed goal + rank;
**does not read** is_distractor/why_wrong/variant_type; labels come only from the **train split**
gold-standard set, never touching eval GT.

Usage:
  python router/code/l4_head.py                 # train with L0-head goals + save weights
  python router/code/l4_head.py --l0 rule        # train with rule goals
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load_pool, load_queries, skill_doc, parse_goal  # noqa: E402
from retrievers import _resolve_bge  # noqa: E402
from l0_head import LABELS, L2I  # noqa: E402

CKPT = os.path.join(HERE, os.pardir, "models", "l4_head.pt")
FEAT_STRUCT = 2 * len(LABELS) + 3 + 1   # goal_mh + eff_mh + [overlap,extra,missing] + rank_norm


def _mh(effset, np):
    v = np.zeros(len(LABELS), dtype="float32")
    for e in effset:
        if e in L2I:
            v[L2I[e]] = 1.0
    return v


def _struct_feat(goal, eff, rank, k, np):
    """The candidate's structural features (concatenated with the embedding)."""
    g = set(goal); e = set(eff)
    gm, em = _mh(g, np), _mh(e, np)
    overlap = len(e & g); extra = len(e - g); missing = len(g - e)
    scal = np.array([overlap, extra, missing, rank / max(1, k)], dtype="float32")
    return np.concatenate([gm, em, scal])


def _resolve_l0(kind):
    if kind == "rule":
        return parse_goal
    if kind == "head":
        from l0_head import L0HeadParser
        return L0HeadParser()
    if kind == "llm":
        from goal_parser_llm import LLMGoalParser
        return LLMGoalParser()
    raise ValueError(kind)


def _build_examples(split, goal_fn, m1, emb_by_id, np, gt_path=None):
    """Iterate over the split's queries and produce (X, y): each row = one L2 legal candidate.
    gt_path can override the default gt_kitchen_{split}.jsonl (WP5 axis-3 trains on the train_baseA subset)."""
    if gt_path is None:
        gt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
            "gt", "datasets", f"gt_kitchen_{split}.jsonl")
    elif not os.path.isabs(gt_path):
        gt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
            "gt", "datasets", gt_path)
    rows = load_queries(n=None, gt_path=gt_path)
    # Labels use the **effect scheme** (consistent with M1's effect-coverage correctness), not gold-standard ids:
    #   composite skills in the pool are synthesized, with ids differing from the canonical gold
    #   standard but effect-equivalent -- labeling by id would misalign.
    #   Positive = "clean contributor": true effects subset-of goal (no extra/wrong transforms) and
    #     intersecting the goal (can cover).
    #   -> a lookalike (with extra/wrong effects, e.g. heat->cool) is automatically negative; a
    #      missing variant can still be positive but covers incompletely, resolved by coverage-first
    #      greedy + L5 fallback, not by L4's discrimination.
    X, y = [], []
    for q in rows:
        instr, binding = q["instruction"], q.get("binding")
        goal = set(goal_fn(instr, binding))
        if not goal:
            continue
        legal = m1.legal_candidates(instr, binding)
        for rank, card, eff in legal:
            eff = set(eff)
            label = 1.0 if (eff and eff <= goal) else 0.0
            sf = _struct_feat(goal, eff, rank, m1.k, np)
            X.append(np.concatenate([sf, emb_by_id[card["skill_id"]]]))
            y.append(label)
    return np.stack(X), np.array(y, dtype="float32")


def _fit(Xtr, ytr, np, torch, max_epochs=3000, lr=0.02, hidden=64,
         val_frac=0.15, patience=10, eval_every=25, seed=0, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(Xtr); perm = np.random.permutation(n); nval = max(1, int(n * val_frac))
    vi, ti = perm[:nval], perm[nval:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(Xtr[ti], device=dev); yt = torch.tensor(ytr[ti], device=dev)
    Xv = torch.tensor(Xtr[vi], device=dev); yv = torch.tensor(ytr[vi], device=dev)
    D = Xtr.shape[1]
    pos = float(ytr[ti].sum()); neg = len(ti) - pos
    pw = torch.tensor([neg / max(1.0, pos)], device=dev)
    net = torch.nn.Sequential(torch.nn.Linear(D, hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    best = (1e9, None, 0); bad = 0
    for ep in range(1, max_epochs + 1):
        net.train(); opt.zero_grad()
        loss = lossf(net(Xt).squeeze(-1), yt); loss.backward(); opt.step()
        if ep % eval_every == 0:
            net.eval()
            with torch.no_grad():
                vl = lossf(net(Xv).squeeze(-1), yv).item()
            if vl < best[0] - 1e-4:
                sd = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
                best = (vl, sd, ep); bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if verbose:
        print(f"  [L4 fit] best val_BCE={best[0]:.4f} @ep{best[2]} (train {len(ti)}/val {len(vi)}, "
              f"pos={int(pos)}/{len(ti)})")
    return best[1], best[2]


class L4HeadScorer:
    """Inference: given (goal, legal candidates [(rank, card, eff)]) -> {skill_id: retention score} (numpy forward)."""
    def __init__(self, ckpt=CKPT, k=None):
        import numpy as np
        self.np = np
        obj = __import__("torch").load(ckpt, map_location="cpu", weights_only=False)
        self.sd = obj["state_dict"]; self.emb_by_id = dict(obj["emb_by_id"])
        self.k = k or obj.get("k", 20)
        self._enc = None  # lazy bge: encode on demand when a confusion-tier pool has a candidate outside the cache

    def _emb(self, card):
        sid = card["skill_id"]
        e = self.emb_by_id.get(sid)
        if e is not None:
            return e
        if self._enc is None:
            from sentence_transformers import SentenceTransformer
            self._enc = SentenceTransformer(_resolve_bge())
        e = self._enc.encode([skill_doc(card)], normalize_embeddings=True,
                             convert_to_numpy=True)[0].astype("float32")
        self.emb_by_id[sid] = e
        return e

    def _forward(self, X):
        np = self.np
        W0 = self.sd["0.weight"]; b0 = self.sd["0.bias"]
        W2 = self.sd["2.weight"]; b2 = self.sd["2.bias"]
        h = np.maximum(0.0, X @ W0.T + b0)
        return (h @ W2.T + b2).squeeze(-1)

    def score(self, goal, legal):
        np = self.np
        feats, sids = [], []
        for rank, card, eff in legal:
            emb = self._emb(card)
            sf = _struct_feat(goal, eff, rank, self.k, np)
            feats.append(np.concatenate([sf, emb])); sids.append(card["skill_id"])
        if not feats:
            return {}
        logits = self._forward(np.stack(feats).astype("float32"))
        return {sid: float(v) for sid, v in zip(sids, np.atleast_1d(logits))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l0", choices=["rule", "head", "llm"], default="head",
                    help="which L0 provides the goal g in training/eval features (paired with deployment, default head)")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--max-epochs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="weight output path (default overwrites l4_head.pt; relative paths land under models/)")
    ap.add_argument("--train-gt", default=None,
                    help="training GT filename (default gt_kitchen_train.jsonl; axis-3 uses gt_kitchen_train_baseA.jsonl)")
    args = ap.parse_args()

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from domain_m1 import M1RuleDomain

    t0 = time.time()
    pool = load_pool()
    enc = SentenceTransformer(_resolve_bge())
    embs = enc.encode([skill_doc(c) for c in pool], normalize_embeddings=True,
                      convert_to_numpy=True, batch_size=64).astype("float32")
    emb_by_id = {c["skill_id"]: embs[i] for i, c in enumerate(pool)}
    goal_fn = _resolve_l0(args.l0)
    m1 = M1RuleDomain(pool, k=args.k, goal_fn=goal_fn)
    print(f"pool {len(pool)}; L0={args.l0}; encoding + building candidates...")

    Xtr, ytr = _build_examples("train", goal_fn, m1, emb_by_id, np, gt_path=args.train_gt)
    print(f"train examples {len(ytr)} (positive {int(ytr.sum())}); dim {Xtr.shape[1]}; {time.time()-t0:.1f}s")

    sd, best_ep = _fit(Xtr, ytr, np, torch, max_epochs=args.max_epochs,
                       seed=args.seed, verbose=True)
    out = args.out or CKPT
    if not os.path.isabs(out):
        out = os.path.join(os.path.dirname(CKPT), out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"state_dict": sd, "emb_by_id": emb_by_id, "l0": args.l0, "k": m1.k}, out)
    print(f"saved {out} (best@ep{best_ep}), total time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
