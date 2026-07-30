"""L0-head: trainable multi-label goal-parsing head (the "learned" version of the production M1 L0).

Architecture: freeze `bge-small-en-v1.5` to encode the instruction (384-d) -> linear head
(384->6) + sigmoid, multi-label BCE.
6 label dimensions: isHot / isCold / isClean / isSliced / isFilledWithLiquid / at_receptacle.
Training labels = required_effects derived from the train split (2607); evaluation = valid (211),
**never trains on eval GT**.

Fairness red line: consumes only the instruction; at inference at_receptacle matches the
rule/LLM -- prefers to trust the binding receptacle.
Key observation: the head, trained on required_effects (= PDDL goals), learns a **data prior**
of "object/context -> effect", and may predict effects the instruction never wrote (e.g. place
spatula -> isClean), thereby **exceeding the 90.5% ceiling of "instruction-only"** on in-distribution
exact-match -- exactly the double-edged sword of "learning a prior to complete underspecified
instructions" (see WP4 - L0 decision).

Usage:
  python router/code/l0_head.py            # train + report exact-match on 211, save weights
  python router/code/l0_head.py --epochs 300
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import parse_goal  # noqa: E402
from retrievers import _resolve_bge  # noqa: E402

ROOT = os.path.join(HERE, os.pardir, os.pardir)
GT_DIR = os.path.join(ROOT, "gt", "datasets")
LABELS = ["isHot", "isCold", "isClean", "isSliced", "isFilledWithLiquid", "at_receptacle"]
L2I = {l: i for i, l in enumerate(LABELS)}
CKPT = os.path.join(HERE, os.pardir, "models", "l0_head.pt")


def _load(split):
    """split in {train, eval}. Returns [(instruction, binding, effset)]."""
    path = os.path.join(GT_DIR, f"gt_kitchen_{split}.jsonl")
    rows = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        rows.append((d["instruction"], d.get("binding") or {}, set(d["required_effects"])))
    return rows


def _multihot(effset, np):
    v = np.zeros(len(LABELS), dtype="float32")
    for e in effset:
        if e in L2I:
            v[L2I[e]] = 1.0
    return v


def _encode(model, texts, np):
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                        batch_size=64, show_progress_bar=False).astype("float32")


def _raw_em(X, Y, Wn, bn, np, thr=0.5):
    """Raw multi-label exact-match (no binding, used for val model selection)."""
    P = (1 / (1 + np.exp(-(X @ Wn + bn)))) >= thr
    return float((P == (Y >= 0.5)).all(axis=1).mean())


def fit_head(X, Y, max_epochs=3000, lr=0.05, val_frac=0.15, patience=8,
             eval_every=25, seed=0, verbose=False):
    """Train the linear multi-label head, early-stopping on the exact-match of an internal
    train val split, returning the best-on-val (W, b, best_ep).
    **Only splits val from within train, never touches eval**, to avoid picking epochs on the test set."""
    import numpy as np
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(X)
    perm = np.random.permutation(n)
    nval = max(1, int(n * val_frac))
    vi, ti = perm[:nval], perm[nval:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(X[ti], device=dev); Yt = torch.tensor(Y[ti], device=dev)
    D, C = X.shape[1], Y.shape[1]
    W = torch.zeros(D, C, device=dev, requires_grad=True)
    b = torch.zeros(C, device=dev, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    best = (-1.0, None, None, 0); bad = 0
    for ep in range(1, max_epochs + 1):
        opt.zero_grad(); loss = lossf(Xt @ W + b, Yt); loss.backward(); opt.step()
        if ep % eval_every == 0:
            Wn = W.detach().cpu().numpy(); bn = b.detach().cpu().numpy()
            em = _raw_em(X[vi], Y[vi], Wn, bn, np)
            if em > best[0]:
                best = (em, Wn.copy(), bn.copy(), ep); bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if verbose:
        print(f"  [fit_head] best val_EM={best[0]:.1%} @ep{best[3]} (train {len(ti)}/val {len(vi)})")
    return best[1], best[2], best[3]


class L0HeadParser:
    """The trained head; parse(instruction, binding) -> effect set (same signature as parse_goal)."""
    def __init__(self, ckpt=CKPT, thresh=0.5):
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch = torch
        self.thresh = thresh
        self.enc = SentenceTransformer(_resolve_bge())
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.W = obj["W"]; self.b = obj["b"]

    def parse(self, instruction, binding=None):
        import numpy as np
        x = self.enc.encode([instruction], normalize_embeddings=True,
                            convert_to_numpy=True).astype("float32")[0]
        logit = x @ self.W + self.b
        prob = 1 / (1 + np.exp(-logit))
        goal = {LABELS[i] for i in range(len(LABELS)) if prob[i] >= self.thresh}
        binding = binding or {}
        if binding.get("receptacle"):
            goal.add("at_receptacle")
        return goal

    def __call__(self, instruction, binding=None):
        return self.parse(instruction, binding)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    tr = _load("train"); ev = _load("eval")
    print(f"train {len(tr)} / eval {len(ev)}; backbone={os.path.basename(_resolve_bge())}")

    t0 = time.time()
    enc = SentenceTransformer(_resolve_bge())
    Xtr = _encode(enc, [r[0] for r in tr], np)
    Xev = _encode(enc, [r[0] for r in ev], np)
    Ytr = np.stack([_multihot(r[2], np) for r in tr])
    print(f"encoding done {time.time()-t0:.1f}s; dim {Xtr.shape[1]}")

    Wn, bn, best_ep = fit_head(Xtr, Ytr, max_epochs=args.max_epochs, lr=args.lr,
                               patience=args.patience, seed=args.seed, verbose=True)
    print(f"training done (early stop best@ep{best_ep}), time {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save({"W": Wn, "b": bn, "labels": LABELS}, CKPT)

    # ---- Evaluate on 211: head vs rule vs GT (exact-match), with binding correction ----
    def head_pred(x, binding):
        logit = x @ Wn + bn
        prob = 1 / (1 + np.exp(-logit))
        g = {LABELS[i] for i in range(len(LABELS)) if prob[i] >= args.thresh}
        if binding.get("receptacle"):
            g.add("at_receptacle")
        return g

    head_ok = rule_ok = 0
    over = under = 0   # head over-predicts / under-predicts (relative to GT)
    beat_ceiling = []  # head correct but rule wrong (incl. "guessing effects the instruction didn't write")
    for i, (ins, binding, req) in enumerate(ev):
        g_head = head_pred(Xev[i], binding)
        g_rule = parse_goal(ins, binding)
        head_ok += int(g_head == req)
        rule_ok += int(g_rule == req)
        if g_head == req and g_rule != req:
            beat_ceiling.append((ins[:65], sorted(g_rule), sorted(req)))
        if g_head - req:
            over += 1
        if req - g_head:
            under += 1
    n = len(ev)
    print(f"\nexact-match (211 valid):")
    print(f"  L0-head(trained) = {head_ok}/{n} = {head_ok/n:.1%}")
    print(f"  rule parse_goal  = {rule_ok}/{n} = {rule_ok/n:.1%}   (LLM few-shot reference ~90%)")
    print(f"  head over-predicts {over} items / under-predicts {under} items")
    print(f"\nItems where head is right but rule is wrong ({len(beat_ceiling)}, incl. 'guessing via prior the effects the instruction didn't write'):")
    for ins, r, g in beat_ceiling[:15]:
        print(f"  - {ins!r}\n    rule={r}  GT={g}")


if __name__ == "__main__":
    main()
