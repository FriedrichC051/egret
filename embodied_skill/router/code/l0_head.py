"""L0-head：可训练多标签目标解析头（M1 正式版 L0 的"学习型"版本）。

结构：冻结 `bge-small-en-v1.5` 编码指令(384维) → 线性头(384->6) + sigmoid，BCE 多标签。
标签 6 维：isHot / isCold / isClean / isSliced / isFilledWithLiquid / at_receptacle。
训练标签 = train split(2607) 派生 required_effects；评测 = valid(211)，**不碰 eval GT 训练**。

公平红线：只吃 instruction；推理时 at_receptacle 与规则/LLM 一致——优先信 binding 容器。
关键观察点：头在 required_effects(=PDDL 目标) 上训练，会学到"物体/语境→效果"的**数据先验**，
可能预测出指令没写的效果（如 place spatula→isClean），从而在同分布 exact-match 上**超过
"只读指令"的 90.5% 上限**——这正是"学先验补全欠定指令"的双刃剑（见 WP4·L0 定案）。

用法：
  python router/code/l0_head.py            # 训练 + 在 211 报 exact-match，存权重
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
    """split ∈ {train, eval}。返回 [(instruction, binding, effset)]。"""
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
    """裸多标签 exact-match（不加 binding，用于 val 模型选择）。"""
    P = (1 / (1 + np.exp(-(X @ Wn + bn)))) >= thr
    return float((P == (Y >= 0.5)).all(axis=1).mean())


def fit_head(X, Y, max_epochs=3000, lr=0.05, val_frac=0.15, patience=8,
             eval_every=25, seed=0, verbose=False):
    """训练线性多标签头，用 train 内部 val 的 exact-match 早停，返回 best-on-val 的 (W,b,best_ep)。
    **只切 train 内部做 val，绝不碰 eval**，避免拿测试集挑轮次。"""
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
        print(f"  [fit_head] best val_EM={best[0]:.1%} @ep{best[3]}（train {len(ti)}/val {len(vi)}）")
    return best[1], best[2], best[3]


class L0HeadParser:
    """训练好的头，parse(instruction, binding) -> 效果集（与 parse_goal 同签名）。"""
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
    print(f"train {len(tr)} / eval {len(ev)}；backbone={os.path.basename(_resolve_bge())}")

    t0 = time.time()
    enc = SentenceTransformer(_resolve_bge())
    Xtr = _encode(enc, [r[0] for r in tr], np)
    Xev = _encode(enc, [r[0] for r in ev], np)
    Ytr = np.stack([_multihot(r[2], np) for r in tr])
    print(f"编码完成 {time.time()-t0:.1f}s；维度 {Xtr.shape[1]}")

    Wn, bn, best_ep = fit_head(Xtr, Ytr, max_epochs=args.max_epochs, lr=args.lr,
                               patience=args.patience, seed=args.seed, verbose=True)
    print(f"训练完成（早停 best@ep{best_ep}），用时 {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save({"W": Wn, "b": bn, "labels": LABELS}, CKPT)

    # ---- 在 211 评测：head vs 规则 vs GT（exact-match），并加 binding 修正 ----
    def head_pred(x, binding):
        logit = x @ Wn + bn
        prob = 1 / (1 + np.exp(-logit))
        g = {LABELS[i] for i in range(len(LABELS)) if prob[i] >= args.thresh}
        if binding.get("receptacle"):
            g.add("at_receptacle")
        return g

    head_ok = rule_ok = 0
    over = under = 0   # head 多预测 / 少预测（相对 GT）
    beat_ceiling = []  # head 对但规则错（含"猜出指令没写的效果"）
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
    print(f"\nexact-match（211 valid）：")
    print(f"  L0-head(训练)  = {head_ok}/{n} = {head_ok/n:.1%}")
    print(f"  规则 parse_goal = {rule_ok}/{n} = {rule_ok/n:.1%}   （LLM few-shot 参考 ~90%）")
    print(f"  head 多预测 {over} 题 / 少预测 {under} 题")
    print(f"\nhead 对而规则错的题（{len(beat_ceiling)} 道，含'靠先验猜出指令没写的效果'）：")
    for ins, r, g in beat_ceiling[:15]:
        print(f"  - {ins!r}\n    规则={r}  GT={g}")


if __name__ == "__main__":
    main()
