"""L4 组合选择器·判别式可训练版（WP4 消融："训练的 L4 能不能比规则 set-cover 更好"）。

主系统 L4 = 规则贪心 set-cover（`domain_m1._cover`，目标+效果正确时可证最小覆盖最优）。
本模块给它一个**学习型对照**：对每个 L2 合法候选，读 [目标 g、候选真实效果签名、
覆盖/多余/缺失计数、检索名次、候选卡文本嵌入]，判别式打一个"保留分"。

接法（见 domain_m1）：**始终覆盖优先**（结构下限不变），学习分只在"同等新覆盖"的候选
之间做取舍——正是 lookalike / 近义变体竞争、规则只能靠名次盲选的地方。若干净小池里学习
分带不来增量，就是 WP4 预期的"训练≈不训练"发现；若能把变体消歧做对，则是正向增量。

公平红线：只用候选卡文本(name/id/effects/desc) + 解析目标 + 名次；**不读**
is_distractor/why_wrong/variant_type，标签仅来自 **train split** 金标准集合，不碰 eval GT。

用法：
  python router/code/l4_head.py                 # 用 L0-head 目标训练 + 存权重
  python router/code/l4_head.py --l0 rule        # 用规则目标训练
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
    """候选的结构特征（与嵌入拼接）。"""
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
    """遍历 split query，产出 (X, y)：每条 = 一个 L2 合法候选。
    gt_path 可覆盖默认 gt_kitchen_{split}.jsonl（WP5 轴3 用 train_baseA 子集训练）。"""
    if gt_path is None:
        gt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
            "gt", "datasets", f"gt_kitchen_{split}.jsonl")
    elif not os.path.isabs(gt_path):
        gt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
            "gt", "datasets", gt_path)
    rows = load_queries(n=None, gt_path=gt_path)
    # 标签用**效果制**（与 M1 效果覆盖判正一致），不用金标准 id：
    #   池里复合技能是合成的、id 与规范金标准不同但效果等价——按 id 打标签会错位。
    #   正 = "干净贡献者"：真实效果 ⊆ 目标（不做多余/错误变换）且与目标有交集（能覆盖）。
    #   → lookalike（带多余错效果，如 heat→cool）自动为负；missing 变体仍可为正但覆盖不全，
    #     靠覆盖优先的贪心+L5 兜底解决，不归 L4 判别。
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
        print(f"  [L4 fit] best val_BCE={best[0]:.4f} @ep{best[2]}（train {len(ti)}/val {len(vi)}，"
              f"pos={int(pos)}/{len(ti)}）")
    return best[1], best[2]


class L4HeadScorer:
    """推理：给定 (goal, 合法候选[(rank,card,eff)]) → {skill_id: 保留分}（numpy 前向）。"""
    def __init__(self, ckpt=CKPT, k=None):
        import numpy as np
        self.np = np
        obj = __import__("torch").load(ckpt, map_location="cpu", weights_only=False)
        self.sd = obj["state_dict"]; self.emb_by_id = dict(obj["emb_by_id"])
        self.k = k or obj.get("k", 20)
        self._enc = None  # 惰性 bge：混淆分档池里遇到缓存外的候选时按需编码

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
                    help="训练/评测特征里目标 g 用哪个 L0（与部署配对，默认 head）")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--max-epochs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="权重输出路径（不给则覆盖默认 l4_head.pt；相对路径落在 models/ 下）")
    ap.add_argument("--train-gt", default=None,
                    help="训练用 GT 文件名（默认 gt_kitchen_train.jsonl；轴3 用 gt_kitchen_train_baseA.jsonl）")
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
    print(f"池 {len(pool)}；L0={args.l0}；编码+构候选…")

    Xtr, ytr = _build_examples("train", goal_fn, m1, emb_by_id, np, gt_path=args.train_gt)
    print(f"训练样本 {len(ytr)}（正 {int(ytr.sum())}）；维度 {Xtr.shape[1]}；{time.time()-t0:.1f}s")

    sd, best_ep = _fit(Xtr, ytr, np, torch, max_epochs=args.max_epochs,
                       seed=args.seed, verbose=True)
    out = args.out or CKPT
    if not os.path.isabs(out):
        out = os.path.join(os.path.dirname(CKPT), out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"state_dict": sd, "emb_by_id": emb_by_id, "l0": args.l0, "k": m1.k}, out)
    print(f"已存 {out}（best@ep{best_ep}），总用时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
