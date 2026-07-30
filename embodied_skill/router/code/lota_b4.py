"""B4 = LoTa-Bench 式 LLM 逐步受限解码选技能（ICLR'24, LLMTaskPlanning）。

与 B0 的本质区别：**不把技能库塞进 prompt**。而是——
  前缀 = few-shot 示例 + 任务 + 已选历史（短）；
  每步对**技能池里每条技能**（其自然语言 name）算"作为续写的 token logprob"，
  取 argmax（受限解码：只能在池内技能里选），append 到历史，自回归到 "done"。
=> 本地开源 completion 模型即可（要 token logits），零 API 费用。chat-LLM 那条路由 B0 覆盖。

模型：本地 HF causal LM（fp16, cuda），默认 Qwen2.5-1.5B，实验用 Qwen2.5-3B（B4_MODEL 覆盖）。
接口：B4LoTaBench(pool).select(query, binding=None) -> [card]（与 M1/B0/B3 一致）。

对 LoTa-Bench 原文的三处偏离（见 stage2_execution_plan.md WP3 脚注）：
  1) 模型：原文用 2023 年 GPT-Neo/GPT-J/OPT/LLaMA-1（+GPT API），无 Qwen；我们复现方法、
     换现代开源基座 Qwen2.5（本地 3B 的 SR≈29% 落在原文 GPT-3/4 区间内，未削弱基线）。
  2) 受限解码：未用原文 `guidance` 库，改用等价的逐候选整串 logprob + argmax（更稳更可复现）。
  3) 打分归一：原文 sum（等长模板短语）→ 我们 mean（复合技能 name 变长，sum 会退化成永选最短原子）。
"""
import os
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load_pool, load_queries, skill_sig  # noqa: E402

DEFAULT_MODEL = os.environ.get("B4_MODEL", "Qwen/Qwen2.5-1.5B")
DONE = "done"

FEWSHOT = (
    "You are a household robot. For each task, list the skills to accomplish it, "
    "one per line, then write 'done'.\n\n"
    "Task: put the mug on the coffee table.\n"
    "Plan:\n1. Pick up object and place into receptacle\n2. done\n\n"
    "Task: heat the potato and put it on the countertop.\n"
    "Plan:\n1. Heat object with microwave then place into receptacle\n2. done\n\n"
    "Task: chill the apple and put it on the table.\n"
    "Plan:\n1. Cool object with fridge then place into receptacle\n2. done\n\n"
    "Task: rinse the lettuce and put it in the fridge.\n"
    "Plan:\n1. Clean object in sink then place into receptacle\n2. done\n\n"
)


def _name(card):
    return (card.get("name") or card.get("skill_id") or "").strip()


_MODEL_CACHE = {}  # model_id -> (tokenizer, model)，多档池复用同一份权重（省显存/加载）


def _load_model(model_id):
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16).to("cuda").eval()
    print(f"[B4] model on {next(model.parameters()).device}, "
          f"dtype {next(model.parameters()).dtype}", flush=True)
    _MODEL_CACHE[model_id] = (tok, model)
    return tok, model


class B4LoTaBench:
    name = "B4-LoTa"
    wants_binding = False

    def __init__(self, pool, model_id=DEFAULT_MODEL, max_steps=3, batch_size=8,
                 norm="sum"):
        import torch
        self.torch = torch
        self.pool = pool
        self.by_id = {c["skill_id"]: c for c in pool}
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.norm = norm  # "sum"(LoTa 原法) 或 "mean"(长度归一)
        self.tok, self.model = _load_model(model_id)
        # 候选文本：每条技能的 name（+ "done"）；预 tokenize 候选 token（不加特殊符）
        self.cand_ids = [c["skill_id"] for c in pool]
        self.cand_txt = [_name(c) for c in pool] + [DONE]
        self.cand_tok = [self.tok.encode(" " + t, add_special_tokens=False)
                         for t in self.cand_txt]

    def _prefix(self, query, history):
        lines = "".join(f"{i+1}. {h}\n" for i, h in enumerate(history))
        nxt = len(history) + 1
        return f"{FEWSHOT}Task: {query}\nPlan:\n{lines}{nxt}."

    @staticmethod
    def _pad(seqs, pad_id):
        m = max(len(s) for s in seqs)
        return [s + [pad_id] * (m - len(s)) for s in seqs], [len(s) for s in seqs]

    def _score_all(self, prefix):
        """批量前缀 KV 复用：前缀按 batch 展开前向一次得 cache，
        每批候选右拼 padding 前向一次，算完 crop 回前缀长度复用（快且省显存）。"""
        torch = self.torch
        B = self.batch_size
        pre = self.tok.encode(prefix, add_special_tokens=True)
        plen = len(pre)
        pad_id = self.tok.pad_token_id
        n = len(self.cand_tok)
        scores = [0.0] * n
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([pre], device="cuda"),
                             use_cache=True)                      # 前缀 batch=1（省显存）
            cache = out.past_key_values
            cache.batch_repeat_interleave(B)                      # 复制到 batch B 复用
            base_lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)  # 第一 token
            pre_attn = torch.ones((B, plen), device="cuda", dtype=torch.long)
            for b in range(0, n, B):
                chunk = self.cand_tok[b:b + B]
                bs = len(chunk)
                first = [base_lp[ct[0]].item() if ct else -1e9 for ct in chunk]
                # 输入 = ct[:-1]（放到 prefix 之后），预测目标 = ct[1:]
                bodies = [ct[:-1] for ct in chunk]
                tgts = [ct[1:] for ct in chunk]
                lens = [len(t) for t in tgts]
                maxL = max(lens, default=0)
                if maxL == 0:                              # 全是单 token 候选
                    for i, ct in enumerate(chunk):
                        s = first[i]
                        scores[b + i] = s / len(ct) if (self.norm == "mean" and ct) else s
                    continue
                inp = torch.full((B, maxL), pad_id, device="cuda", dtype=torch.long)
                tgt = torch.zeros((B, maxL), device="cuda", dtype=torch.long)
                msk = torch.zeros((B, maxL), device="cuda")
                for i in range(bs):
                    L = lens[i]
                    if L:
                        inp[i, :L] = torch.tensor(bodies[i], device="cuda")
                        tgt[i, :L] = torch.tensor(tgts[i], device="cuda")
                        msk[i, :L] = 1.0
                new_attn = (torch.arange(maxL, device="cuda")[None, :]
                            < torch.tensor(lens + [0] * (B - bs),
                                           device="cuda")[:, None]).long()
                attn = torch.cat([pre_attn, new_attn], dim=1)
                cache.crop(plen)                          # 复用前缀，去掉上批 KV
                o = self.model(input_ids=inp, past_key_values=cache,
                               attention_mask=attn, use_cache=True)
                lg = o.logits.float()                      # [B, maxL, V]
                gathered = lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [B, maxL]
                lse = torch.logsumexp(lg, dim=-1)          # [B, maxL]
                rest_lp = ((gathered - lse) * msk).sum(dim=1)           # [B]
                for i in range(bs):
                    ct = chunk[i]
                    s = first[i] + rest_lp[i].item()
                    scores[b + i] = s / len(ct) if (self.norm == "mean" and ct) else s
        return scores

    def select(self, query, binding=None):
        history, chosen, used = [], [], set()
        for _ in range(self.max_steps):
            scores = self._score_all(self._prefix(query, history))
            for i in used:                    # 已选技能不重复选（受限解码去重）
                scores[i] = float("-inf")
            best = max(range(len(scores)), key=lambda i: scores[i])
            if best == len(self.pool):        # "done"
                break
            used.add(best)
            sid = self.cand_ids[best]
            chosen.append(self.by_id[sid])
            history.append(_name(self.by_id[sid]))
        return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--norm", default="sum", choices=["sum", "mean"])
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    pool = load_pool()
    queries = load_queries(n=args.n, seed=0)
    t0 = time.time()
    b4 = B4LoTaBench(pool, model_id=args.model, batch_size=args.batch, norm=args.norm)
    print(f"池 {len(pool)} 条；模型 {args.model}；加载 {time.time()-t0:.1f}s\n")
    cov = 0
    for q in queries:
        t = time.time()
        chosen = b4.select(q["instruction"])
        req = set(q["required_effects"])
        union = set()
        for c in chosen:
            union |= skill_sig(c)
        ok = union >= req
        cov += int(ok)
        print(f"[{time.time()-t:5.1f}s] req={sorted(req)} -> {[c['skill_id'] for c in chosen]} "
              f"覆盖(声明)={'OK' if ok else 'X'}")
    print(f"\n覆盖(声明) {cov}/{len(queries)} = {cov/max(len(queries),1):.1%}")


if __name__ == "__main__":
    main()
