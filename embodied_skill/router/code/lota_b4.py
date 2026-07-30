"""B4 = LoTa-Bench-style LLM step-wise constrained decoding for skill selection (ICLR'24, LLMTaskPlanning).

Fundamental difference from B0: **does not stuff the skill library into the prompt**. Instead --
  prefix = few-shot examples + task + selected history (short);
  at each step, for **every skill in the pool** (its natural-language name) compute the
  "token logprob as a continuation" and take the argmax (constrained decoding: only in-pool
  skills can be chosen), append to the history, and autoregress until "done".
=> A local open-source completion model suffices (token logits needed), zero API cost. The
   chat-LLM route is covered by B0.

Model: local HF causal LM (fp16, cuda), default Qwen2.5-1.5B, experiments use Qwen2.5-3B (override via B4_MODEL).
Interface: B4LoTaBench(pool).select(query, binding=None) -> [card] (consistent with M1/B0/B3).

Three deviations from the LoTa-Bench paper (see stage2_execution_plan.md WP3 footnote):
  1) Model: the paper uses 2023-era GPT-Neo/GPT-J/OPT/LLaMA-1 (+GPT API), no Qwen; we reproduce
     the method with the modern open-source backbone Qwen2.5 (local 3B SR ~= 29% falls within the
     paper's GPT-3/4 range, not weakening the baseline).
  2) Constrained decoding: not using the paper's `guidance` library; instead an equivalent
     per-candidate whole-string logprob + argmax (more stable and reproducible).
  3) Score normalization: the paper uses sum (equal-length template phrases) -> we use mean
     (composite skill names vary in length, and sum would degenerate into always picking the
     shortest atomic).
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


_MODEL_CACHE = {}  # model_id -> (tokenizer, model); multiple tier pools reuse the same weights (saves VRAM/loading)


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
        self.norm = norm  # "sum" (LoTa's original method) or "mean" (length-normalized)
        self.tok, self.model = _load_model(model_id)
        # candidate texts: each skill's name (+ "done"); pre-tokenize candidate tokens (no special tokens)
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
        """Batched prefix KV reuse: expand the prefix across the batch and run one forward pass
        to obtain the cache; for each candidate batch, right-pad and run one forward pass, then
        crop back to the prefix length for reuse (fast and VRAM-efficient)."""
        torch = self.torch
        B = self.batch_size
        pre = self.tok.encode(prefix, add_special_tokens=True)
        plen = len(pre)
        pad_id = self.tok.pad_token_id
        n = len(self.cand_tok)
        scores = [0.0] * n
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([pre], device="cuda"),
                             use_cache=True)                      # prefix batch=1 (saves VRAM)
            cache = out.past_key_values
            cache.batch_repeat_interleave(B)                      # replicate to batch B for reuse
            base_lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)  # first token
            pre_attn = torch.ones((B, plen), device="cuda", dtype=torch.long)
            for b in range(0, n, B):
                chunk = self.cand_tok[b:b + B]
                bs = len(chunk)
                first = [base_lp[ct[0]].item() if ct else -1e9 for ct in chunk]
                # input = ct[:-1] (placed after the prefix), prediction target = ct[1:]
                bodies = [ct[:-1] for ct in chunk]
                tgts = [ct[1:] for ct in chunk]
                lens = [len(t) for t in tgts]
                maxL = max(lens, default=0)
                if maxL == 0:                              # all single-token candidates
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
                cache.crop(plen)                          # reuse the prefix, drop the previous batch's KV
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
            for i in used:                    # do not reselect an already-chosen skill (constrained-decoding dedup)
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
    print(f"pool {len(pool)} cards; model {args.model}; loaded {time.time()-t0:.1f}s\n")
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
              f"covered(declared)={'OK' if ok else 'X'}")
    print(f"\ncovered(declared) {cov}/{len(queries)} = {cov/max(len(queries),1):.1%}")


if __name__ == "__main__":
    main()
