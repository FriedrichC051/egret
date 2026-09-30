"""Phase 4 LLM-style methods:
- B0: stuff all skills (id + name + effect signature) into the prompt and let the LLM pick the
  skill set directly.
- (B3's "selection layer" and M1 reuse llm_call from here.)

Standalone timing/self-check:
  python router/code/llm_baselines.py --n 3
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load_pool, load_queries, skill_sig, skill_struct_text  # noqa: E402

KIMI_BASE = "https://api.moonshot.cn/v1"
# High-speed variant: 2-5s/call with quality on par with k2.7-code (the standard variant
# often times out >150s during peak hours and is unusable)
KIMI_MODEL = "kimi-k2.7-code-highspeed"

_ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(HERE)), ".env")  # embodied_skill/.env


def _ensure_api_key():
    """When the routing stage runs on Windows, the Kimi key is not in the system environment --
    read it from the gitignored .env. Supports both MOONSHOT_API_KEY / OPENAI_API_KEY names."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    mk = os.environ.get("MOONSHOT_API_KEY")
    if mk:
        os.environ["OPENAI_API_KEY"] = mk
        return
    if os.path.exists(_ENV_FILE):
        for line in open(_ENV_FILE, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k in ("OPENAI_API_KEY", "MOONSHOT_API_KEY") and v:
                os.environ["OPENAI_API_KEY"] = v
                return

SYSTEM = ("You are a skill router for embodied-AI tasks. Given a natural-language task and a "
          "skill library, select from the library the minimal skill set that **together "
          "completes the task**. "
          "A skill may be a composite skill (one card completing multiple steps) or a "
          "combination of several atomic skills. "
          "Output only the skill_id of the selected skills as a JSON array of strings, "
          "with no explanation.")


def llm_call(system, user, model=KIMI_MODEL, base_url=KIMI_BASE, timeout=120.0,
             temperature=1):
    from openai import OpenAI
    _ensure_api_key()
    if base_url and "moonshot" in base_url:
        for _p in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"]:
            os.environ.pop(_p, None)
    # max_retries=0: disable the SDK's automatic retries, otherwise one timeout retries twice -> actual cost 3x timeout
    client = OpenAI(base_url=base_url, timeout=timeout, max_retries=0)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature,
    )
    return resp.choices[0].message.content


def catalog_text(pool, full_card=False, by_id=None):
    """Skill catalog. With full_card=True each entry also carries its symbolic preconditions
    and atomic-expanded sub-skill sequence (information-matched baseline setting: the raw
    structure is given, the recomputed effects are not)."""
    if full_card and by_id is None:
        by_id = {c["skill_id"]: c for c in pool}
    lines = []
    for c in pool:
        sig = ",".join(sorted(skill_sig(c))) or "-"
        line = f"- {c['skill_id']} | {c.get('name', '')} | effects: {sig}"
        if full_card:
            line += " | " + skill_struct_text(c, by_id)
        lines.append(line)
    return "\n".join(lines)


def parse_id_array(text, valid_ids):
    s, e = text.find("["), text.rfind("]")
    if s == -1 or e == -1:
        return []
    try:
        arr = json.loads(text[s:e + 1])
    except Exception:
        return []
    return [x for x in arr if isinstance(x, str) and x in valid_ids]


class B0LLMFull:
    name = "B0-LLM-Full"

    def __init__(self, pool, model=KIMI_MODEL, base_url=KIMI_BASE, full_card=False):
        self.pool = pool
        self.by_id = {c["skill_id"]: c for c in pool}
        self.full_card = full_card
        if full_card:
            self.name = "B0-LLM-Full-Card"
        self.catalog = catalog_text(pool, full_card=full_card, by_id=self.by_id)
        self.model = model
        self.base_url = base_url

    def select(self, query):
        user = (f"Skill library ({len(self.pool)} cards total):\n{self.catalog}\n\n"
                f"Task: {query}\n\nOutput the minimal skill set that completes the task "
                f"(a JSON array of skill_id).")
        text = llm_call(SYSTEM, user, self.model, self.base_url)
        ids = parse_id_array(text, set(self.by_id))
        return [self.by_id[i] for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    pool = load_pool()
    queries = load_queries(n=args.n, seed=0)
    b0 = B0LLMFull(pool)
    print(f"Skill pool: {len(pool)} cards; timing {len(queries)} B0 calls\n")
    for q in queries:
        t = time.time()
        chosen = b0.select(q["instruction"])
        dt = time.time() - t
        req = set(q["required_effects"])
        union = set()
        for c in chosen:
            union |= skill_sig(c)
        print(f"[{dt:5.1f}s] req={sorted(req)} -> selected {[c['skill_id'] for c in chosen]} "
              f"covered={'OK' if union >= req else 'X'}")


if __name__ == "__main__":
    main()
