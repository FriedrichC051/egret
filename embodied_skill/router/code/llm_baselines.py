"""Phase 4 LLM 类方法：
- B0：把全部技能（id + name + 效果签名）塞进 prompt，让 LLM 直接选出技能集合。
- （B3 的"选择层"、M1 复用这里的 llm_call。）

单跑计时/自检：
  python router/code/llm_baselines.py --n 3
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load_pool, load_queries, skill_sig  # noqa: E402

KIMI_BASE = "https://api.moonshot.cn/v1"
# 高速版：2~5s/次且质量与 k2.7-code 相当（普通版高峰期常 >150s 超时，不可用）
KIMI_MODEL = "kimi-k2.7-code-highspeed"

_ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(HERE)), ".env")  # embodied_skill/.env


def _ensure_api_key():
    """路由阶段在 Windows 跑时，Kimi key 不在系统环境里 —— 从 gitignored .env 读。
    兼容 MOONSHOT_API_KEY / OPENAI_API_KEY 两种命名。"""
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

SYSTEM = ("你是具身智能任务的技能路由器。给定一个自然语言任务和一个技能库，"
          "你要从库中选出**能共同完成该任务**的最小技能集合。"
          "技能可以是复合技能（一条即可完成多步）或多条原子技能的组合。"
          "只输出被选中技能的 skill_id，构成一个 JSON 字符串数组，不要输出任何解释。")


def llm_call(system, user, model=KIMI_MODEL, base_url=KIMI_BASE, timeout=120.0,
             temperature=1):
    from openai import OpenAI
    _ensure_api_key()
    if base_url and "moonshot" in base_url:
        for _p in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"]:
            os.environ.pop(_p, None)
    # max_retries=0：关掉 SDK 自动重试，否则一次超时会重试 2 次 → 实际耗时 3×timeout
    client = OpenAI(base_url=base_url, timeout=timeout, max_retries=0)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature,
    )
    return resp.choices[0].message.content


def catalog_text(pool):
    lines = []
    for c in pool:
        sig = ",".join(sorted(skill_sig(c))) or "-"
        lines.append(f"- {c['skill_id']} | {c.get('name', '')} | effects: {sig}")
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

    def __init__(self, pool, model=KIMI_MODEL, base_url=KIMI_BASE):
        self.pool = pool
        self.by_id = {c["skill_id"]: c for c in pool}
        self.catalog = catalog_text(pool)
        self.model = model
        self.base_url = base_url

    def select(self, query):
        user = (f"技能库（共 {len(self.pool)} 条）：\n{self.catalog}\n\n"
                f"任务：{query}\n\n请输出能完成该任务的最小技能集合（skill_id 的 JSON 数组）。")
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
    print(f"技能池 {len(pool)} 条；计时 {len(queries)} 次 B0 调用\n")
    for q in queries:
        t = time.time()
        chosen = b0.select(q["instruction"])
        dt = time.time() - t
        req = set(q["required_effects"])
        union = set()
        for c in chosen:
            union |= skill_sig(c)
        print(f"[{dt:5.1f}s] req={sorted(req)} -> 选中 {[c['skill_id'] for c in chosen]} "
              f"覆盖={'✓' if union >= req else '✗'}")


if __name__ == "__main__":
    main()
