"""LLM 批量合成复合技能 & 易混淆变体的 prompt 模板与调用脚手架。

设计要点：
- 复合技能的 body = sub_skills 序列（只能引用已有原子技能 skill_id），LLM 不生成代码，只产出 JSON。
- 两种合成模式：
    1) composite —— 生成"新的、语义合理、可执行"的复合技能（beyond 5 条金标准）。
    2) variant   —— 针对已有技能生成"易混淆变体"（错序 / 缺步 / 近义换汤不换药），当硬干扰项。
- 默认 DRY-RUN：只打印拼好的 prompt，供审阅"先定模板"；加 --run 才真正调用 OpenAI。
- 建议在 Windows 的 skill-router 环境跑（已装 openai）：需环境变量 OPENAI_API_KEY。

用法：
    python synth_composite.py --mode composite --n 50            # 只看 prompt
    python synth_composite.py --mode composite --n 50 --run      # 真正调用生成
    python synth_composite.py --mode variant --run               # 对现有技能造变体
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))          # skills/code/
DATA = os.path.join(HERE, os.pardir, "datasets")           # skills/datasets/

# 原子技能中允许在 sub_skills 里引用的动作及其参数
ATOMIC_FILE = os.path.join(DATA, "atomic_skills.json")
SEED_FILE = os.path.join(DATA, "composite_skills_seed.json")

# 允许出现在 args 里的字面容器/工具类型（占位符之外）
LITERAL_TYPES = ["Microwave", "Fridge", "SinkBasin", "Faucet", "Knife",
                 "CounterTop", "DiningTable", "GarbageCan", "StoveBurner", "Cabinet"]
# 参数占位符
PLACEHOLDERS = ["obj", "receptacle"]


def load_atomic_catalog():
    with open(ATOMIC_FILE, encoding="utf-8") as f:
        atoms = json.load(f)
    return [{"skill_id": a["skill_id"], "params": a["params"],
             "effects": a["effects"], "description": a["description"]} for a in atoms]


def load_seed_examples():
    with open(SEED_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_json_if_exists(path):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


def merge_dedup(existing, new_cards):
    """按 skill_id 累加去重，返回 (merged, added_count)。"""
    merged = list(existing)
    seen = {c.get("skill_id") for c in existing}
    added = 0
    for c in new_cards:
        sid = c.get("skill_id")
        if sid and sid not in seen:
            merged.append(c)
            seen.add(sid)
            added += 1
    return merged, added


SYSTEM_PROMPT = """你是具身智能（AI2-THOR 厨房场景）技能库的构建助手。
你要产出"复合技能卡"，每张卡的执行体是一串 sub_skills（子技能序列），只能引用给定的原子技能 skill_id。
你不写任何可执行代码，只输出严格的 JSON。所有技能面向"厨房家务"任务（拿取/放置/加热/冷却/清洗/切片等）。"""

COMPOSITE_RULES = """产出要求（composite 模式）：
1. 只能在 sub_skills[].skill 里使用下列原子技能 skill_id：{atomic_ids}
2. sub_skills[].args 里的参数只能是：占位符 {placeholders}，或字面物体类型 {literals}。
3. 语义要自洽：先 goto 再交互；加热/冷却/清洗前必须已 pickup 目标物；放置(put_object)前应先 goto 目标容器。
4. effects 必须与 sub_skills 实际造成的状态变化一致（如含 heat_object_microwave 则 effects 含 "obj.isHot"；含 put_object 则含 "obj.at(receptacle)"）。
5. 生成的复合技能要"多样且真实"：覆盖不同动作组合与顺序（如 切片→加热→放置、清洗→放置、冷却→放置、两步放置等），避免与示例重复。
6. 每张卡字段：skill_id(蛇形命名,唯一)、name、description(中文,1句)、type="composite"、params、sub_skills、preconditions、effects、is_executable=true、gold=false、body="composite_bodies.execute_composite"。
7. 只输出一个 JSON 数组，不要多余文字。"""

VARIANT_RULES = """产出要求（variant 模式，造"易混淆硬干扰项"）：
针对给定的每个"基准技能"，生成 2~3 个"看起来很像但其实错误/不等价"的变体，用来考验路由器的判别力。变体类型：
- 错序(reorder)：把关键子技能顺序打乱（如先 put 再 heat），语义被破坏。
- 缺步(missing)：删掉一个必要子技能（如加热任务里删掉 heat 步）。
- 近义诱饵(lookalike)：描述与名字非常接近，但把某个动作替换成错的（如把 heat 换成 cool）。
每张变体卡额外字段：`is_distractor=true`、`variant_of`(基准 skill_id)、`variant_type`(reorder/missing/lookalike)、`why_wrong`(一句话说明为何不满足原任务)。
is_executable 按其是否仍能在仿真里跑通如实标注（错序/缺步通常仍可执行但结果错）。
只输出一个 JSON 数组，不要多余文字。"""


def build_composite_prompt(n, avoid_ids=None):
    catalog = load_atomic_catalog()
    atomic_ids = ", ".join(a["skill_id"] for a in catalog)
    catalog_desc = "\n".join(f"- {a['skill_id']}(params={a['params']}, effects={a['effects']}): {a['description']}"
                             for a in catalog)
    examples = load_seed_examples()
    rules = COMPOSITE_RULES.format(
        atomic_ids=atomic_ids,
        placeholders=PLACEHOLDERS,
        literals=LITERAL_TYPES,
    )
    avoid_block = ""
    if avoid_ids:
        avoid_block = ("\n\n**已存在的 skill_id（禁止重复，必须全部使用新的、语义不同的命名）：**\n"
                       + ", ".join(sorted(avoid_ids)))
    user = f"""可用原子技能目录：
{catalog_desc}

复合技能示例（金标准，供你模仿格式，勿照抄）：
{json.dumps(examples[:3], ensure_ascii=False, indent=2)}

{rules}{avoid_block}

现在请生成 {n} 张**新的、互不重复**的复合技能卡（JSON 数组）。"""
    return SYSTEM_PROMPT, user


def build_variant_prompt(base):
    user = f"""基准技能（为它们各造 2~3 个易混淆变体）：
{json.dumps(base, ensure_ascii=False, indent=2)}

{VARIANT_RULES}"""
    return SYSTEM_PROMPT, user


def call_llm(system, user, model, base_url=None):
    """兼容 OpenAI 与 Kimi(Moonshot)。Kimi 只需 base_url=https://api.moonshot.cn/v1。
    key 统一从环境变量 OPENAI_API_KEY 读；base_url 从参数或 OPENAI_BASE_URL 读。"""
    from openai import OpenAI
    # Kimi 国内域名必须绕开系统代理(Clash)
    if base_url and "moonshot" in base_url:
        for _p in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"]:
            os.environ.pop(_p, None)
    base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    # timeout 防止推理模型单请求无限挂起；大批量时按需调大
    client = OpenAI(base_url=base_url, timeout=300.0) if base_url else OpenAI(timeout=300.0)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=1,  # kimi-k2.7-code 仅允许 temperature=1
    )
    return resp.choices[0].message.content


def extract_json_array(text):
    s = text.find("[")
    e = text.rfind("]")
    if s == -1 or e == -1:
        raise ValueError("LLM 输出里未找到 JSON 数组")
    return json.loads(text[s:e + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["composite", "variant"], default="composite")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--run", action="store_true", help="真正调用 LLM（否则仅打印 prompt）")
    ap.add_argument("--provider", choices=["openai", "kimi"], default="kimi",
                    help="kimi=Moonshot(默认)；openai=OpenAI")
    ap.add_argument("--model", default=None, help="模型名；不填按 provider 取默认")
    ap.add_argument("--base-url", default=None, help="覆盖 base_url；kimi 默认 https://api.moonshot.cn/v1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--base-limit", type=int, default=8,
                    help="variant 模式：每次取多少张基准技能造变体（防单次 JSON 过长被截断）")
    ap.add_argument("--base-offset", type=int, default=0,
                    help="variant 模式：从基准池的第几张开始取（分批覆盖不同基准，避免重复）")
    args = ap.parse_args()

    # 按 provider 决定默认 model 与 base_url
    if args.provider == "kimi":
        base_url = args.base_url or "https://api.moonshot.cn/v1"
        model = args.model or "kimi-k2.7-code"
    else:
        base_url = args.base_url  # None -> 官方
        model = args.model or "gpt-4o-mini"

    if args.mode == "composite":
        default_out = os.path.join(DATA, "composite_skills_generated.json")
        out = args.out or default_out
        existing = load_json_if_exists(out)
        # 已存在 id = 种子 + 已生成，注入 prompt 防重复
        avoid_ids = {c.get("skill_id") for c in load_seed_examples()}
        avoid_ids |= {c.get("skill_id") for c in existing}
        system, user = build_composite_prompt(args.n, avoid_ids=avoid_ids)
    else:
        default_out = os.path.join(DATA, "confusing_variants_generated.json")
        out = args.out or default_out
        existing = load_json_if_exists(out)
        # 变体基准优先用已校验的 pool_clean.json（种子+生成），否则退回种子。
        # 只取"金标准正例"(非干扰项)当基准，且按 offset 分批覆盖不同基准，避免重复。
        pool = load_json_if_exists(os.path.join(DATA, "pool_clean.json")) or load_seed_examples()
        pool = [c for c in pool if not c.get("is_distractor")]
        base = pool[args.base_offset: args.base_offset + args.base_limit]
        if not base:
            sys.exit(f"base 为空：base_offset={args.base_offset} 超出基准池大小 {len(pool)}")
        system, user = build_variant_prompt(base)

    if not args.run:
        print("===== SYSTEM =====\n" + system)
        print("\n===== USER =====\n" + user)
        print(f"\n[DRY-RUN] provider={args.provider} model={model} base_url={base_url}")
        print("加 --run 才会真正调用 LLM 并写出", out)
        return

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("缺少 OPENAI_API_KEY 环境变量（Kimi 的 key 也放这个变量）")
    print(f"[调用] provider={args.provider} model={model} base_url={base_url}")
    text = call_llm(system, user, model=model, base_url=base_url)
    cards = extract_json_array(text)
    merged, added = merge_dedup(existing, cards)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"本批生成 {len(cards)} 张，新增 {added} 张（去重后），累计 {len(merged)} 张 -> {out}")


if __name__ == "__main__":
    main()
