"""Prompt templates and calling scaffold for LLM batch-synthesis of composite
skills & confusing variants.

Design points:
- A composite skill's body = a sub_skills sequence (can only reference existing
  atomic skill_ids); the LLM does not generate code, only JSON.
- Two synthesis modes:
    1) composite -- generate "new, semantically sound, executable" composite
       skills (beyond the 5 gold standards).
    2) variant   -- generate "confusing variants" of existing skills (reorder /
       missing step / lookalike rewording) as hard distractors.
- Default DRY-RUN: only print the assembled prompt for review ("fix the template
  first"); add --run to actually call OpenAI.
- Recommended to run in the Windows skill-router env (openai already installed):
  requires the OPENAI_API_KEY environment variable.

Usage:
    python synth_composite.py --mode composite --n 50            # just view the prompt
    python synth_composite.py --mode composite --n 50 --run      # actually call and generate
    python synth_composite.py --mode variant --run               # build variants of existing skills
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))          # skills/code/
DATA = os.path.join(HERE, os.pardir, "datasets")           # skills/datasets/

# Atomic-skill actions (and their params) that may be referenced in sub_skills
ATOMIC_FILE = os.path.join(DATA, "atomic_skills.json")
SEED_FILE = os.path.join(DATA, "composite_skills_seed.json")

# Literal container/tool types allowed to appear in args (besides placeholders)
LITERAL_TYPES = ["Microwave", "Fridge", "SinkBasin", "Faucet", "Knife",
                 "CounterTop", "DiningTable", "GarbageCan", "StoveBurner", "Cabinet"]
# Parameter placeholders
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
    """Accumulate and dedup by skill_id, return (merged, added_count)."""
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


SYSTEM_PROMPT = """You are an assistant that builds a skill library for embodied AI (AI2-THOR kitchen scenes).
You produce "composite skill cards"; each card's body is a sequence of sub_skills (a sub-skill sequence) that may only reference the given atomic skill_ids.
You write no executable code, only strict JSON. All skills target "kitchen chore" tasks (pick up / place / heat / cool / clean / slice, etc.)."""

COMPOSITE_RULES = """Output requirements (composite mode):
1. In sub_skills[].skill you may only use the following atomic skill_ids: {atomic_ids}
2. Parameters in sub_skills[].args may only be: placeholders {placeholders}, or literal object types {literals}.
3. Semantics must be self-consistent: goto before interacting; the target object must already be picked up before heat/cool/clean; goto the target container before put_object.
4. effects must match the actual state changes caused by sub_skills (e.g. if it contains heat_object_microwave then effects contains "obj.isHot"; if it contains put_object then it contains "obj.at(receptacle)").
5. The generated composite skills must be "diverse and realistic": cover different action combinations and orders (e.g. slice->heat->place, clean->place, cool->place, two-step place, etc.), and avoid duplicating the examples.
6. Each card's fields: skill_id (snake_case, unique), name, description (English, 1 sentence), type="composite", params, sub_skills, preconditions, effects, is_executable=true, gold=false, body="composite_bodies.execute_composite".
7. Output only a single JSON array, no extra text."""

VARIANT_RULES = """Output requirements (variant mode, build "confusing hard distractors"):
For each given "base skill", generate 2-3 variants that "look very similar but are actually wrong/non-equivalent", to test the router's discrimination. Variant types:
- reorder: shuffle the order of key sub-skills (e.g. put before heat), breaking the semantics.
- missing: delete a necessary sub-skill (e.g. remove the heat step from a heating task).
- lookalike: description and name are very close, but one action is replaced with a wrong one (e.g. replace heat with cool).
Each variant card has extra fields: `is_distractor=true`, `variant_of` (base skill_id), `variant_type` (reorder/missing/lookalike), `why_wrong` (one sentence on why it does not satisfy the original task).
Label is_executable truthfully according to whether it can still run in simulation (reorder/missing usually still execute but produce a wrong result).
Output only a single JSON array, no extra text."""


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
        avoid_block = ("\n\n**Existing skill_ids (no duplicates; you must use entirely new, semantically distinct names):**\n"
                       + ", ".join(sorted(avoid_ids)))
    user = f"""Available atomic skill catalog:
{catalog_desc}

Composite skill examples (gold standard, for you to imitate the format, do not copy verbatim):
{json.dumps(examples[:3], ensure_ascii=False, indent=2)}

{rules}{avoid_block}

Now generate {n} **new, mutually distinct** composite skill cards (JSON array)."""
    return SYSTEM_PROMPT, user


def build_variant_prompt(base):
    user = f"""Base skills (build 2-3 confusing variants for each):
{json.dumps(base, ensure_ascii=False, indent=2)}

{VARIANT_RULES}"""
    return SYSTEM_PROMPT, user


def call_llm(system, user, model, base_url=None):
    """Compatible with OpenAI and Kimi (Moonshot). For Kimi just set
    base_url=https://api.moonshot.cn/v1.
    The key is always read from the OPENAI_API_KEY env var; base_url from the
    argument or OPENAI_BASE_URL."""
    from openai import OpenAI
    # Kimi's China domain must bypass the system proxy (Clash)
    if base_url and "moonshot" in base_url:
        for _p in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"]:
            os.environ.pop(_p, None)
    base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    # timeout prevents a single request to a reasoning model from hanging forever; raise for large batches
    client = OpenAI(base_url=base_url, timeout=300.0) if base_url else OpenAI(timeout=300.0)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=1,  # kimi-k2.7-code only allows temperature=1
    )
    return resp.choices[0].message.content


def extract_json_array(text):
    s = text.find("[")
    e = text.rfind("]")
    if s == -1 or e == -1:
        raise ValueError("no JSON array found in the LLM output")
    return json.loads(text[s:e + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["composite", "variant"], default="composite")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--run", action="store_true", help="actually call the LLM (otherwise just print the prompt)")
    ap.add_argument("--provider", choices=["openai", "kimi"], default="kimi",
                    help="kimi=Moonshot (default); openai=OpenAI")
    ap.add_argument("--model", default=None, help="model name; if omitted, use the provider default")
    ap.add_argument("--base-url", default=None, help="override base_url; kimi defaults to https://api.moonshot.cn/v1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--base-limit", type=int, default=8,
                    help="variant mode: how many base skills to take per batch to build variants (prevents a single JSON from being truncated for being too long)")
    ap.add_argument("--base-offset", type=int, default=0,
                    help="variant mode: which base to start from in the base pool (batch over different bases to avoid duplicates)")
    args = ap.parse_args()

    # Decide default model and base_url per provider
    if args.provider == "kimi":
        base_url = args.base_url or "https://api.moonshot.cn/v1"
        model = args.model or "kimi-k2.7-code"
    else:
        base_url = args.base_url  # None -> official
        model = args.model or "gpt-4o-mini"

    if args.mode == "composite":
        default_out = os.path.join(DATA, "composite_skills_generated.json")
        out = args.out or default_out
        existing = load_json_if_exists(out)
        # Existing ids = seeds + already generated, injected into the prompt to prevent duplicates
        avoid_ids = {c.get("skill_id") for c in load_seed_examples()}
        avoid_ids |= {c.get("skill_id") for c in existing}
        system, user = build_composite_prompt(args.n, avoid_ids=avoid_ids)
    else:
        default_out = os.path.join(DATA, "confusing_variants_generated.json")
        out = args.out or default_out
        existing = load_json_if_exists(out)
        # Prefer the validated pool_clean.json (seeds + generated) as the variant
        # base, otherwise fall back to seeds. Only take "gold positives"
        # (non-distractors) as bases, and batch over different bases by offset to
        # avoid duplicates.
        pool = load_json_if_exists(os.path.join(DATA, "pool_clean.json")) or load_seed_examples()
        pool = [c for c in pool if not c.get("is_distractor")]
        base = pool[args.base_offset: args.base_offset + args.base_limit]
        if not base:
            sys.exit(f"base is empty: base_offset={args.base_offset} exceeds base pool size {len(pool)}")
        system, user = build_variant_prompt(base)

    if not args.run:
        print("===== SYSTEM =====\n" + system)
        print("\n===== USER =====\n" + user)
        print(f"\n[DRY-RUN] provider={args.provider} model={model} base_url={base_url}")
        print("add --run to actually call the LLM and write out", out)
        return

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("missing OPENAI_API_KEY environment variable (Kimi's key also goes in this variable)")
    print(f"[call] provider={args.provider} model={model} base_url={base_url}")
    text = call_llm(system, user, model=model, base_url=base_url)
    cards = extract_json_array(text)
    merged, added = merge_dedup(existing, cards)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"this batch generated {len(cards)}, added {added} (after dedup), total {len(merged)} -> {out}")


if __name__ == "__main__":
    main()
