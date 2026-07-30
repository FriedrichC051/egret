"""L0-zero：LLM few-shot 目标效果解析器（零训练版，M1 正式版 pipeline 的 L0）。

与规则 parse_goal **同签名、同输出**——从指令解析出固定 6 词表的效果集合：
  {isHot, isCold, isClean, isSliced, isFilledWithLiquid, at_receptacle}
故可无缝替换 domain_m1 里的 parse_goal，下游 L2 合法性过滤 / L4 set-cover / L5 修复全不动。

设计要点：
  · 闭集本体：prompt 里穷举 6 个合法 token 并逐个解释，禁止发明新词（防幻觉第一闸）。
  · few-shot 挑难例：微波炉/冰箱"当容器 vs 当动词"歧义、同义改写、多步连词。
  · 结构化输出 + 后校验：只保留落在 6 词表内的 token；JSON 解析失败 → 回退规则 parse_goal。
  · at_receptacle 优先信 binding（容器实例来自任务，不让 LLM 猜）。
  · temperature=0 + 按 instruction 磁盘缓存（211 题各调一次，跨方法/重跑零重复开销）。
公平红线：只吃 instruction(+binding 容器名)，**绝不读 required_effects(GT)**，与 M1/baseline 同源。

单跑自检（对 211 评测集，报 LLM vs 规则 vs GT 的 exact-match）：
  python router/code/goal_parser_llm.py --n 30
"""
import os
import re
import sys
import json
import time
import hashlib
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import parse_goal  # noqa: E402  规则兜底

VOCAB = {"isHot", "isCold", "isClean", "isSliced", "isFilledWithLiquid", "at_receptacle"}
_CACHE_PATH = os.path.join(HERE, os.pardir, os.pardir, "eval", "datasets", "goal_llm_cache.json")

SYSTEM = (
    "你是具身家务任务的目标效果解析器。给定一条自然语言指令，判断任务要求物体达到哪些"
    "最终状态效果。只能从以下固定 6 个标签中选，禁止发明新标签：\n"
    "- isHot              物体被加热（heat/cook/microwave 作动词/nuke/warm/boil/roast/bake）\n"
    "- isCold             物体被冷却（cool/chill/freeze/refrigerate 作动词）\n"
    "- isClean            物体被清洗（clean/wash/rinse/scrub）\n"
    "- isSliced           物体被切开（slice/cut/chop/section）\n"
    "- isFilledWithLiquid 容器被装液体（fill/pour water）\n"
    "- at_receptacle      物体被放到指定容器/表面（put/place/on/into）\n"
    "注意：\"put it IN the microwave/fridge\" 是把微波炉/冰箱当**容器**（只算 at_receptacle），"
    "不是加热/制冷；只有 \"microwave/heat the X\"、\"cool/chill the X\" 作**动词**才算状态变化。\n"
    "关键：状态形容词（clean/chilled/heated/sliced/washed 修饰物体，如 \"clean lettuce\"、"
    "\"chilled apple\"、\"a slice of X\"）表示**机器人必须达成该状态**（要去洗/去冷却/去切），"
    "而不是物体已经是那样——一律计入对应效果。\n"
    "只输出一个 JSON 字符串数组，例如 [\"isHot\",\"at_receptacle\"]，不要任何解释。"
)

FEWSHOT = (
    "指令: microwave the potato and put it on the countertop\n输出: [\"isHot\",\"at_receptacle\"]\n\n"
    "指令: put the mug in the microwave\n输出: [\"at_receptacle\"]\n\n"
    "指令: rinse the lettuce then chill it in the fridge\n输出: [\"isClean\",\"isCold\",\"at_receptacle\"]\n\n"
    "指令: slice the apple\n输出: [\"isSliced\"]\n\n"
    "指令: cut a slice of tomato, heat it, and place it on the table\n"
    "输出: [\"isSliced\",\"isHot\",\"at_receptacle\"]\n\n"
    "指令: place the spatula on the table\n输出: [\"at_receptacle\"]\n\n"
    "指令: place clean lettuce inside of the kitchen fridge\n输出: [\"isClean\",\"at_receptacle\"]\n\n"
    "指令: put a chilled pan on the counter\n输出: [\"isCold\",\"at_receptacle\"]\n\n"
    "指令: put a slice of apple on the table\n输出: [\"isSliced\",\"at_receptacle\"]\n\n"
)


def _parse_arr(text):
    """从 LLM 输出抽 JSON 数组并只保留合法 token；失败返回 None。"""
    if not text:
        return None
    s, e = text.find("["), text.rfind("]")
    if s == -1 or e == -1:
        return None
    try:
        arr = json.loads(text[s:e + 1])
    except Exception:
        return None
    return {x for x in arr if isinstance(x, str) and x in VOCAB}


class LLMGoalParser:
    """LLM few-shot 目标解析器，带磁盘缓存 + 规则兜底。callable：parse(instruction, binding)。"""

    def __init__(self, cache_path=_CACHE_PATH, temperature=1, verbose=False):
        # 注：Kimi highspeed 仅允许 temperature=1；靠 few-shot + 闭集 + 磁盘缓存保证稳定/可复现
        self.cache_path = cache_path
        self.temperature = temperature
        self.verbose = verbose
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            try:
                self.cache = json.load(open(cache_path, encoding="utf-8"))
            except Exception:
                self.cache = {}
        self.n_call = self.n_hit = self.n_fallback = 0

    def _flush(self):
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            json.dump(self.cache, open(self.cache_path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=0)

    def parse(self, instruction, binding=None):
        key = hashlib.md5((instruction or "").encode("utf-8")).hexdigest()
        if key in self.cache:
            self.n_hit += 1
            goal = set(self.cache[key])
        else:
            from llm_baselines import llm_call
            goal = None
            try:
                self.n_call += 1
                user = f"{FEWSHOT}指令: {instruction}\n输出:"
                text = llm_call(SYSTEM, user, temperature=self.temperature)
                goal = _parse_arr(text)
            except Exception as ex:
                if self.verbose:
                    print(f"  [L0] LLM 调用异常，规则兜底：{ex}", flush=True)
                goal = None
            if goal is None:                          # 解析失败 → 规则兜底
                self.n_fallback += 1
                goal = parse_goal(instruction, binding)
            self.cache[key] = sorted(goal)
            self._flush()
        # at_receptacle 优先信 binding（容器实例来自任务，不让 LLM 猜）
        binding = binding or {}
        if binding.get("receptacle"):
            goal.add("at_receptacle")
        return goal

    # 与 parse_goal 同签名，便于直接替换
    def __call__(self, instruction, binding=None):
        return self.parse(instruction, binding)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    from common import load_queries
    queries = load_queries(
        n=(None if args.n < 0 else args.n), seed=0,
        gt_path=os.path.join(HERE, os.pardir, os.pardir, "gt", "datasets", "gt_kitchen_eval.jsonl"))
    llm = LLMGoalParser(verbose=True)
    t0 = time.time()
    llm_ok = rule_ok = 0
    mism = []
    for q in queries:
        req = set(q["required_effects"])
        g_llm = llm.parse(q["instruction"], q.get("binding"))
        g_rule = parse_goal(q["instruction"], q.get("binding"))
        llm_ok += int(g_llm == req)
        rule_ok += int(g_rule == req)
        if g_llm != req and len(mism) < 15:
            mism.append((q["instruction"][:70], sorted(g_llm), sorted(g_rule), sorted(req)))
    n = len(queries)
    print(f"\nN={n}  LLM 调用 {llm.n_call}（缓存命中 {llm.n_hit}，兜底 {llm.n_fallback}）"
          f"  用时 {time.time()-t0:.1f}s")
    print(f"exact-match：LLM = {llm_ok}/{n} = {llm_ok/n:.1%}   规则 = {rule_ok}/{n} = {rule_ok/n:.1%}")
    print("\nLLM 仍解析错样例（指令 | LLM | 规则 | GT）：")
    for ins, l, r, g in mism:
        print(f"  - {ins!r}\n    LLM={l}  规则={r}  GT={g}")


if __name__ == "__main__":
    main()
