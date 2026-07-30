"""Phase 4 路由评测的公共数据层：技能池加载、检索文档构造、效果签名、query 加载。

关键约定（见 plan 第七节「效果签名 + set-cover 等价判定」）：
- 判正确不看技能 ID 是否完全一致，而看**目标效果覆盖**：
  - 单技能命中（Hit）：某技能的效果签名 ⊇ required_effects（一条就够）。
  - 完整覆盖（FullCoverage）：一组技能的效果签名并集 ⊇ required_effects。
- 技能 name/effects/description 均为英文（description 已由 englishify 批量翻译），
  ALFRED query 也是英文；检索文档 = name + skill_id 拆词 + effects + description。
"""
import os
import sys
import json
import re
import random

HERE = os.path.dirname(os.path.abspath(__file__))                 # router/code/
ROOT = os.path.dirname(os.path.dirname(HERE))                     # embodied_skill/
SKILLS_CODE = os.path.join(ROOT, "skills", "code")
SKILLS_DATA = os.path.join(ROOT, "skills", "datasets")
GT_DATA = os.path.join(ROOT, "gt", "datasets")
sys.path.insert(0, SKILLS_CODE)
from skill_effects import goal_signature, combine_sequential  # noqa: E402


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_pool():
    """完整技能池：13 原子 + pool_clean（种子复合 + 生成复合 + 干扰变体）。按 skill_id 去重。
    可用环境变量 POOL_FILE 覆盖 pool_clean（用于混淆度分档池 pool_tier_{low,mid,high}.json）；
    传绝对路径或 skills/datasets 下的文件名均可。"""
    pool_file = os.environ.get("POOL_FILE") or os.path.join(SKILLS_DATA, "pool_clean.json")
    if not os.path.isabs(pool_file):
        pool_file = os.path.join(SKILLS_DATA, pool_file)
    pool = []
    seen = set()
    for path in [os.path.join(SKILLS_DATA, "atomic_skills.json"), pool_file]:
        for c in load_json(path):
            sid = c.get("skill_id")
            if sid and sid not in seen:
                pool.append(c)
                seen.add(sid)
    return pool


def _split_id(sid):
    return " ".join(str(sid).split("_"))


def skill_doc(card):
    """检索用文档文本：name + 拆词后的 skill_id + 效果 token + description。"""
    effs = " ".join(sorted(skill_sig(card)))
    return f"{card.get('name', '')}. {_split_id(card.get('skill_id', ''))}. " \
           f"effects: {effs}. {card.get('description', '')}"


def skill_sig(card):
    """技能【声明】的目标效果签名（token 集合）。注意：干扰项会**谎报** effects，
    仅用于检索文档/展示；判正确请用下面的 true_effect_sig（从 sub_skills 重算）。"""
    return frozenset(goal_signature(card.get("effects", [])))


# ---------- effect-grounded coverage（WP2a）：从 sub_skills 重算真实效果 ----------
# 干扰项的声明 effects 可能撒谎（lookalike 把 heat 写成 cool、missing 少一步仍写全）。
# 判正确必须看"真实会造成的状态变化"，即按 sub_skills 顺序合成的效果签名 + 是否
# 把任务物体放到任务容器（at_receptacle）。所有方法统一用它，纸面指标才与仿真对齐。

def flatten_atomic(card, by_id, depth=0):
    """展开为原子 (skill_id, args) 序列；sub_skills 若引用复合则递归。原子卡→自身。"""
    if depth > 6:
        return []
    subs = card.get("sub_skills")
    if not subs:
        return [(card.get("skill_id"), card.get("params", []))]
    seq = []
    for st in subs:
        sid, args = st.get("skill"), st.get("args", [])
        child = by_id.get(sid)
        if child and child.get("sub_skills"):
            seq += flatten_atomic(child, by_id, depth + 1)
        else:
            seq.append((sid, args))
    return seq


def true_transform_sig(card, by_id):
    """卡的"变换类"真实目标签名（isHot/isCold/isClean/isSliced/isFilled，热态互斥覆盖）。
    **不含 at_receptacle**——放置是否命中要按任务容器判定（见 places_at_target）。"""
    subs = card.get("sub_skills")
    if not subs:
        toks = set(goal_signature(card.get("effects", [])))
        toks.discard("at_receptacle")
        return toks
    token_sets = []
    for sid, _ in flatten_atomic(card, by_id):
        child = by_id.get(sid)
        if child:
            token_sets.append(goal_signature(child.get("effects", [])))
    sig = set(combine_sequential(token_sets))
    sig.discard("at_receptacle")
    return sig


def places_at_target(card, by_id, binding):
    """卡是否把"任务物体"放到"任务容器"：取最后一个 put_object，其容器须为占位符
    receptacle（会绑定到任务容器）或硬编码值恰等于任务容器。→ 挡住"换错容器"lookalike。"""
    binding = binding or {}
    obj, recep = binding.get("obj"), binding.get("receptacle")
    last_put = None
    for sid, args in flatten_atomic(card, by_id):
        if sid == "put_object":
            last_put = args
    if not last_put:
        return False
    put_obj = last_put[0] if len(last_put) > 0 else None
    put_rec = last_put[1] if len(last_put) > 1 else None
    obj_ok = put_obj in ("obj", "target") or (obj and put_obj == obj)
    rec_ok = put_rec == "receptacle" or (recep and put_rec == recep)
    return bool(obj_ok and rec_ok)


def true_effect_sig(card, by_id, binding=None):
    """给定任务 binding 的**真实**有效覆盖集：变换签名 +（命中任务容器则 at_receptacle）。
    对干扰项如实反映"实际能达成什么"，是 effect-grounded coverage 的判定基元。"""
    sig = set(true_transform_sig(card, by_id))
    if places_at_target(card, by_id, binding):
        sig.add("at_receptacle")
    return sig


# ---------- 指令 → 目标效果解析（M1 与各 baseline 同源：只吃指令，不读 GT/标签）----------
# 放在 common 便于 M1、WP2b 校验脚本共用同一套规则（单一真源）。
_GOAL_KEYWORDS = [
    # 注意：microwave 作"容器"(put ... in the microwave)不加热，只有作**动词**
    # (microwave the/a/it ...) 才触发 isHot；否则会把"放进微波炉"误判为加热。
    ("isHot", r"\bheat|\bhot\b|warm|\bcook|boil|roast|microwaved|\bmicrowave\s+(a|an|the|some|this|it|your)\b"),
    # fridge/refrigerator 常作"容器"(place in fridge)，不触发 isCold；只认明确降温动词
    # （含 refrigerate/refrigerated 动词，但"refrigerator"名词不匹配——词形不同）。
    ("isCold", r"\bcool|\bchill|\bcold\b|frozen|freeze|refrigerate"),
    ("isClean", r"clean|wash|rins"),
    ("isSliced", r"slice|\bcut\b|chop|\bpiece|\bslices?\b"),
    ("isFilledWithLiquid", r"\bfill|filled|pour"),
]
_GOAL_RE = [(tok, re.compile(pat)) for tok, pat in _GOAL_KEYWORDS]


def parse_goal(instruction, binding=None):
    """从指令文本解析目标效果 token 集合。有 receptacle 绑定/放置动词 → at_receptacle。"""
    s = (instruction or "").lower()
    goal = set()
    for tok, rx in _GOAL_RE:
        if rx.search(s):
            goal.add(tok)
    binding = binding or {}
    if binding.get("receptacle") or re.search(r"\bput|place|\bon\b|\binto\b|\bin\b", s):
        goal.add("at_receptacle")
    return goal


_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text):
    """英文按词、中文按单字切分（BM25 用）。"""
    return _TOKEN_RE.findall(str(text).lower())


def load_queries(n=None, seed=0, gt_path=None):
    """从 gt_kitchen.jsonl 均衡采样 query。每条含 instruction + required_effects +
    金标准集合（id 级）+ binding/floor_plan（供仿真）。n=None 取全部。"""
    gt_path = gt_path or os.path.join(GT_DATA, "gt_kitchen.jsonl")
    rows = []
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if n is None:
        return rows
    rng = random.Random(seed)
    by_type = {}
    for r in rows:
        by_type.setdefault(r["task_type"], []).append(r)
    per = max(1, n // max(1, len(by_type)))
    sample = []
    for _, lst in by_type.items():
        lst = list(lst)
        rng.shuffle(lst)
        sample += lst[:per]
    rng.shuffle(sample)
    return sample[:n]
