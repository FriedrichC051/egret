"""Phase 3：执行器 + 成功检查器 + 「金标准 vs 错误技能」校准对比。

目的（Pilot 关键校准，非路由本身）：
  验证因果链「给对技能 → 仿真成功；给错/给少技能 → 仿真失败」成立，
  从而证明后面用 SR/GC 评判路由器是可信的。

做法：对采样的若干任务，在其所属场景里分别执行三种技能集，比较 SR/GC：
  - gold ：该题的金标准复合技能（gt_composite_set[0]，是完整可执行的 sub_skills 序列）
  - naive：一个"看似合理但不充分"的技能（pick_and_place，只放置不加热/切片/清洗）
  - empty：什么都不做（下限对照）

成功检查（ALFRED 式）：
  - at_receptacle：终态 obj 的 parentReceptacles 含目标容器（放置持久，读终态）
  - isHot/isCold ：执行过程中 obj 温度"曾达到" Hot/Cold（峰值追踪，因后续步骤会改温度）
  - isSliced     ：执行过程中 obj "曾被切片"
  - isClean      ：清洗子技能成功（isDirty 由真变假）
  SR = required_effects 全部达成；GC = 达成比例。

用法（WSL，需 GPU 显示环境）：
  DISPLAY=:0 GALLIUM_DRIVER=d3d12 python exec/code/run_calibration.py --n 20
"""
import os
import sys
import json
import argparse
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))              # exec/code/
ROOT = os.path.dirname(os.path.dirname(HERE))                 # embodied_skill/
SKILLS_CODE = os.path.join(ROOT, "skills", "code")            # 可导入模块
SKILLS_DATA = os.path.join(ROOT, "skills", "datasets")        # 技能卡 JSON
GT_DATA = os.path.join(ROOT, "gt", "datasets")               # gt_kitchen.jsonl
sys.path.insert(0, SKILLS_CODE)
import atomic_bodies as ab            # noqa: E402
from atomic_bodies import _norm       # noqa: E402
from skill_effects import goal_signature  # noqa: E402

# 5 条金标准种子（优先用它们做可执行 gold，命名最规范）
SEED_IDS = ["pick_and_place", "heat_then_place", "cool_then_place",
            "clean_then_place", "slice_object_with_knife", "slice_then_place",
            "heat_then_slice_then_place"]


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_pool_index():
    """skill_id -> card（含 sub_skills）。合并原子 + pool_clean（复合+干扰）。"""
    idx = {}
    for c in load_json(os.path.join(SKILLS_DATA, "atomic_skills.json")):
        idx[c["skill_id"]] = c
    for c in load_json(os.path.join(SKILLS_DATA, "pool_clean.json")):
        idx[c["skill_id"]] = c
    return idx


def build_exec_gold_index(pool):
    """效果签名 -> 可执行复合金标准卡。

    gt_composite_set 的 set-cover 平局时可能选中裸原子(如 put_object)——它无 goto/pickup
    胶水步、不能独立执行。Phase 3 需要"真正可跑"的复合卡，故按 required_effects 的效果签名
    从 gold 复合卡里挑一条等价且完整可执行的来执行。种子卡优先（命名最规范）。
    """
    def rank(card):
        # 选卡优先级：种子 > 标注 gold > 其它可执行复合
        return (card["skill_id"] in SEED_IDS, bool(card.get("gold")))

    sig2card = {}
    for c in pool.values():
        # 纳入所有"可执行复合"（含 LLM 生成的 slice_then_place 等），不再只限种子 gold，
        # 否则切片类签名无匹配 → gold 回退裸原子 → 天花板被误拉到 0。
        if c.get("type") != "composite" or not c.get("is_executable"):
            continue
        if c.get("is_distractor") or not c.get("sub_skills"):
            continue
        sig = frozenset(goal_signature(c.get("effects", [])))
        prev = sig2card.get(sig)
        if prev is None or rank(c) > rank(prev):
            sig2card[sig] = c
    return sig2card


def pick_exec_gold(task, exec_gold_index, pool):
    """为任务挑一条可执行金标准复合卡：按 required_effects 效果签名匹配。"""
    sig = frozenset(task["required_effects"])
    card = exec_gold_index.get(sig)
    if card is not None:
        return card
    # 兜底：用 gt_composite_set[0]（可能是裸原子，仅当无匹配复合时）
    gid = task["gt_composite_set"][0] if task["gt_composite_set"] else None
    return pool.get(gid)


def find_target_objs(ev, base_type):
    """匹配目标物体：objectType 归一化后以 base 开头（兼容切片后 'AppleSliced'）。"""
    b = _norm(base_type)
    out = []
    for o in ev.metadata["objects"]:
        t = _norm(o["objectType"])
        if t.startswith(b) or b.startswith(t):
            out.append(o)
    return out


# 子技能 -> 它负责达成的效果 token（用于判"执行中确实做了这个动作"）
ACTION_TOKEN = {
    "heat_object_microwave": "isHot",
    "cool_object_fridge": "isCold",
    "slice_object": "isSliced",
    "clean_object_sink": "isClean",
}
EMPTY_FLAGS = {"isHot": False, "isCold": False, "isSliced": False, "isClean": False}


def read_state_flags(ctrl, task):
    """读当前世界状态下目标物体的效果 flag（用于记录初始态，供"扣除初始态"用）。"""
    obj_type = task["binding"]["obj"]
    recep = task["binding"]["receptacle"]
    f = {"isHot": False, "isCold": False, "isSliced": False, "isClean": False, "at_receptacle": False}
    rb = _norm(recep) if recep else None
    for o in find_target_objs(ctrl.last_event, obj_type):
        if o.get("temperature") == "Hot":
            f["isHot"] = True
        if o.get("temperature") == "Cold":
            f["isCold"] = True
        if o.get("isSliced") or "sliced" in _norm(o["objectType"]):
            f["isSliced"] = True
        if not o.get("isDirty", False):
            f["isClean"] = True
        if rb:
            for p in (o.get("parentReceptacles") or []):
                if rb in _norm(p):
                    f["at_receptacle"] = True
    return f


def execute_and_track(ctrl, card, bindings):
    """执行 sub_skills，返回 (flags 峰值达成, did 是否执行了对应动作, placed_to_target, log)。"""
    flags = dict(EMPTY_FLAGS)
    did = dict(EMPTY_FLAGS)
    placed_to_target = False
    log = []
    obj_type = bindings["obj"]
    for st in card.get("sub_skills", []):
        skill = st["skill"]
        raw_args = st.get("args", [])
        fn = ab.REGISTRY.get(skill)
        if fn is None:
            log.append((skill, False, "unknown skill"))
            continue
        args = [bindings.get(a, a) for a in raw_args]
        _, ok, info = fn(ctrl, *args)
        log.append((skill, ok, info))
        if ok and skill in ACTION_TOKEN:
            did[ACTION_TOKEN[skill]] = True
        # 放到"目标容器"（原始占位符 receptacle，非中间台面）且成功
        if ok and skill == "put_object" and len(raw_args) >= 2 and raw_args[1] == "receptacle":
            placed_to_target = True
        if skill == "clean_object_sink" and ok:
            flags["isClean"] = True
        # 峰值追踪目标物体温度/切片
        for o in find_target_objs(ctrl.last_event, obj_type):
            if o.get("temperature") == "Hot":
                flags["isHot"] = True
            if o.get("temperature") == "Cold":
                flags["isCold"] = True
            if o.get("isSliced") or "sliced" in _norm(o["objectType"]):
                flags["isSliced"] = True
    return flags, did, placed_to_target, log


def check_success(ctrl, task, init, flags, did, placed_to_target):
    """达成判定（扣除初始态）：token 达成 = 最终满足 AND (初始未满足 OR 执行了对应动作)。
    这样 empty(什么都不做) 无法靠"物体初始就在目标容器/初始就热"蒙混过关，
    同时不冤枉真执行的 gold（即便初始已满足，只要执行了对应动作也算它达成）。"""
    req = set(task["required_effects"])
    obj_type = task["binding"]["obj"]
    recep = task["binding"]["receptacle"]

    final_at = False
    if recep:
        rb = _norm(recep)
        for o in find_target_objs(ctrl.last_event, obj_type):
            for p in (o.get("parentReceptacles") or []):
                if rb in _norm(p):
                    final_at = True

    achieved = set()
    for tok in ["isHot", "isCold", "isSliced", "isClean"]:
        if tok in req and flags.get(tok) and (not init.get(tok) or did.get(tok)):
            achieved.add(tok)
    if "at_receptacle" in req and final_at and (not init.get("at_receptacle") or placed_to_target):
        achieved.add("at_receptacle")

    SR = (achieved == req)
    GC = len(achieved) / max(1, len(req))
    return SR, GC, achieved


def run_condition(ctrl, task, card, bindings):
    ctrl.reset(task["floor_plan"])
    init = read_state_flags(ctrl, task)      # 执行前记录初始态
    if card is None:                         # empty 对照
        flags, did, placed, log = dict(EMPTY_FLAGS), dict(EMPTY_FLAGS), False, []
    else:
        flags, did, placed, log = execute_and_track(ctrl, card, bindings)
    SR, GC, achieved = check_success(ctrl, task, init, flags, did, placed)
    return SR, GC, achieved, log, init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=os.path.join(GT_DATA, "gt_kitchen.jsonl"))
    ap.add_argument("--n", type=int, default=20, help="采样任务数")
    ap.add_argument("--seed", type=int, default=0, help="随机采样种子")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    pool = load_pool_index()
    exec_gold_index = build_exec_gold_index(pool)
    naive_card = pool.get("pick_and_place")

    # 读 GT，均衡采样两种任务类型
    rows = []
    with open(args.gt, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    import random
    rng = random.Random(args.seed)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["task_type"]].append(r)
    sample = []
    per = max(1, args.n // max(1, len(by_type)))
    for tt, lst in by_type.items():
        lst = list(lst)
        rng.shuffle(lst)
        sample += lst[:per]
    rng.shuffle(sample)
    sample = sample[: args.n]

    from ai2thor.controller import Controller
    from ai2thor.platform import Linux64
    ctrl = Controller(platform=Linux64, scene="FloorPlan1", width=300, height=300, x_display="0")

    agg = {m: {"SR": 0, "GC": 0.0} for m in ["gold", "naive", "empty"]}
    dirty_init = 0    # 初始态已满足部分 required 的题数（数据初始态不干净度量）
    print(f"== Phase3 校准：{len(sample)} 题，每题跑 gold/naive/empty 三条件 ==\n")
    for r in sample:
        binding = dict(r["binding"])
        binding.setdefault("liquid", "water")
        gold_card = pick_exec_gold(r, exec_gold_index, pool)
        gold_id = gold_card["skill_id"] if gold_card else None
        conds = {"gold": gold_card, "naive": naive_card, "empty": None}
        line = f"[{r['task_type'][:16]:16}] {r['task_id'][:22]:22} req={sorted(r['required_effects'])}"
        res = {}
        init_seen = None
        for m, card in conds.items():
            SR, GC, ach, log, init = run_condition(ctrl, r, card, binding)
            init_seen = init
            agg[m]["SR"] += int(SR)
            agg[m]["GC"] += GC
            res[m] = (SR, GC, ach)
            if args.verbose and m == "gold" and not SR:
                fails = [s for s in log if s[1] is False]
                line += f"\n      gold失败步={fails}"
        pre = {t for t in r["required_effects"] if init_seen and init_seen.get(t)}
        if pre:
            dirty_init += 1
            line += f"\n      [初始已满足 {sorted(pre)}，已扣除]"
        print(line)
        print(f"      gold SR={res['gold'][0]} GC={res['gold'][1]:.2f} 达成{sorted(res['gold'][2])}"
              f" | naive SR={res['naive'][0]} GC={res['naive'][1]:.2f}"
              f" | empty GC={res['empty'][1]:.2f}  (gold={gold_id})")
    ctrl.stop()

    n = len(sample)
    print("\n================ 校准汇总 ================")
    for m in ["gold", "naive", "empty"]:
        print(f"  {m:6}: SR={agg[m]['SR']}/{n}={agg[m]['SR']/n:.1%}  GC平均={agg[m]['GC']/n:.1%}")
    print(f"  初始态不干净的题（已扣除）：{dirty_init}/{n}")
    print("\n结论判据：gold 的 SR 应显著高于 naive/empty → 因果链成立，SR/GC 可信。")


if __name__ == "__main__":
    main()
