"""Phase 4 领域方法 M1（Pilot 免训练版）：B3 骨架 + 两个"领域钩子"。

把 SkillFlow(B3) 的后两层换成利用技能卡**结构化信息**的领域感知模块——
这些是通用文本流水线（B0/B3）用不上的东西：

  ① 检索层   ：Dense top-K（保留，和 B3 一致）
  ② 重排层→ 领域过滤：对候选做「手部状态符号模拟」，剔除物理非法链
              （put/heat/cool/clean/slice 前没先握住对应物体/刀 → 非法）。
              这刀砍掉所有 reorder / 缺 pickup 干扰项，也过滤掉不自洽的裸原子。
  ③ 选择层→ 组合覆盖：用 combine_sequential 从 sub_skills **重算真实效果签名**
              （lookalike 把加热换成制冷 → 真签名变 isCold；missing 少一个 token），
              对「从指令解析出的目标」贪心 set-cover，选最小精确覆盖集。

公平性：目标只从 instruction 关键词解析（和 B0~B3 同样只吃指令），
**不读** is_distractor/why_wrong/variant_type 标签，也**不用** GT required_effects。

单跑自检（含目标解析准确率诊断）：
  python router/code/domain_m1.py --n 20
"""
import os
import re
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (load_pool, load_queries, skill_sig, parse_goal,  # noqa: E402
                    flatten_atomic, true_transform_sig, places_at_target, true_effect_sig)
from retrievers import DenseRetriever                  # noqa: E402

SKILLS_CODE = os.path.join(HERE, os.pardir, os.pardir, "skills", "code")
sys.path.insert(0, SKILLS_CODE)
from skill_effects import goal_signature, combine_sequential  # noqa: E402

# ---- 子技能类别（用于手部状态模拟 + 效果推导） ----
_HOLD_OPS = {"heat_object_microwave", "cool_object_fridge",
             "clean_object_sink", "fill_with_liquid", "empty_liquid"}
_PASSIVE = {"open_receptacle", "close_receptacle", "toggle_on", "toggle_off",
            "goto_object"}


def is_legal(card, by_id):
    """手部状态符号模拟：从空手开始逐步检查前置条件是否满足。
    catches: reorder（先放后拿）、缺 pickup、裸的 put/heat/slice 等不自洽卡。
    （效果重算 transform_sig/places_at_target/flatten 已抽到 common，单一真源。）"""
    hand = None  # None=空手；否则为所持物名
    for sid, args in flatten_atomic(card, by_id):
        a0 = args[0] if args else None
        if sid == "pickup_object":
            if hand is not None:            # 手非空却要再拿
                return False
            hand = a0
        elif sid == "put_object":
            if hand is None or hand != a0:  # 没握住要放的东西
                return False
            hand = None
        elif sid in _HOLD_OPS:
            if hand is None or hand != a0:  # 加热/制冷/清洗/装液前必须正握该物
                return False
        elif sid == "slice_object":
            if hand is None or "knife" not in str(hand).lower():  # 切需握刀
                return False
        elif sid in _PASSIVE:
            pass
        # 未知技能：宽松放过（不误杀）
    return True


class M1RuleDomain:
    """规则型 M1：Dense top-K → 合法性过滤 → 目标 set-cover（无 LLM）。

    覆盖判定：变换类效果从 sub_skills 重算；at_receptacle 按任务容器逐题判定。"""
    name = "M1-Rule"
    wants_binding = True   # run_pilot 会把 query 的 binding 传进 select

    def __init__(self, pool, k=20, retriever=None, goal_fn=None, name=None,
                 l4_scorer=None, use_retrieval=False,
                 no_legality=False, no_repair=False, use_declared=False):
        self.pool = pool
        self.by_id = {c["skill_id"]: c for c in pool}
        # WP5 轴5 反过拟合消融：去掉符号脚手架，考察纯神经 L4 是否过拟合/过选。
        #   no_legality=True → 关 L2 合法性硬掩码（reorder/缺pickup/裸原子也进候选）；
        #   no_repair=True   → 关"覆盖优先 set-cover + 兜底"，改为纯 L4 保留分>0 选集。
        self.no_legality = no_legality
        self.no_repair = no_repair
        # 抗混淆逐层归因消融（WP-attr）：关掉 L4 的"效果重算落地"，改用技能卡**声明** effects
        #   做 set-cover（干扰项谎报 effects → 应被骗）。用于证明"效果重算"是抗混淆来源。
        self.use_declared = use_declared
        # 删 L1（默认）：不检索，全池 L2 合法候选直接进 set-cover（真实 10¹–10² 小库检索非必要，
        #   K 截断反而丢正确技能；见 stage2_execution_plan「零训练探针·L1 定案」）。
        #   use_retrieval=True 仅供 ablation / 库放大到 10⁴+ 时复用检索。
        self.use_retrieval = use_retrieval
        self.k = k if use_retrieval else len(pool)
        self.retriever = (retriever or DenseRetriever(pool)) if use_retrieval else None
        # L0 目标解析器：默认规则 parse_goal；正式版可传 LLMGoalParser（同签名）
        self.goal_fn = goal_fn or parse_goal
        # L4 判别式保留打分器（可选）：给定 (goal, 合法候选) 返回每候选保留分，
        # 仅在"覆盖优先"之后作为选择键；主系统 =None 走纯规则 set-cover。
        self.l4_scorer = l4_scorer
        if name:
            self.name = name
        # query 无关的量可缓存
        self._tsig = {c["skill_id"]: true_transform_sig(c, self.by_id) for c in pool}
        self._legal = {c["skill_id"]: is_legal(c, self.by_id) for c in pool}

    def eff_sig(self, card, binding):
        """给定任务 binding 的有效覆盖集：变换签名 + (命中任务容器则 at_receptacle)。
        use_declared=True（消融）时改用技能卡**声明** effects（不重算、不落地）——
        干扰项谎报的声明值会蒙混过关，用于隔离"效果重算落地"对抗混淆的贡献。"""
        if self.use_declared:
            return set(skill_sig(card))
        sig = set(self._tsig[card["skill_id"]])
        if places_at_target(card, self.by_id, binding):
            sig.add("at_receptacle")
        return sig

    def _cover(self, cands, goal, scores=None):
        """贪心 set-cover：每轮选(最大新覆盖 → [L4 学习保留分高] → 最小多余效果 → 名次靠前)。
        scores：可选 {skill_id: float} 学习分；**始终以覆盖优先(结构下限不变)**，学习分只在
        "同等新覆盖"的候选间做取舍——正是 lookalike/变体消歧发生的地方。"""
        remaining = set(goal)
        chosen = []
        pool = list(cands)  # (rank, card, sig)
        while remaining:
            best, best_key = None, None
            for rank, card, sig in pool:
                new = len(sig & remaining)
                if new == 0:
                    continue
                extra = len(sig - goal)          # 越少越好：不做没要求的变换
                sc = -scores.get(card["skill_id"], 0.0) if scores else 0.0  # 学习分越高越先
                key = (-new, sc, extra, rank)     # 覆盖优先 → 学习分 → 少多余 → 靠前
                if best_key is None or key < best_key:
                    best, best_key = (card, sig), key
            if best is None:
                break                             # 无人能再覆盖 → 停
            card, sig = best
            chosen.append(card)
            remaining -= sig
            pool = [(r, c, s) for (r, c, s) in pool if c["skill_id"] != card["skill_id"]]
        return chosen

    def legal_candidates(self, query, binding=None):
        """[删 L1] 全池按原始序 → L2 合法过滤；(use_retrieval=True 时才 L1 检索 top-K)。
        返回 [(rank, card, eff_sig)]。供 L4 训练/推理共用。"""
        cand = self.retriever.rank(query)[:self.k] if self.use_retrieval else self.pool
        return [(i, c, self.eff_sig(c, binding))
                for i, c in enumerate(cand)
                if self.no_legality or self._legal[c["skill_id"]]]

    def select(self, query, binding=None):
        goal = self.goal_fn(query, binding)
        legal = self.legal_candidates(query, binding)
        if not goal:                              # 解析不到目标 → 退化为 top-1 合法
            return [legal[0][1]] if legal else []
        scores = None
        if self.l4_scorer is not None and legal:
            scores = self.l4_scorer.score(goal, legal)
        # 轴5 纯神经消融：去覆盖/去兜底，只留 L4 保留分>0 的候选（考察过选/过拟合）。
        if self.no_repair and scores is not None:
            chosen = [c for (_, c, _) in legal if scores.get(c["skill_id"], -1e9) > 0.0]
            if not chosen and legal:              # 全为负也需非空输出 → 取分最高一张
                best = max(legal, key=lambda t: scores.get(t[1]["skill_id"], -1e9))
                chosen = [best[1]]
            return chosen
        chosen = self._cover(legal, goal, scores)
        if not chosen and legal:                  # 兜底：没覆盖上也给个最靠前的合法卡
            chosen = [legal[0][1]]
        return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()
    pool = load_pool()
    queries = load_queries(n=args.n, seed=0)
    m1 = M1RuleDomain(pool, k=args.k)
    n_legal = sum(m1._legal.values())
    print(f"技能池 {len(pool)} 条（合法 {n_legal} / 非法 {len(pool)-n_legal}）；"
          f"M1 检索 top{args.k} → 合法过滤 → 目标 set-cover\n")

    cov_ok = goal_ok = 0
    for q in queries:
        t = time.time()
        chosen = m1.select(q["instruction"], q.get("binding"))
        dt = time.time() - t
        req = set(q["required_effects"])
        parsed = parse_goal(q["instruction"], q.get("binding"))
        union = set()
        for c in chosen:
            union |= m1.eff_sig(c, q.get("binding"))
        cov = union >= req
        gok = parsed == req
        cov_ok += cov
        goal_ok += gok
        print(f"[{dt*1000:5.0f}ms] 解析={sorted(parsed)} {'✓' if gok else '≠GT '+str(sorted(req))}")
        print(f"          -> {[c['skill_id'] for c in chosen]} 覆盖GT={'✓' if cov else '✗'}")
    n = len(queries)
    print(f"\n目标解析==GT：{goal_ok}/{n}={goal_ok/n:.0%}；"
          f"选中集合覆盖GT：{cov_ok}/{n}={cov_ok/n:.0%}")


if __name__ == "__main__":
    main()
