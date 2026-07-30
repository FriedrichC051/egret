"""抗混淆逐层归因（WP-attr）：在混淆度曲线上逐个敲除 M1 的 L1 物理合法过滤 / L2 效果重算，
证明"强抗混淆"到底来自哪一层，并把'哪层挡哪族干扰'钉死。
（编号已按正式文档连续化为 L0–L3；代码内部 dict 键 m1_nol2/no_legality 沿用旧名不改。）

变体（L0 一律用规则 parse_goal，保持恒定 → 只隔离 L1 合法过滤与 L2 效果重算）：
  m1           全层（对照，应平）
  m1_nol2      关 L1 物理合法过滤（no_legality）——图例标注 "M1 -L1"
  m1_egoff     关 L2 效果重算，改用声明 effects（use_declared）——图例标注 "M1 -EG"
  m1_nol2_egoff 两者都关
  dense        纯相似度端点（参照）

产出（写到独立文件，不覆盖 confusion_curve.json / .png）：
  eval/datasets/confusion_ablation.json     曲线 + 逐族泄漏
  eval/datasets/confusion_ablation_a.{png,pdf}  子图 a：覆盖率 vs 干扰密度
  eval/datasets/confusion_ablation_b.{png,pdf}  子图 b：选中撒谎技能数 vs 干扰密度

用法（Windows, skill-router 环境）：
  python eval/code/_confusion_ablation.py            # 重算 + 画图
  python eval/code/_confusion_ablation.py --plot-only  # 只读已有 JSON 重画（不重算）
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ROUTER_CODE = os.path.join(ROOT, "router", "code")
SKILLS_DATA = os.path.join(ROOT, "skills", "datasets")
GT_DATA = os.path.join(ROOT, "gt", "datasets")
EVAL_DATA = os.path.join(ROOT, "eval", "datasets")
sys.path.insert(0, ROUTER_CODE)
sys.path.insert(0, HERE)

DENSITY = {"low": 1, "mid": 3, "high": 6}
TIERS = ["low", "mid", "high"]


def make_variants(pool):
    """返回 {name: (obj, kind)}；kind='sel' 走 select，'ret' 走 rank@1。"""
    from domain_m1 import M1RuleDomain
    from retrievers import DenseRetriever
    return {
        "m1":            (M1RuleDomain(pool, name="M1 (full)"), "sel"),
        "m1_nol2":       (M1RuleDomain(pool, no_legality=True, name="M1 -L1"), "sel"),
        "m1_egoff":      (M1RuleDomain(pool, use_declared=True, name="M1 -EG"), "sel"),
        "m1_nol2_egoff": (M1RuleDomain(pool, no_legality=True, use_declared=True,
                                       name="M1 -L1 -EG"), "sel"),
        "dense":         (DenseRetriever(pool), "ret"),
    }


def run():
    import common  # 重依赖仅在实算时导入，--plot-only 不需要
    import metrics as M
    from run_confusion_curve import load_answerable
    manifest = json.load(open(os.path.join(SKILLS_DATA, "tiers_manifest.json"),
                              encoding="utf-8"))
    subset_ids = manifest["subset_base_ids"]
    queries = common.load_queries(n=None, seed=0,
                                  gt_path=os.path.join(GT_DATA, "gt_kitchen_eval.jsonl"))
    os.environ.pop("POOL_FILE", None)
    full_by_id = {c["skill_id"]: c for c in common.load_pool()}
    answerable = load_answerable(subset_ids, queries, full_by_id)
    print(f"子集基准 {len(subset_ids)}；可答子集 {len(answerable)} 题（三档共用）\n")

    fullcov = {}   # variant -> {tier: sel_fullcov}
    leak = {}      # variant -> {tier: {reorder,missing,lookalike}}  选中干扰项计数
    for tier in TIERS:
        os.environ["POOL_FILE"] = f"pool_tier_{tier}.json"
        pool = common.load_pool()
        by_id = {c["skill_id"]: c for c in pool}
        ndist = sum(1 for c in pool if c.get("is_distractor"))
        print(f"==== tier {tier}（密度×{DENSITY[tier]}）：池 {len(pool)}（干扰 {ndist}）====")
        for name, (obj, kind) in make_variants(pool).items():
            t = time.time()
            hit = 0.0
            # overclaim=选中"声明⊋真实"的撒谎技能数（含未标注 is_distractor 的换容器混淆项，
            #   如 *_discard_in_garbage：声明 at_receptacle 实则扔垃圾桶）——效果重算才挡得住；
            # fam=选中"已标注干扰项"的族分布（reorder/missing/lookalike）。
            over = 0
            fam = {"reorder": 0, "missing": 0, "lookalike": 0}
            for q in answerable:
                req = q["required_effects"]
                binding = q.get("binding")
                if kind == "ret":
                    ranked = obj.rank(q["instruction"])
                    hit += M.fullcov_at_k(ranked, req, 1, by_id, binding)
                    chosen = ranked[:1]
                else:
                    chosen = obj.select(q["instruction"], binding)
                    hit += M.selection_scores(chosen, req, by_id, binding)["sel_fullcov"]
                for c in chosen:
                    declared = set(common.skill_sig(c))
                    true_s = common.true_effect_sig(c, by_id, binding)
                    if declared - true_s:            # 声明了做不到的效果 = 撒谎/过度声明
                        over += 1
                    vt = c.get("variant_type")
                    if c.get("is_distractor") and vt in fam:
                        fam[vt] += 1
            fullcov.setdefault(name, {})[tier] = round(hit / max(1, len(answerable)), 4)
            leak.setdefault(name, {})[tier] = {"overclaim": over, **fam}
            print(f"  {name:14} FullCov={fullcov[name][tier]:.3f}  "
                  f"选中撒谎技能={over}  已标注干扰 r/m/l="
                  f"{fam['reorder']}/{fam['missing']}/{fam['lookalike']}  "
                  f"({time.time()-t:.1f}s)")
        print()

    out = {"n_answerable": len(answerable), "density": DENSITY,
           "note": "L0=规则恒定；仅隔离 L2(no_legality) 与 L4效果重算(use_declared)",
           "fullcov": fullcov, "leakage": leak}
    op = os.path.join(EVAL_DATA, "confusion_ablation.json")
    json.dump(out, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"已写 {op}")

    # 汇总表
    print("\n===== 曲线（Sel-FullCov，effect-grounded 判定）=====")
    print(f"{'variant':14} {'low(1)':>8} {'mid(3)':>8} {'high(6)':>8} {'掉幅':>8}")
    for name, r in fullcov.items():
        drop = r["low"] - r["high"]
        print(f"{name:14} {r['low']:8.3f} {r['mid']:8.3f} {r['high']:8.3f} {drop:+8.3f}")
    print("\n===== 逐档：选中撒谎技能数（overclaim，越低越抗混淆）=====")
    print(f"{'variant':14} {'low':>6} {'mid':>6} {'high':>6}")
    for name, r in leak.items():
        print(f"{name:14} {r['low']['overclaim']:6d} {r['mid']['overclaim']:6d} "
              f"{r['high']['overclaim']:6d}")
    print("\n===== high 档：已标注干扰项族分布（reorder/missing/lookalike）=====")
    print(f"{'variant':14} {'reorder':>8} {'missing':>8} {'lookalike':>10}")
    for name, r in leak.items():
        h = r["high"]
        print(f"{name:14} {h['reorder']:8d} {h['missing']:8d} {h['lookalike']:10d}")
    return out


def plot(out):
    """两图（无图题），各出 PNG+PDF（论文引 PDF）。
    a：FullCoverage vs 干扰密度（y 收紧到 0.40--0.95，去空白）；
    b：选中撒谎技能数 vs 干扰密度（纵轴断轴 0--8 / 24--54，压掉中间空带）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [1, 3, 6]
    order = ["m1", "m1_nol2", "m1_egoff", "m1_nol2_egoff", "dense"]
    labels = {"m1": "ESCOVER (full)", "m1_nol2": "ESCOVER $-$L1",
              "m1_egoff": "ESCOVER $-$EG", "m1_nol2_egoff": "ESCOVER $-$L1 $-$EG",
              "dense": "Dense (similarity)"}
    styles = {"m1": ("-", "o"), "m1_nol2": ("--", "s"), "m1_egoff": ("--", "^"),
              "m1_nol2_egoff": (":", "d"), "dense": ("-.", "x")}
    colors = {"m1": "#c0392b", "m1_nol2": "#2c3e50", "m1_egoff": "#e67e22",
              "m1_nol2_egoff": "#8e44ad", "dense": "#7f8c8d"}
    FIGSIZE = (5.4, 3.8)

    # ---- a：FullCoverage vs 密度（单轴，y 收紧）----
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name in order:
        if name not in out["fullcov"]:
            continue
        ls, mk = styles[name]
        ax.plot(xs, [out["fullcov"][name][t] for t in TIERS], ls, marker=mk,
                color=colors[name], label=labels[name], linewidth=2, markersize=7)
    ax.set_xticks(xs)
    ax.set_xlabel("Distractor density (variants per gold skill)")
    ax.set_ylabel("FullCoverage (effect-grounded)")
    ax.set_ylim(0.40, 0.95)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.02, 0.15),
              framealpha=0.95)
    fig.subplots_adjust(left=0.15, right=0.96, top=0.96, bottom=0.15)
    for ext in ("png", "pdf"):
        op = os.path.join(EVAL_DATA, f"confusion_ablation_a.{ext}")
        fig.savefig(op, dpi=150)
        print(f"已写 {op}")
    plt.close(fig)

    # ---- b：选中撒谎技能数 vs 密度（断轴：底 0--8，顶 24--54）----
    fig, (axt, axb) = plt.subplots(
        2, 1, sharex=True, figsize=FIGSIZE,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.06})
    for name in order:
        if name not in out["leakage"]:
            continue
        ls, mk = styles[name]
        ys = [out["leakage"][name][t]["overclaim"] for t in TIERS]
        axt.plot(xs, ys, ls, marker=mk, color=colors[name], label=labels[name],
                 linewidth=2, markersize=7)
        axb.plot(xs, ys, ls, marker=mk, color=colors[name],
                 linewidth=2, markersize=7)
    axt.set_ylim(24, 54)
    axb.set_ylim(0, 8)
    axt.set_yticks([30, 40, 50])
    axb.set_yticks([0, 4, 8])
    axt.grid(True, alpha=0.3)
    axb.grid(True, alpha=0.3)
    axt.spines["bottom"].set_visible(False)
    axb.spines["top"].set_visible(False)
    axt.tick_params(labelbottom=False, bottom=False)
    dd = 0.5
    cut = dict(marker=[(-1, -dd), (1, dd)], markersize=8, linestyle="none",
               color="k", mec="k", mew=1, clip_on=False)
    axt.plot([0, 1], [0, 0], transform=axt.transAxes, **cut)
    axb.plot([0, 1], [1, 1], transform=axb.transAxes, **cut)
    axb.set_xticks(xs)
    axb.set_xlabel("Distractor density (variants per gold skill)")
    fig.supylabel("# over-declaring skills selected", fontsize=10)
    axt.legend(fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.55),
               framealpha=0.95)
    fig.subplots_adjust(left=0.15, right=0.96, top=0.96, bottom=0.15, hspace=0.06)
    for ext in ("png", "pdf"):
        op = os.path.join(EVAL_DATA, f"confusion_ablation_b.{ext}")
        fig.savefig(op, dpi=150)
        print(f"已写 {op}")
    plt.close(fig)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        op = os.path.join(EVAL_DATA, "confusion_ablation.json")
        out = json.load(open(op, encoding="utf-8"))
        print(f"已读 {op}（--plot-only，跳过重算）")
        plot(out)
    else:
        out = run()
        try:
            plot(out)
        except Exception as ex:
            print(f"[warn] 画图跳过：{ex}")
