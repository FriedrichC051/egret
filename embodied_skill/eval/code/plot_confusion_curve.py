"""WP8 王牌图：混淆度曲线 FullCov vs 干扰密度。读 confusion_curve.json 出 PNG。

主图口径（2026-07 叙事同步）：**只画旗舰 M1-Neuro + baselines + gold 天花板**，
不再画 M1-Rule/M1-Head（它们是消融变体，归表 5b）。既然 M1 家族只剩一条线，
不再需要断轴——用单条连续 y 轴（0~1），直观且诚实。
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "eval", "datasets")

d = json.load(open(os.path.join(EVAL_DATA, "confusion_curve.json"), encoding="utf-8"))
res = d["results"]
x = [1, 3, 6]
tiers = ["low", "mid", "high"]

# 主图只呈现旗舰 M1-Neuro（m1h4）+ baselines；M1-Rule(m1)/M1-Head(m1h) 归消融表 5b，不入主图。
style = {
    "m1h4":  ("ESCOVER", "o-",  "#c0392b", 2.8, 8),
    "b0":    ("B0-LLM-Full",     "s-",  "#2c3e50", 2.0, 7),
    "b3":    ("B3-SkillFlow",    "^-",  "#e67e22", 2.0, 7),
    "dense": ("B2-Dense",        "d--", "#7f8c8d", 1.6, 6),
    "b4":    ("B4-LoTa (local)", "P-",  "#8e44ad", 2.0, 7),
    "bm25":  ("B1-BM25",         "v:",  "#95a5a6", 1.4, 6),
}

fig, ax = plt.subplots(figsize=(6.4, 4.6))

for m, (label, fmt, color, lw, ms) in style.items():
    if m not in res:
        continue
    y = [res[m][t] for t in tiers]
    ax.plot(x, y, fmt, color=color, linewidth=lw, markersize=ms, label=label)

ax.set_ylim(0.0, 1.0)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True, alpha=0.3)

# 紧凑图例放进中间空白带（B2-Dense ~0.44 与 B3 ~0.75 之间），不挡折线；无标题
ax.legend(loc="center right", bbox_to_anchor=(0.985, 0.585),
          fontsize=7.5, framealpha=0.9, handlelength=3.2,
          labelspacing=0.3, borderpad=0.5, handletextpad=0.6)

ax.set_xticks(x)
ax.set_xticklabels(["low (×1)", "mid (×3)", "high (×6)"])
ax.set_xlabel("Confusion density (distractors per gold skill)")
ax.set_ylabel("FullCov (effect-grounded)")

plt.tight_layout()
out_png = os.path.join(EVAL_DATA, "confusion_curve.png")
out_pdf = os.path.join(EVAL_DATA, "confusion_curve.pdf")
plt.savefig(out_png, dpi=150)
plt.savefig(out_pdf)  # 矢量 PDF，直接进 LaTeX \includegraphics
print("saved", out_png)
print("saved", out_pdf)
