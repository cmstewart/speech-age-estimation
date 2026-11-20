"""Figure 2 of the four-study analysis, drawn as a grouped dumbbell chart.

Audio against transcript on the MyST studies. Rows are grouped by model.

Usage: python3 figure2.py REPO_ROOT OUT_STEM
"""
import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT, STEM = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location(
    "gr", os.path.join(ROOT, "Gating rule and refusal sensitivity", "gating_and_refusals.py"))
gr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gr)
gr.ROOT = ROOT
np.random.seed(20261011)
b = gr.load_binary()
b = b[b.study.isin(["2A", "2B"])]

MODELS = [("Gemini", "Gemini"), ("GPT-4o / gpt-audio", "GPT-4o"), ("Nemotron", "Nemotron")]
STUDIES = [("2A", "30-second clips"), ("2B", "6-second clips")]
STYLE = {"audio": dict(color="#2a78d6", marker="o", label="Audio"),
         "text": dict(color="#eb6834", marker="s", label="Transcript")}
INK, INK2, GRID, LINK = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b3"
REF = {"q1_over_13": 5, "q2_over_18": 1}
TITLE = {"q1_over_13": "Placed above 13", "q2_over_18": "Placed above 18"}

res = {}
for m, lab in MODELS:
    for s, _ in STUDIES:
        for q in REF:
            for mo in ["audio", "text"]:
                g = b[(b.model == m) & (b.study == s) & (b.question == q) & (b.modality == mo)]
                v = g[g.code.isin(["yes", "no"])]
                y = (v.code == "yes").to_numpy(int)
                lo, hi = gr.boot_ci(y, v.child.to_numpy())
                res[(m, s, q, mo)] = (100 * y.mean(), lo, hi, len(v))

# Row positions. Each model gets a header slot and two rows.
rows, headers, ypos = [], [], 0.0
for m, lab in MODELS:
    headers.append((ypos, lab))
    ypos -= 0.8
    for s, slab in STUDIES:
        rows.append((ypos, m, s, slab))
        ypos -= 1.0
    ypos -= 0.5

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.5), sharey=True)
for ax, q in zip(axes, REF):
    for (yi, m, s, slab) in rows:
        pa = res[(m, s, q, "audio")]
        pt = res[(m, s, q, "text")]
        ax.plot([pa[0], pt[0]], [yi, yi], color=LINK, lw=1.0, zorder=1)
        for mo, p in [("audio", pa), ("text", pt)]:
            st = STYLE[mo]
            ax.plot([p[1], p[2]], [yi, yi], color=st["color"], lw=3.2, alpha=0.30,
                    solid_capstyle="butt", zorder=2)
            ax.scatter([p[0]], [yi], s=36, color=st["color"], marker=st["marker"],
                       edgecolor="white", linewidth=1.0, zorder=3)
    ax.axvline(REF[q], color=INK2, ls=(0, (4, 3)), lw=0.9, zorder=0)
    ax.text(REF[q] + 1.5, ypos + 0.55, f"{REF[q]}% reference point", fontsize=7.5, color=INK2, va="center")
    ax.set_xlim(-2, 85)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel("Percent of valid responses", fontsize=8.5, color=INK)
    ax.set_title(TITLE[q], fontsize=9.5, color=INK, loc="left")
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(ypos + 0.2, 0.5)

axes[0].set_yticks([r[0] for r in rows])
axes[0].set_yticklabels([r[3] for r in rows], color=INK2, fontsize=8.5)
for (yh, lab) in headers:
    axes[0].text(-0.02, yh, lab, transform=axes[0].get_yaxis_transform(), ha="right", va="center",
                 fontsize=9.5, color=INK, fontweight="bold")

# Direct labels on the first row of the first panel replace the legend.
y0, m0, s0, _ = rows[0]
for mo in ["audio", "text"]:
    p = res[(m0, s0, "q1_over_13", mo)]
    axes[0].text(p[0], y0 + 0.42, STYLE[mo]["label"], color=STYLE[mo]["color"], fontsize=8.5,
                 ha="center", va="bottom", fontweight="bold",
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.2), zorder=4)

fig.tight_layout()
fig.savefig(STEM + ".png", dpi=300)
fig.savefig(STEM + ".pdf")
for k, v in res.items():
    print(k, [round(x, 1) for x in v[:3]], v[3])
