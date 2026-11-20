"""Figure 1 of the four-study analysis. Studies 1A and 1B only.

Usage: python3 figure1.py REPO_ROOT OUT_STEM
Correlation intervals are fixed at the published values so text and figure agree.
"""
import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT, STEM = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location(
    "gr", os.path.join(ROOT, "Gating rule and refusal sensitivity", "gating_and_refusals.py"))
gr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gr)
gr.ROOT = ROOT
a = gr.load_ages()

truth = {}
for study in ["1A", "1B"]:
    path, idcol, _ = gr.RAW[study]
    d = pd.read_csv(os.path.join(ROOT, path))
    truth.update({(study, c): age for c, age in zip(d[idcol], d.actual_age)})

CI = {("1A", "Gemini"): (0.31, 0.80), ("1B", "Gemini"): (0.32, 0.79),
      ("1A", "Nemotron"): (0.22, 0.75), ("1B", "Nemotron"): (-0.12, 0.44)}
COLOR = {"Gemini": "#2a78d6", "Nemotron": "#1baf7a"}
EDGE = {"Gemini": "#1d5ca8", "Nemotron": "#127a55"}
ROWLAB = {"Gemini": "Gemini 2.5 Flash", "Nemotron": "Nemotron 3 Nano Omni"}
COLLAB = {"1A": "Study 1A\nscripted, 6 s", "1B": "Study 1B\nspontaneous, 6 s"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
YMAX = 16

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.2), sharex=True, sharey=True)
for i, model in enumerate(["Gemini", "Nemotron"]):
    for j, study in enumerate(["1A", "1B"]):
        ax = axes[i, j]
        g = a[(a.study == study) & (a.model == model)].copy()
        g["true"] = [truth[(study, c)] for c in g.child]
        per = g.groupby("child").agg(pred=("pred_age", "mean"), true=("true", "first"))
        r = stats.pearsonr(per.true, per.pred).statistic
        slope, icpt = np.polyfit(per.true, per.pred, 1)
        xs = np.array([4.5, 10.5])
        ax.plot(xs, xs, ls=(0, (4, 3)), color=INK2, lw=1.1, zorder=1)
        ax.axhline(7, ls=(0, (1, 2)), color=INK2, lw=1.1, zorder=1)
        ax.plot(xs, icpt + slope * xs, color=COLOR[model], lw=2, zorder=2)
        inside = per[per.pred <= YMAX]
        ax.scatter(inside.true, inside.pred, s=26, color=COLOR[model], edgecolor="white", linewidth=1.2, zorder=3)
        above = per[per.pred > YMAX]
        if len(above):
            vals = ", ".join(f"{v:.1f}" for v in sorted(above.pred))
            ax.annotate(f"{len(above)} children above axis\nat {vals} years", xy=(10.3, 3.4),
                        ha="right", va="bottom", fontsize=7.5, color=INK2)
        lo, hi = CI[(study, model)]
        ax.text(4.7, YMAX - 0.4, f"r = {r:.2f} [{lo:.2f}, {hi:.2f}]\nslope = {slope:.2f}",
                va="top", ha="left", fontsize=8, color=INK)
        ax.set_xlim(4.5, 10.5)
        ax.set_ylim(2.5, YMAX)
        ax.set_xticks(range(5, 11))
        ax.set_yticks([4, 7, 10, 13])
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        if i == 0:
            ax.set_title(COLLAB[study], fontsize=9, color=INK)
        if j == 0:
            ax.set_ylabel(ROWLAB[model], fontsize=9, color=INK)
fig.supxlabel("Child’s true age in years", fontsize=9.5, color=INK, y=0.085)
fig.supylabel("Mean estimated age in years", fontsize=9.5, color=INK, x=0.015)
handles = [plt.Line2D([], [], color=INK, lw=2, label="model, fitted"),
           plt.Line2D([], [], color=INK2, lw=1.1, ls=(0, (4, 3)), label="perfect calibration"),
           plt.Line2D([], [], color=INK2, lw=1.1, ls=(0, (1, 2)), label="constant guess of 7 years")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.53, 0.0))
fig.tight_layout(rect=(0.02, 0.07, 1, 1))
fig.savefig(STEM + ".png", dpi=300)
fig.savefig(STEM + ".pdf")
print("saved", STEM)
