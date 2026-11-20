"""Additional checks on the four-study analysis. Usage: python3 additional_checks.py REPO_ROOT"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = sys.argv[1]
spec = importlib.util.spec_from_file_location(
    "gr", os.path.join(ROOT, "Gating rule and refusal sensitivity", "gating_and_refusals.py"))
gr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gr)
gr.ROOT = ROOT
KEEP = ["1A", "1B", "2A", "2B"]
b = gr.load_binary()
b = b[b.study.isin(KEEP)]
a = gr.load_ages()
a = a[a.study.isin(KEEP)]
rng = np.random.default_rng(11)

print("== (a) 2B minus 2A, audio, paired by student, speaker bootstrap")
for q in ["q1_over_13", "q2_over_18"]:
    for model in ["Gemini", "GPT-4o / gpt-audio", "Nemotron"]:
        g = b[(b.modality == "audio") & (b.model == model) & (b.question == q) & b.study.isin(["2A", "2B"])
              & b.code.isin(["yes", "no"])].copy()
        g["y"] = (g.code == "yes").astype(int)
        per = g.groupby(["child", "study"]).y.agg(["sum", "size"]).unstack(fill_value=0)
        kids = per.index.to_numpy()
        s2a, n2a = per[("sum", "2A")].to_numpy(), per[("size", "2A")].to_numpy()
        s2b, n2b = per[("sum", "2B")].to_numpy(), per[("size", "2B")].to_numpy()
        obs = s2b.sum() / n2b.sum() - s2a.sum() / n2a.sum()
        d = []
        for _ in range(10000):
            i = rng.integers(0, len(kids), len(kids))
            if n2a[i].sum() and n2b[i].sum():
                d.append(s2b[i].sum() / n2b[i].sum() - s2a[i].sum() / n2a[i].sum())
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{q} {model}: diff {100*obs:.1f} points, 95% CI [{100*lo:.1f}, {100*hi:.1f}]")

print("\n== (b) share of numeric estimates at 13 or more, and at 18 or more")
for (study, model, modality), g in a[a.study.isin(["2A", "2B"])].groupby(["study", "model", "modality"]):
    print(f"{study} {model} {modality}: n={len(g)} >=13 {100*(g.pred_age>=13).mean():.1f}%  >=18 {100*(g.pred_age>=18).mean():.1f}%  "
          f"median {g.pred_age.median():.1f}  share within 8-11 {100*g.pred_age.between(8,11).mean():.1f}%")

print("\n== (c) contradictions against independence within cell")
keys = ["study", "child", "model", "modality", "acoustic", "repeat"]
q1 = b[b.question == "q1_over_13"][keys + ["code"]].rename(columns={"code": "q1"})
q2 = b[b.question == "q2_over_18"][keys + ["code"]].rename(columns={"code": "q2"})
p = q1.merge(q2, on=keys)
p = p[p.q1.isin(["yes", "no"]) & p.q2.isin(["yes", "no"])]
exp_total, obs_total = 0.0, 0
rows = []
for (study, model, modality), g in p.groupby(["study", "model", "modality"]):
    p13, p18 = (g.q1 == "yes").mean(), (g.q2 == "yes").mean()
    e = len(g) * (1 - p13) * p18
    o = int(((g.q1 == "no") & (g.q2 == "yes")).sum())
    yy = int(((g.q1 == "yes") & (g.q2 == "yes")).sum())
    exp_total += e
    obs_total += o
    rows.append((study, model, modality, len(g), o, round(e, 1), yy, int((g.q2 == "yes").sum())))
print(pd.DataFrame(rows, columns=["study", "model", "modality", "pairs", "obs_no13_yes18", "expected_indep", "yes_both", "yes18"]).to_string(index=False))
print(f"observed {obs_total}, expected under independence {exp_total:.1f}")
y18 = int((p.q2 == "yes").sum())
print(f"of {y18} yes answers to over 18, {obs_total} came with a no to over 13 = {100*obs_total/y18:.1f}%")

print("\n== (d) slope CIs, Kids VA, speaker bootstrap")
truth = {}
for study in ["1A", "1B"]:
    path, idcol, _ = gr.RAW[study]
    d = pd.read_csv(os.path.join(ROOT, path))
    truth.update({(study, c): age for c, age in zip(d[idcol], d.actual_age)})
for study in ["1A", "1B"]:
    for model in ["Gemini", "Nemotron"]:
        g = a[(a.study == study) & (a.model == model)].copy()
        g["true"] = [truth[(study, c)] for c in g.child]
        per = g.groupby("child").agg(pred=("pred_age", "mean"), true=("true", "first"))
        slope = np.polyfit(per.true, per.pred, 1)[0]
        bs = []
        for _ in range(10000):
            s = per.iloc[rng.integers(0, len(per), len(per))]
            if s.true.nunique() > 1:
                bs.append(np.polyfit(s.true, s.pred, 1)[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{study} {model}: slope {slope:.2f} [{lo:.2f}, {hi:.2f}]  mean estimate {g.pred_age.mean():.3f}")
