"""Recompute the reported numbers on the four studies 1A, 1B, 2A and 2B.

Reuses the coding rule and the Study 2B de-duplication from the
repository's gating_and_refusals.py. That module also swaps in the text-first
gpt-audio rerun for Studies 1A and 1B. Usage: python3 four_study_numbers.py REPO_ROOT OUT_JSON
"""
import importlib.util
import json
import os
import re
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

OUT = {}
KEEP = ["1A", "1B", "2A", "2B"]

b_all = gr.load_binary()
a_all = gr.load_ages()
b = b_all[b_all.study.isin(KEEP)].copy()
a = a_all[a_all.study.isin(KEEP)].copy()
print("binary responses, all five studies", len(b_all), "four studies", len(b))
OUT["binary_responses_4"] = len(b)

# Contradictory pairs. No to over 13 and yes to over 18 on the same clip and repeat.
keys = ["study", "child", "model", "modality", "acoustic", "repeat"]


def contradictions(bb):
    q1 = bb[bb.question == "q1_over_13"][keys + ["code"]].rename(columns={"code": "q1"})
    q2 = bb[bb.question == "q2_over_18"][keys + ["code"]].rename(columns={"code": "q2"})
    p = q1.merge(q2, on=keys)
    p = p[p.q1.isin(["yes", "no"]) & p.q2.isin(["yes", "no"])]
    p["bad"] = (p.q1 == "no") & (p.q2 == "yes")
    cells = p.groupby(["study", "model", "modality"]).bad.agg(["sum", "size"])
    return len(p), int(p.bad.sum()), int((cells["sum"] > 0).sum()), len(cells), cells


n, k, c_pos, c_all, _ = contradictions(b_all)
print(f"all five studies: {k} of {n} pairs, {c_pos} of {c_all} cells")
n, k, c_pos, c_all, cells = contradictions(b)
print(f"four studies: {k} of {n} pairs = {100*k/n:.1f}%, {c_pos} of {c_all} cells")
print(cells.to_string())
OUT["contradictions"] = dict(pairs=n, bad=k, pct=round(100 * k / n, 1), cells_with=c_pos, cells=c_all)

# The coding-order check needs the raw response text. The repository holds coded
# outcomes only, so that check is not reproduced here.

# Response-level bootstrap CIs, audio, both questions.
np.random.seed(20261011)
rows = []
cells = list(b[b.modality == "audio"].groupby(["study", "model", "modality", "question"]))
# The text-first gpt-audio cells draw last, so the other cells keep their draws.
for late in (False, True):
    for (study, model, modality, q), g in cells:
        v = g[g.code.isin(["yes", "no"])]
        if not len(v) or gr.is_rerun_cell(study, model) != late:
            continue
        lo, hi = gr.boot_ci((v.code == "yes").to_numpy(int), v.child.to_numpy())
        rows.append(dict(study=study, model=model, q=q, n=len(v), rate=round(100 * (v.code == "yes").mean(), 1),
                         lo=round(lo, 1), hi=round(hi, 1)))
ci = pd.DataFrame(rows).sort_values(["study", "model", "q"]).reset_index(drop=True)
print(ci.to_string(index=False))

# Reference-point counts. The text-first gpt-audio cells of Studies 1A and 1B
# have 9 to 19 valid answers each. They are too small to count for or against a
# reference point, so they stay out of every count and range here. Their
# per-cell rows are kept in the table above and in per_child_rates.csv.
counted = ci[[not gr.is_rerun_cell(s, m) for s, m in zip(ci.study, ci.model)]]
o13 = counted[counted.q == "q1_over_13"]
o18 = counted[counted.q == "q2_over_18"]
OUT["o13_cells"] = len(o13)
OUT["o13_meet"] = int((o13.rate <= 5).sum())
OUT["o13_over3x"] = int((o13.rate > 15).sum())
OUT["o18_exceed"] = int((o18.rate > 1).sum())
pc = gr.per_child_rates(b)
pca = pc[(pc.modality == "audio") & np.array([not gr.is_rerun_cell(s, m) for s, m in zip(pc.study, pc.model)])]
OUT["o13_child_maj_exceed"] = int((pca[pca.question == "over 13"].children_majority > 5).sum())
OUT["o18_child_maj_exceed"] = int((pca[pca.question == "over 18"].children_majority > 1).sum())
excluded = ci[[gr.is_rerun_cell(s, m) for s, m in zip(ci.study, ci.model)]]
print(f"reference-point counts and the over-13 range exclude {len(excluded)} text-first gpt-audio cells "
      f"of Studies 1A and 1B, valid n {sorted(excluded.n.tolist())}. Their rows stay in the table above.")
print({k: OUT[k] for k in OUT if k.startswith("o1")})
print("range over 13 audio", o13.rate.min(), o13.rate.max())

# Refusal counts by cell.
rc = b.groupby(["study", "model", "modality", "question"]).code.value_counts().unstack(fill_value=0)
print(rc.to_string())

# Numeric estimates. Kids VA truth from the raw files.
truth = {}
for study in ["1A", "1B"]:
    path, idcol, _ = gr.RAW[study]
    d = pd.read_csv(os.path.join(ROOT, path))
    truth.update({(study, c): age for c, age in zip(d[idcol], d.actual_age)})
OUT["kidsva_mean_age"] = round(pd.Series({c: v for (s, c), v in truth.items() if s == "1A"}).mean(), 2)

print("\n== Kids VA calibration")
rng = np.random.default_rng(7)
for study in ["1A", "1B"]:
    for model in ["Gemini", "Nemotron"]:
        g = a[(a.study == study) & (a.model == model)].copy()
        g["true"] = [truth[(study, c)] for c in g.child]
        per = g.groupby("child").agg(pred=("pred_age", "mean"), true=("true", "first"))
        r = stats.pearsonr(per.true, per.pred).statistic
        slope = np.polyfit(per.true, per.pred, 1)[0]
        mae = (g.pred_age - g.true).abs().mean()
        mae7 = (7 - g.true).abs().mean()
        boots = []
        for _ in range(5000):
            idx = rng.integers(0, len(per), len(per))
            s = per.iloc[idx]
            if s.true.nunique() > 1 and s.pred.nunique() > 1:
                boots.append(stats.pearsonr(s.true, s.pred).statistic)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        perm = [stats.pearsonr(per.true, rng.permutation(per.pred.to_numpy())).statistic for _ in range(10000)]
        p = (np.sum(np.abs(perm) >= abs(r)) + 1) / (len(perm) + 1)
        above13 = (g.pred_age > 13).mean() * 100
        ge13 = (g.pred_age >= 13).mean() * 100
        span = (per.pred.min(), per.pred.max())
        print(f"{study} {model}: n={len(g)} children={len(per)} mean={g.pred_age.mean():.2f} MAE={mae:.2f} "
              f"MAE7={mae7:.2f} r={r:.2f} [{lo:.2f},{hi:.2f}] perm p={p:.4f} slope={slope:.2f} "
              f">13={above13:.1f}% >=13={ge13:.1f}% child-mean span {span[0]:.1f}-{span[1]:.1f}")
        OUT[f"cal_{study}_{model}"] = dict(n=len(g), mean=round(g.pred_age.mean(), 1), mae=round(mae, 1),
                                           mae7=round(mae7, 1), r=round(r, 2), lo=round(lo, 2), hi=round(hi, 2),
                                           p=round(p, 4), slope=round(slope, 2), above13=round(above13, 1),
                                           ge13=round(ge13, 1), span=[round(span[0], 1), round(span[1], 1)])

print("\n== Mean estimates, all cells")
means = a.groupby(["study", "model", "modality"]).pred_age.agg(["mean", "size"]).round(2)
print(means.to_string())
OUT["means"] = {f"{s}|{m}|{mo}": [round(r["mean"], 1), int(r["size"])] for (s, m, mo), r in means.iterrows()}

# Paired modality tests on per-student means.
print("\n== Modality tests")
for study in ["2A", "2B"]:
    for model in ["Gemini", "GPT-4o / gpt-audio", "Nemotron"]:
        g = a[(a.study == study) & (a.model == model)]
        per = g.groupby(["child", "modality"]).pred_age.mean().unstack()
        per = per.dropna()
        diff = per.audio - per.text
        t = stats.ttest_rel(per.audio, per.text)
        d = diff.mean() / diff.std(ddof=1)
        print(f"{study} {model}: n={len(per)} mean diff={diff.mean():.2f} t({len(per)-1})={t.statistic:.2f} "
              f"p={t.pvalue:.3g} d={d:.2f}")
        OUT[f"mod_{study}_{model}"] = dict(n=len(per), diff=round(diff.mean(), 1), t=round(t.statistic, 2),
                                           df=len(per) - 1, p=float(t.pvalue), d=round(d, 2))

# Text-condition threshold rates.
rs = gr.refusal_sensitivity(b)
print("\n== Text rates")
print(rs[rs.modality == "text"].to_string(index=False))

# First collection phase, Kids VA scripted. The repository holds coded outcomes.
fp = pd.read_csv(os.path.join(ROOT, "First collection phase (superseded, google-genai 0.4.0)", "study1a_raw_results.csv"))
print("\n== First phase")
for model in ["gemini", "gpt4o", "nemotron"]:
    for q in ["q1_over_13", "q2_over_18", "q3_age"]:
        g = fp[(fp.model == model) & (fp.question == q)]
        print(model, q, "n", len(g),
              "refused by outcome", round(100 * (g.outcome == "refusal").mean(), 1),
              "refused by refusal_check", round(100 * (g.refusal_check == "refused").mean(), 1))
gq = fp[(fp.model == "gemini") & (fp.question == "q3_age")]
v = gq.dropna(subset=["pred_age"])
print("corrupted Gemini: valid", len(v), "of", len(gq), "mean", round(v.pred_age.mean(), 1),
      "MAE", round((v.pred_age - v.actual_age).abs().mean(), 1), "true mean", round(gq.actual_age.mean(), 2))
OUT["corrupted"] = dict(valid=len(v), n=len(gq), mean=round(v.pred_age.mean(), 1),
                        mae=round((v.pred_age - v.actual_age).abs().mean(), 1))

# gpt-audio open-age responses on 1A and 1B, text-first rerun.
for study in ["1A", "1B"]:
    d = gr.load_rerun(study)
    g = d[d.question == "q3_age"]
    nonref = g.refusal_check != "refused"
    print(study, "gpt-audio q3 attempts", len(g), "non-refusal", round(100 * nonref.mean(), 1),
          "with an age", int(g.pred_age.notna().sum()))

json.dump(OUT, open(sys.argv[2], "w"), indent=1, default=str)
