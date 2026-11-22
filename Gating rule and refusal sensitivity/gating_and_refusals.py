"""Two analyses requested by reviewers.

Analysis 1, refusal sensitivity. Every threshold rate in the papers excludes
refusals, API errors and unclear responses from the denominator. This script
reports each rate three ways. The valid-only rate matches the published
tables. The fail-safe rate counts every excluded response as a child. The
fail-open rate counts every excluded response as an adult.

Analysis 2, the gating rule. The paper recommends asking
both a numeric and a yes-or-no question at runtime. This script scores that
rule on the existing logs. A response passes the child as an adult only when
the numeric estimate is 13 or more and the over-13 answer is yes, paired
within child, model, modality, acoustic condition and repeat.

Both analyses reuse the binary coding rule and the Study 2B deduplication from
regen_rates.py, so every denominator matches rates_regenerated.csv.

Every results file in this repository holds a coded outcome for each response
and not the response text. The codes were produced with the binary coding rule
in code_binary below, which is kept here as the record of that rule.

gpt-audio on Kids VA. The gpt-audio rows in the Study 1A and 1B folders come
from requests that placed the audio part before the text part. A control on
2026-11-22 showed that gpt-audio ignored the audio in that order. Those rows
stay in the study folders as the record of the failure. Every rate here uses
the text-first rerun in the folder named by RERUN_DIR in their place. Study 1C
was not rerun, so gpt-audio on Study 1C is dropped from every rate.
"""
import os
import re
import numpy as np
import pandas as pd

np.random.seed(20261011)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = HERE

S1A = "Study 1A - Kids VA Scripted Speech (google-genai 2.0.0)/Results/"
S1B = "Study 1B - Kids VA Spontaneous Speech (google-genai 2.0.0)/Results/"
S1C = "Study 1C - Kids VA Scripted Speech w- Background Noise Injection (google-genai 2.0.0)/"
S2A = "Study 2A - MyST 30-Second Speech (google-genai 2.0.0)/"
S2B = "Study 2B - MyST 6-Second Speech (google-genai 2.0.0)/"

RAW = {
    "1A": (S1A + "study1_raw_results.csv", "child", "outcome"),
    "1B": (S1B + "study1b_pinned_raw_results.csv", "child", "outcome"),
    "1C": (S1C + "study1c_raw_results.csv", "child", "outcome"),
    "2A": (S2A + "study2a_results.csv", "student_id", "outcome"),
    "2B": (S2B + "study2b_results_combined.csv", "student_id", "outcome"),
}
AGES = {
    "1A": (S1A + "study1_q3_parsed.csv", "child"),
    "1B": (S1B + "study1b_pinned_q3_parsed.csv", "child"),
    "1C": (S1C + "study1c_q3_parsed.csv", "child"),
    "2A": (S2A + "study2a_q3_parsed.csv", "student_id"),
    "2B": (S2B + "study2b_q3_parsed.csv", "student_id"),
}
RERUN_DIR = "gpt-audio control and rerun (2026-11-22)/"
RERUN = RERUN_DIR + "rerun_coded.csv"
RERUN_STUDIES = ("1A", "1B")
DROP_GPT_AUDIO = ("1C",)
MODEL_NAME = {"gemini": "Gemini", "gemini_200": "Gemini",
              "gpt4o": "GPT-4o / gpt-audio", "gpt_audio": "GPT-4o / gpt-audio",
              "nemotron": "Nemotron"}
REFUSAL = re.compile(r"sorry|can.t|cannot|unable|i.m not able|not able to", re.I)


def code_binary(text):
    """Binary coding rule that produced the outcome codes in the results files."""
    s = str(text).strip()
    if s.startswith("ERROR"):
        return "error"
    u = s.upper()
    if re.match(r"^(YES|NO)\b", u):
        return "yes" if u.startswith("YES") else "no"
    if REFUSAL.search(s):
        return "refusal"
    first = u.split("\n")[0]
    has_y, has_n = "YES" in first, "NO" in first
    if has_y and not has_n:
        return "yes"
    if has_n and not has_y:
        return "no"
    return "unclear"


def code_outcome(text):
    """Results files are outcome-coded. Map their labels onto one set."""
    s = str(text).strip().lower()
    if s in ("yes", "no", "error", "refusal", "unclear"):
        return s
    if s.startswith("err"):
        return "error"
    if s.startswith("refus"):
        return "refusal"
    return "unclear"


def add_keys(d, study, idcol):
    d = d.rename(columns={idcol: "child"})
    if "condition" in d.columns and set(d.condition.dropna().unique()) <= {"audio", "text"}:
        d["modality"] = d.condition
    else:
        d["modality"] = "audio"
    if study == "1C":
        d["acoustic"] = d.condition_label
    else:
        d["acoustic"] = "n/a"
    d["study"] = study
    d["model"] = d.model.map(lambda m: MODEL_NAME.get(m, m))
    return d


def dedup_2b(d):
    """Study 2B GPT-4o audio. Keep each student's first presentation only."""
    mask = ((d.study == "2B") & (d.model == "GPT-4o / gpt-audio") & (d.modality == "audio"))
    dup = d[mask].drop_duplicates(["child", "question", "repeat"], keep="first")
    return pd.concat([d[~mask], dup], ignore_index=True)


def load_rerun(study):
    """Text-first gpt-audio rows for one Kids VA study, coded with the same rules."""
    r = pd.read_csv(os.path.join(ROOT, RERUN))
    return r[r.study == study].drop(columns=["study"])


def use_rerun(d, study):
    """Swap the audio-first gpt-audio rows for the text-first rerun, or drop them."""
    if study in RERUN_STUDIES:
        keep = d[d.model != "gpt_audio"]
        return pd.concat([keep, load_rerun(study)[["child", "model", "question", "repeat",
                                                   "outcome", "pred_age"]]], ignore_index=True)
    if study in DROP_GPT_AUDIO:
        return d[d.model != "gpt_audio"].copy()
    return d


def is_rerun_cell(study, model):
    return study in RERUN_STUDIES and model == MODEL_NAME["gpt_audio"]


def load_binary():
    frames = []
    for study, (path, idcol, textcol) in RAW.items():
        d = pd.read_csv(os.path.join(ROOT, path))
        d = use_rerun(d, study)
        d = add_keys(d, study, idcol)
        d = d[d.question.isin(["q1_over_13", "q2_over_18"])].copy()
        coder = code_binary if textcol == "response" else code_outcome
        d["code"] = d[textcol].map(coder)
        frames.append(d[["study", "child", "model", "modality", "acoustic", "question", "repeat", "code"]])
    return dedup_2b(pd.concat(frames, ignore_index=True))


def load_ages():
    frames = []
    for study, (path, idcol) in AGES.items():
        d = pd.read_csv(os.path.join(ROOT, path))
        d = use_rerun(d, study)
        d = add_keys(d, study, idcol)
        d["question"] = "q3_age"
        frames.append(d[["study", "child", "model", "modality", "acoustic", "question", "repeat", "pred_age"]])
    a = dedup_2b(pd.concat(frames, ignore_index=True))
    return a.dropna(subset=["pred_age"])


def boot_ci(flags, children, n=10000):
    """Speaker-clustered percentile bootstrap on a 0/1 vector."""
    if not len(flags):
        return np.nan, np.nan
    per = pd.DataFrame({"y": flags, "c": children}).groupby("c").y.agg(["sum", "size"])
    y, t = per["sum"].to_numpy(float), per["size"].to_numpy(float)
    idx = np.random.randint(0, len(y), (n, len(y)))
    num, den = y[idx].sum(1), t[idx].sum(1)
    ok = den > 0
    return np.percentile(100 * num[ok] / den[ok], [2.5, 97.5])


def r1(x):
    return round(float(x), 1) if x == x else np.nan


def refusal_sensitivity(b):
    rows = []
    for (study, model, modality, question), g in b.groupby(["study", "model", "modality", "question"]):
        attempts = len(g)
        yes = int((g.code == "yes").sum())
        valid = int(g.code.isin(["yes", "no"]).sum())
        excluded = attempts - valid
        rows.append(dict(
            study=study, model=model, modality=modality,
            question="over 13" if question == "q1_over_13" else "over 18",
            attempts=attempts, valid_n=valid, excluded=excluded,
            rate_valid_only=r1(100 * yes / valid) if valid else np.nan,
            rate_fail_safe=r1(100 * yes / attempts),
            rate_fail_open=r1(100 * (yes + excluded) / attempts)))
    return pd.DataFrame(rows).sort_values(["question", "study", "model", "modality"])


def gating_rule(b, a):
    keys = ["study", "child", "model", "modality", "acoustic", "repeat"]
    q1 = b[b.question == "q1_over_13"][keys + ["code"]].rename(columns={"code": "q1"})
    q1_all = q1.copy()
    q1["q1_yes"] = q1.q1 == "yes"
    q1 = q1[q1.q1.isin(["yes", "no"])]
    a = a.copy()
    a["q3_adult"] = a.pred_age >= 13
    pairs = q1.merge(a[keys + ["q3_adult", "pred_age"]], on=keys, how="inner")
    pairs["and_adult"] = pairs.q1_yes & pairs.q3_adult
    pairs["or_adult"] = pairs.q1_yes | pairs.q3_adult
    rows = []
    groups = list(pairs.groupby(["study", "model", "modality"]))
    # Rerun cells draw their intervals after every other cell, so adding them
    # leaves the random draws of the unchanged cells as they were.
    cis = {}
    for late in (False, True):
        for (study, model, modality), g in groups:
            if is_rerun_cell(study, model) == late:
                cis[(study, model, modality)] = boot_ci(g.and_adult.to_numpy(int), g.child.to_numpy())
    for (study, model, modality), g in groups:
        attempts = int(((q1_all.study == study) & (q1_all.model == model) & (q1_all.modality == modality)).sum())
        per_child = g.groupby("child").and_adult.agg(["any", "mean"])
        lo, hi = cis[(study, model, modality)]
        rows.append(dict(
            study=study, model=model, modality=modality,
            attempts=attempts, pairs_with_both=len(g), children=g.child.nunique(),
            rate_binary_alone=r1(100 * g.q1_yes.mean()),
            rate_numeric_alone=r1(100 * g.q3_adult.mean()),
            rate_and_rule=r1(100 * g.and_adult.mean()),
            and_ci_lo=r1(lo), and_ci_hi=r1(hi),
            rate_or_rule=r1(100 * g.or_adult.mean()),
            rate_and_rule_fail_safe=r1(100 * g.and_adult.sum() / attempts),
            and_rule_children_at_least_once=r1(100 * per_child["any"].mean()),
            and_rule_children_majority=r1(100 * (per_child["mean"] > 0.5).mean()),
            binary_disagrees_with_numeric=r1(100 * (g.q1_yes != g.q3_adult).mean())))
    return pd.DataFrame(rows).sort_values(["study", "model", "modality"])


def per_child_rates(b):
    """At-least-once and majority per-child rates for every cell, deduplicated."""
    rows = []
    for (study, model, modality, question), g in b.groupby(["study", "model", "modality", "question"]):
        valid = g[g.code.isin(["yes", "no"])]
        if not len(valid):
            continue
        per = valid.groupby("child").code.agg(any_yes=lambda s: (s == "yes").any(),
                                              maj_yes=lambda s: (s == "yes").mean() > 0.5)
        rows.append(dict(
            study=study, model=model, modality=modality,
            question="over 13" if question == "q1_over_13" else "over 18",
            valid_n=len(valid), children=len(per),
            rate_responses=r1(100 * (valid.code == "yes").mean()),
            children_at_least_once=r1(100 * per.any_yes.mean()),
            children_majority=r1(100 * per.maj_yes.mean())))
    return pd.DataFrame(rows).sort_values(["question", "study", "model", "modality"])


SHORT = {"Gemini": "Gemini", "GPT-4o / gpt-audio": "GPT-4o", "Nemotron": "Nemotron"}
TABLE_STUDIES = ["1A", "1B", "1C", "2A", "2B"]
TABLE_MODELS = ["Gemini", "GPT-4o / gpt-audio", "Nemotron"]
TABLE_QS = [("q1_over_13", "over 13"), ("q2_over_18", "over 18")]


def f1(x):
    return "--" if x != x else f"{x:.1f}"


def table_rows(b, rs, gt, pc):
    """Every table row in the paper in LaTeX form.

    The threshold intervals are drawn from a fresh seed in pandas groupby order,
    with the rerun cells drawn last. The gating intervals are the ones in
    gating_rule.csv. A cell with no
    attempts is left out. That removes gpt-audio on Study 1C.
    """
    lines = ["%% threshold tables, audio, with CI and both per-child rules"]
    np.random.seed(20261011)
    ci = {}
    cells = list(b[b.modality == "audio"].groupby(["question", "study", "model"]))
    for late in (False, True):
        for (q, s, m), g in cells:
            v = g[g.code.isin(["yes", "no"])]
            if len(v) and is_rerun_cell(s, m) == late:
                ci[(q, s, m)] = boot_ci((v.code == "yes").to_numpy(int), v.child.to_numpy())
    for q, qlab in TABLE_QS:
        lines.append(f"%% {qlab}")
        for s in TABLE_STUDIES:
            for m in TABLE_MODELS:
                g = b[(b.study == s) & (b.model == m) & (b.modality == "audio") & (b.question == q)]
                if not len(g):
                    continue
                v = g[g.code.isin(["yes", "no"])]
                if not len(v):
                    lines.append(f"{s} & {SHORT[m]} & refused all & & 0 & & & \\\\")
                    continue
                lo, hi = ci[(q, s, m)]
                row = pc[(pc.study == s) & (pc.model == m) & (pc.modality == "audio") & (pc.question == qlab)].iloc[0]
                lines.append(f"{s} & {SHORT[m]} & {f1(r1(100 * (v.code == 'yes').mean()))} & "
                             f"{f1(r1(lo))}--{f1(r1(hi))} & {len(v)} & {v.child.nunique()} & "
                             f"{f1(row.children_at_least_once)} & {f1(row.children_majority)} \\\\")
    lines.append("%% refusal sensitivity audio")
    for q, qlab in TABLE_QS:
        lines.append(f"%% {qlab}")
        for s in TABLE_STUDIES:
            for m in TABLE_MODELS:
                sel = rs[(rs.study == s) & (rs.model == m) & (rs.modality == "audio") & (rs.question == qlab)]
                if not len(sel):
                    continue
                r = sel.iloc[0]
                lines.append(f"{s} & {SHORT[m]} & {r.attempts} & {r.excluded} & {f1(r.rate_valid_only)} & "
                             f"{f1(r.rate_fail_safe)} & {f1(r.rate_fail_open)} \\\\")
    lines.append("%% gating rule")
    g2 = gt.assign(_m=gt.model.map(TABLE_MODELS.index)).sort_values(["study", "modality", "_m"], kind="stable")
    for _, r in g2.iterrows():
        lines.append(f"{r.study} & {SHORT[r.model]} & {r.modality} & {r.pairs_with_both} & {r.children} & "
                     f"{f1(r.rate_binary_alone)} & {f1(r.rate_numeric_alone)} & {f1(r.rate_and_rule)} & "
                     f"{f1(r.and_ci_lo)}--{f1(r.and_ci_hi)} & {f1(r.and_rule_children_at_least_once)} & "
                     f"{f1(r.and_rule_children_majority)} \\\\")
    lines.append("%% full H1/H2 all modalities")
    for q, qlab in TABLE_QS:
        lines.append(f"%% {qlab}")
        for s in TABLE_STUDIES:
            for m in TABLE_MODELS:
                mods = sorted(b[(b.study == s) & (b.model == m)].modality.unique())
                groups = [(mo, b[(b.study == s) & (b.model == m) & (b.modality == mo) & (b.question == q)])
                          for mo in mods]
                if len(mods) > 1:
                    groups.append(("pooled", b[(b.study == s) & (b.model == m) & (b.question == q)]))
                for mo, g in groups:
                    valid = int(g.code.isin(["yes", "no"]).sum())
                    yes = int((g.code == "yes").sum())
                    rate = "n/a" if valid == 0 else f1(r1(100 * yes / valid))
                    lines.append(f"{s} & {SHORT[m]} & {mo} & {rate} & {valid} & {int((g.code == 'refusal').sum())} & "
                                 f"{int(g.code.str.startswith('error').sum())} & {int((g.code == 'unclear').sum())} & "
                                 f"{len(g)} \\\\")
    return "\n".join(lines) + "\n"


def main():
    b = load_binary()
    a = load_ages()
    print(f"binary responses {len(b)}, numeric estimates {len(a)}")
    rs = refusal_sensitivity(b)
    rs.to_csv(os.path.join(OUT, "refusal_sensitivity.csv"), index=False)
    gr = gating_rule(b, a)
    gr.to_csv(os.path.join(OUT, "gating_rule.csv"), index=False)
    pc = per_child_rates(b)
    pc.to_csv(os.path.join(OUT, "per_child_rates.csv"), index=False)
    with open(os.path.join(OUT, "table_rows.tex"), "w") as f:
        f.write(table_rows(b, rs, gr, pc))
    pd.set_option("display.width", 250)
    print("\n== refusal sensitivity, audio only\n", rs[rs.modality == "audio"].to_string(index=False))
    print("\n== gating rule\n", gr.to_string(index=False))
    print("\n== per-child rates, audio only\n", pc[pc.modality == "audio"].to_string(index=False))


if __name__ == "__main__":
    main()
