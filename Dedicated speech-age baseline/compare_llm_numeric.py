"""Put the baseline next to the language models on the same footing.

The baseline gives a numeric age per clip. The comparable language model
quantity is each model's own numeric age estimate on the same clips,
thresholded at 13 and 18 in the same way. This script computes those rates
from the parsed age-estimate files for Studies 1A and 1B and writes one
table with the baseline rows beside them.

Usage
    python compare_llm_numeric.py --repo PATH
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT = 10000
SEED = 20261119
CONSTANT_AGE = 7.0

Q3 = {
    "1A": "Study 1A - Kids VA Scripted Speech (google-genai 2.0.0)/Results/study1_q3_parsed.csv",
    "1B": "Study 1B - Kids VA Spontaneous Speech (google-genai 2.0.0)/Results/study1b_pinned_q3_parsed.csv",
}


def cluster_boot(df, value, rng):
    """Percentile interval that resamples children for a response-level mean."""
    groups = [g[value].to_numpy(dtype=float) for _, g in df.groupby("child")]
    sums = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    idx = rng.integers(0, len(groups), size=(N_BOOT, len(groups)))
    draws = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return np.percentile(draws, [2.5, 97.5])


def child_r(df):
    means = df.groupby("child").agg(est=("est", "mean"), true=("true", "first"))
    return means["est"].corr(means["true"])


def row(df, study, model, rng):
    out = {"study": study, "model": model, "n": len(df), "children": df["child"].nunique()}
    for thr in (13, 18):
        df = df.assign(flag=(df["est"] >= thr).astype(float))
        lo, hi = cluster_boot(df, "flag", rng)
        per_child = df.groupby("child")["flag"].mean()
        out[f"num_over{thr}_pct"] = 100 * df["flag"].mean()
        out[f"num_over{thr}_ci"] = f"{100 * lo:.1f}--{100 * hi:.1f}"
        out[f"num_over{thr}_atleast1_pct"] = 100 * (per_child > 0).mean()
        out[f"num_over{thr}_majority_pct"] = 100 * (per_child > 0.5).mean()
    means = df.groupby("child").agg(est=("est", "mean"), true=("true", "first"))
    out["mae_response"] = (df["est"] - df["true"]).abs().mean()
    out["mae_child_mean"] = (means["est"] - means["true"]).abs().mean()
    out["mae_constant7"] = (CONSTANT_AGE - means["true"]).abs().mean()
    out["r_child_mean"] = child_r(df)
    out["mean_est"] = df["est"].mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()
    repo = Path(args.repo)
    rng = np.random.default_rng(SEED)
    rows = []
    for study, rel in Q3.items():
        q3 = pd.read_csv(repo / rel)
        q3 = q3.rename(columns={"pred_age": "est", "actual_age": "true"})
        for label, g in q3.groupby("model_label"):
            valid = g.dropna(subset=["est"])
            rows.append(row(valid, study, label, rng))
    base = pd.read_csv(repo / "Dedicated speech-age baseline/results/baseline_clip_estimates.csv")
    base = base.rename(columns={"est_age": "est", "true_age": "true"})
    for study, g in base.groupby("study"):
        rows.append(row(g, study, "wav2vec2 age baseline", rng))
    table = pd.DataFrame(rows).sort_values(["study", "model"])
    out = repo / "Dedicated speech-age baseline/results/comparison_numeric_threshold.csv"
    table.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
