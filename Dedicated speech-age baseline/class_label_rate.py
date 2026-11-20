"""Class label error rate for the dedicated speech-age baseline.

The baseline returns a class label for each clip as well as an age estimate.
Every speaker is a child, so any label other than child is an error. This
script reports that error rate for each study and for both studies pooled.

Usage
    python class_label_rate.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CLIPS = HERE / "results" / "baseline_clip_estimates.csv"
OUT = HERE / "results" / "baseline_class_label.csv"


def main():
    clips = pd.read_csv(CLIPS)
    clips["not_child"] = clips["top_class"] != "child"
    pooled = clips.assign(study="1A+1B")
    rows = []
    for study, g in pd.concat([clips, pooled]).groupby("study"):
        rows.append({
            "study": study,
            "n_clips": len(g),
            "n_not_child": int(g["not_child"].sum()),
            "pct_not_child": round(100 * g["not_child"].mean(), 1),
            "n_female": int((g["top_class"] == "female").sum()),
            "n_male": int((g["top_class"] == "male").sum()),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
