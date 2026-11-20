# Four-study analysis

This folder holds the analysis restricted to Studies 1A, 1B, 2A and 2B. It
leaves out Study 1C. Every script reads the coded outcomes in this repository
and reuses the coding rule and the Study 2B deduplication in
`Gating rule and refusal sensitivity/gating_and_refusals.py`. No script needs
audio, transcripts or model weights.

## Files

| File | Contents |
|---|---|
| `four_study_numbers.py` | Recomputes the reported rates, intervals, per-child rates, paired tests and contradiction counts for the four studies |
| `additional_checks.py` | Recomputes the clip-length differences, the share of estimates inside the true age range, the contradiction baseline and the slope intervals |
| `figure1.py` | Draws mean estimated age against true age for Studies 1A and 1B |
| `figure2.py` | Draws the audio and transcript error rates for Studies 2A and 2B |
| `table_rows_to_json.py` | Converts `table_rows.tex` to JSON without the Study 1C rows and the pooled rows |
| `results/` | The output of each script as run on the committed data |

## Reproducing

Run each script from the repository root.

```
python "Four-study analysis/four_study_numbers.py" . out_numbers.json
python "Four-study analysis/additional_checks.py" .
python "Four-study analysis/figure1.py" . out_figure1
python "Four-study analysis/figure2.py" . out_figure2
python "Four-study analysis/table_rows_to_json.py" "Gating rule and refusal sensitivity/table_rows.tex" out_tables.json
```

The scripts need numpy, pandas, scipy and matplotlib. Intervals come from a
percentile bootstrap that resamples speakers over 10,000 draws with a fixed
seed.

One check from an earlier version is not reproduced here. It compared two
orders of the response coding rule and needs the raw response text. The
repository holds coded outcomes only.
