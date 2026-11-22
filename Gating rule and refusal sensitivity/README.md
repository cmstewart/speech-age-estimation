# Gating rule and refusal sensitivity

`gating_and_refusals.py` reproduces every threshold table in the paper from the
committed response files. Run it from this folder with no arguments. It writes
`refusal_sensitivity.csv`, `per_child_rates.csv`, `gating_rule.csv`, and
`table_rows.tex`, which holds each table row in the paper's LaTeX form. The
docstring at the top of the script describes the two analyses and the coding
rules. For gpt-audio on Studies 1A and 1B the script uses the text-first rerun
in `gpt-audio control and rerun (2026-11-22)`. It drops gpt-audio on Study 1C,
which was not rerun. Requires pandas and numpy.
