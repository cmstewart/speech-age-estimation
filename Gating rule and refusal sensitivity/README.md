# Gating rule and refusal sensitivity

`gating_and_refusals.py` reproduces every threshold table in the paper from the
committed response files. Run it from this folder with no arguments. It writes
`refusal_sensitivity.csv`, `per_child_rates.csv`, `gating_rule.csv`, and
`table_rows.tex`, which holds each table row in the paper's LaTeX form. The
docstring at the top of the script describes the two analyses and the coding
rules. Requires pandas and numpy.
