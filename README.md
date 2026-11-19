# Speech-Based Age Estimation for Child Safety

Code, coded model responses, and figures for a study of whether audio-capable
language models can tell that a speaker is a child.

<!-- After the journal decision, add the paper title, venue, author list, and citation block. -->

## Summary

Child safety mechanisms in conversational AI depend on knowing when the user is
a child. Young children talk rather than type, so that judgment is made from
speech. We asked three audio-capable language models to judge whether a child
speaker is over 13 and over 18 and to estimate the speaker's age. The models
were Gemini 2.5 Flash, GPT-4o Audio Preview and its successor gpt-audio, and
Nemotron 3 Nano Omni. Every speaker in both corpora is under 13, so every
positive answer to a threshold question is an error.

Across five studies, models classified children as over 13 on 2.5% to 42.5% of
audio responses, and every model that answered placed some 5 to 11 year olds
over 18. Along the way we caught three failures that gave no warning. A client
library double-encoded audio and one model estimated ages in the thirties and
forties for children under 10. A model endpoint was replaced by a successor
that refused every threshold question. A model's threshold answers contradicted
its own numeric estimates. Each would have passed the monitoring a deployment
ordinarily runs.

## Studies

| Study | Corpus | Children | Clip | Manipulation | Modality |
|---|---|---|---|---|---|
| 1A | Kids VA | 28, ages 5 to 10 | 6 s | Scripted speech, baseline | Audio |
| 1B | Kids VA | 28, ages 5 to 10 | 6 s | Spontaneous speech | Audio |
| 1C | Kids VA | 28, ages 5 to 10 | 6 s | Scripted speech, nine background-noise conditions | Audio |
| 2A | MyST | 250, grades 3 to 5 | 30 s | Conversational speech, baseline | Audio and text |
| 2B | MyST | 250, grades 3 to 5 | 6 s | Same speech truncated | Audio and text |

Each stimulus carried three questions, asked three times each with no system
prompt. The two threshold questions ask whether the speaker is over 13 and over
18. The third asks for a numeric age estimate with a confidence rating and a
one-sentence rationale. The full prompt text is in the paper's appendix and in
each study notebook.

## Repository layout

| Folder | Contents |
|---|---|
| `Study 1A - Kids VA Scripted Speech (google-genai 2.0.0)` | Collection notebook, coded outcomes, parsed estimates, figures, pinned environment, and the `google_genai regression` folder documenting the double-encoding defect at the transport layer |
| `Study 1B - Kids VA Spontaneous Speech (google-genai 2.0.0)` | Collection notebook, stimulus builder, coded outcomes, parsed estimates, figures |
| `Study 1C - Kids VA Scripted Speech w- Background Noise Injection (google-genai 2.0.0)` | Noise injection script, collection notebook, coded outcomes, parsed estimates |
| `Study 2A - MyST 30-Second Speech (google-genai 2.0.0)` | Student selection script, stimulus builder, collection script, analysis notebook, coded outcomes, parsed estimates, figures |
| `Study 2B - MyST 6-Second Speech (google-genai 2.0.0)` | Collection script, GPT-4o retry script, analysis notebook, coded outcomes, parsed estimates, figures |
| `Dedicated speech-age baseline` | A dedicated speech-age model run on the Kids VA clips of Studies 1A and 1B, its per-clip estimates, and the comparison with the language models' numeric estimates |
| `Gating rule and refusal sensitivity` | The script that reproduces every threshold table in the paper, its output CSVs, and the table rows as LaTeX |
| `Binary question analysis` | Earlier consistency check between threshold answers and numeric estimates, with its figures |
| `First collection phase (superseded, google-genai 0.4.0)` | Runs under the defective client library, kept as evidence of the defect and as the source of the GPT-4o Audio Preview refusal rates on Kids VA |

Each study folder has its own README.

## What the response files contain

Every results file holds a coded outcome for each response and not the
response text. Model responses quoted the children's words. The MyST license
from the Linguistic Data Consortium prohibits redistribution of corpus content,
and the Kids VA authors asked that none of their corpus be distributed with
this code. The text was therefore replaced with codes. Every rate in the paper
can be recomputed from the codes.

The `outcome` column holds `yes`, `no`, `refusal` or `unclear` for a threshold
question and `answered` for the age question. The Kids VA files also use
`no_age` when a response to the age question held no age. Errors are coded as
an `error` class with its cause. The parsed numeric estimate is kept in
`pred_age`. The Kids VA files carry one more column named `refusal_check`. It
records the classification behind the refusal comparison across endpoints in
the paper. Its values are `refused`, `no_audio`, `error` and `other`.

No audio and no transcripts are redistributed. The Kids VA corpus can be
requested from its authors through the publication that describes it. That
publication is Kim et al. 2022, Examining voice assistants in the context of
children's speech, International Journal of Child-Computer Interaction 34.
The MyST corpus is available from the Linguistic Data Consortium under
LDC2021S05. Each corpus comes under its distributors' own terms. The stimulus
construction scripts rebuild the stimuli from licensed copies of the corpora.
Manifests in the Kids VA folders pin the exact clips used. The script that
drew the 250 MyST students is in the Study 2A `Code` folder. The list of
selected students holds corpus identifiers and file paths, so it is not
distributed here.

## Reproducing the analyses

Every table in the paper recomputes from the committed CSV files without
contacting any API.

`Gating rule and refusal sensitivity/gating_and_refusals.py` reproduces every
threshold table in the paper. It reads the coded outcomes for every study and
keeps each student's first presentation in the Study 2B GPT-4o audio
condition. The codes were produced by checking for a leading YES or NO before
checking for refusal language. It writes the rates under
three treatments of excluded responses, the per-child rates under both rules,
the gating-rule analysis, and `table_rows.tex`, which holds every table row in
the paper. Confidence intervals come from a percentile bootstrap that resamples
speakers with replacement over 10,000 draws with a fixed seed. The script runs
in a few seconds and contacts no API.

The study notebooks reproduce the numeric estimates, the calibration figures,
and the audio-versus-text comparisons. The Kids VA collection notebooks and the
binary analysis notebook were run on the response text. Their saved outputs are
cleared where they quoted the children, and they do not rerun from the coded
files.

`Dedicated speech-age baseline/compare_llm_numeric.py` reproduces the baseline
comparison table from the committed CSV files. Rerunning the baseline model
itself needs the Kids VA audio and the model weights. That folder's README
gives the model revision and the steps.

The Kids VA collection runs used the pinned environment in
`Study 1A - Kids VA Scripted Speech (google-genai 2.0.0)/requirements_pinned.txt`.
The MyST collection ran under google-genai 2.0.0, which rules out the
double-encoding defect. Where a notebook can collect fresh data, API keys are
entered at runtime and never written to disk.

## Ethics

This work analyzed two existing corpora and collected no new data from
children. Both corpora are deidentified and no attempt was made to reidentify
any speaker. Institutional determinations at two of the authors' institutions
found that the work does not involve human subjects research. No adversarial
prompting or refusal bypass was used, and refusals are reported as findings.
The paper's ethics statement describes the data handling in full, including
the retention of uploaded clips on one inference platform's object storage and
its remediation.

## License

MIT. See `LICENSE`.
