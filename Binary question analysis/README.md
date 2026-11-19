# Binary Question Analysis

This folder checks whether the continuous age-estimation question and the two binary questions, "over 13?" and "over 18?", agree with each other. All three were asked in the same prompts across the five studies, so a claim drawn from the age estimate should hold on the binary answers from the same responses. Where they disagree, that disagreement points to a lack of coherence in age estimates

The notebook `binary_threshold_consistency.ipynb` loads the raw response files, parses the binary and continuous answers with the same logic used in the study notebooks, and reports every rate with its denominator, since refusals leave very different sample sizes across models and conditions.

The results files in this folder now hold coded outcomes in place of the response text. The notebook was run on the response text, so it documents the analysis and does not rerun from these files. The rate script in `Gating rule and refusal sensitivity` reproduces the threshold rates from the codes.

## Figures

The notebook produces two figures.

`binary_over13_by_study.png` shows the incorrect over-13 rate for each model across the five studies, with the valid-response count on each bar.

`binary_modality_over13.png` shows the incorrect over-13 rate for audio against text in the two MyST studies, which carries the audio-beats-text result onto the threshold question.

## Two findings

Two results from this analysis stand out.

Gemini degrades on short conversational audio. On Study 2B audio it flagged these 8 to 11 year old children as over 13 on 15.5% of clips and as over 18 on 10.9%. Both rates are internally consistent, but both are high for children this far from either threshold, and the age-estimation framing alone had hidden this.

Nemotron's binary answers expose its bias rather than its accuracy. On Study 1B its over-13 error rate reaches 39.5% while its own age estimates for the same clips exceed 13 only 8.9% of the time, so the two questions disagree on the same responses. On the MyST studies its text over-13 rate is near zero, which reflects a young-guessing default rather than accurate judgment. The conclusion is that a model that passes a safety check by guessing young has not really learned to judge age.
