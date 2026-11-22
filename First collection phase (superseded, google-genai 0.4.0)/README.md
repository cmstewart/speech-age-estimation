# First collection phase, superseded

These results were produced under `google-genai 0.4.0`, the library version that
double-encoded inline audio before transmission. The Gemini estimates in this
folder are artifacts of that defect and are reported in the paper only as
evidence of it. Nothing here should be read as a finding about model capability.

The study letters used here do not correspond to the study labels in the paper.

| Phase one | Manipulation | Status |
|---|---|---|
| 1a | Kids VA scripted | Rerun as Study 1A |
| 1b | Kids VA spontaneous | Rerun as Study 1B |
| 1c | Kids VA downsampled to 16 kHz | Not reported as a finding, hypothesis eliminated |
| 1d | Kids VA noise injection | Rerun as Study 1C |
| 2c | MyST denoised | Not reported as a finding, hypothesis eliminated |

The MyST studies, 2a and 2b, were unaffected by the defect and were not rerun.
Their data is in the main study folders of this repository.

Every file here holds a coded outcome for each response and not the response
text. The `outcome`, `pred_age` and `refusal_check` columns are described in the
main README.

The coded GPT-4o Audio Preview responses in `study1a_raw_results.csv` and
`study1b_raw_results.csv` are the source for the refusal rates reported in the
paper's Discussion under Failure 2. Preview was retired before the second
collection phase, so those responses cannot be regenerated.

Those Preview calls placed the audio part before the text part. A control on
2026-11-22 showed that gpt-audio ignores the audio in that order. Preview's
use of the audio in these calls cannot be confirmed. Its few Kids VA age
estimates do not track the children's true ages. The control is described in
`../gpt-audio control and rerun (2026-11-22)`.

## Files

- `study1a_raw_results.csv` — Kids VA scripted. Source for the corrupted-pipeline
  mean of 36.8 years, its mean absolute error of 29.4 years, and the 70 of 84
  valid responses, all reported in the paper's Table 8. Also the source for
  Preview's Kids VA refusal rates.
- `study1b_raw_results.csv` — Kids VA spontaneous, under the same defect.
- `study1c_raw_results.csv` — Kids VA downsampled to 16 kHz to match the MyST
  sample rate. Mean estimate 37.1, so sample rate did not explain the apparent
  corpus gap.
- `study1d_raw_results.csv` — Kids VA with injected background noise across nine
  conditions. Every condition fell within two years of a clean baseline of 36.3,
  so background noise did not explain the apparent corpus gap.
- `study2c_results_gemini_250.csv` — MyST with ambient classroom noise removed by
  spectral gating. Mean estimate moved from 9.2 to 9.5, so ambient noise did not
  explain the apparent corpus gap either.
