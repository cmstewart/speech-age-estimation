# Dedicated speech-age baseline

This folder holds a dedicated speech-age model run on the Kids VA clips of
Studies 1A and 1B. It tests whether the errors in the main studies are specific
to prompted language models. The dedicated model did worse than the language
models on the same clips.

## Model

The model is `audeering/wav2vec2-large-robust-24-ft-age-gender` from Hugging
Face at revision `bad2737f529168bc3d3fcd97edfaba9583c7d36d`. It is a 24-layer
wav2vec 2.0 model fine-tuned on aGender, Mozilla Common Voice, TIMIT and
VoxCeleb 2. It returns a point estimate of age and probabilities for three
classes labeled child, female and male. Its developers release it under
CC BY-NC-SA 4.0. The weights are not redistributed here.

## What was run

The model scored the 56 clips of Studies 1A and 1B exactly as the language
models received them. Clips were resampled to 16 kHz to match the model's input
rate. Inference ran on a local CPU, and no audio was sent to any external
service. The model is deterministic, so each clip has one estimate.

Each estimate is thresholded at 13 and at 18. An estimate at or above the
threshold counts as an error, since every speaker is under 13. The model
answers no yes-or-no question. The comparison therefore uses each language
model's own numeric estimates on the same clips with the same thresholds.

## Results

Rates are in percent. MAE is mean absolute error in years. The last column is
the correlation between estimated and true age.

| Study | Model | n | Over 13 | Over 18 | MAE | r |
|---|---|---|---|---|---|---|
| 1A | Gemini | 84 | 1.2 | 1.2 | 1.5 | .54 |
| 1A | Nemotron | 84 | 0.0 | 0.0 | 1.6 | .52 |
| 1A | gpt-audio, text first | 57 | 0.0 | 0.0 | 1.0 | .52 |
| 1A | wav2vec2 baseline | 28 | 42.9 | 10.7 | 5.2 | .46 |
| 1B | Gemini | 84 | 0.0 | 0.0 | 1.5 | .57 |
| 1B | Nemotron | 79 | 8.9 | 6.3 | 3.6 | .15 |
| 1B | gpt-audio, text first | 56 | 5.4 | 1.8 | 1.6 | .62 |
| 1B | wav2vec2 baseline | 28 | 64.3 | 35.7 | 8.9 | .33 |

A constant prediction of 7 years gives a mean absolute error of 1.1. The
gpt-audio rows come from the text-first rerun in
`gpt-audio control and rerun (2026-11-22)`. gpt-audio gave estimates for 21 children
in Study 1A and 19 in Study 1B. All of its Study 1B estimates at or above 13
came from one 10 year old.

The model also returns a class label for each clip. `class_label_rate.py`
counts the clips whose label is not child and writes
`results/baseline_class_label.csv`. The model gave 15 of the 56 clips the female
label in place of the child label. That is 26.8%.

## Files

| File | Contents |
|---|---|
| `baseline_wav2vec2_age.py` | Runs the model on the clips and writes the three baseline files in `results` |
| `compare_llm_numeric.py` | Thresholds the language models' numeric estimates the same way and writes the comparison table |
| `results/baseline_clip_estimates.csv` | One row per clip with the age estimate, the class probabilities and a SHA-256 hash of the audio file |
| `results/baseline_summary.csv` | Rates, intervals, mean absolute error and correlation for each study |
| `results/comparison_numeric_threshold.csv` | The baseline beside the language models, including per-child rates |
| `results/baseline_run_info.json` | Model revision, weight hashes, seed and software versions for the run |

## Reproducing

The comparison table recomputes from the committed CSV files. It needs no audio
and no model weights.

```
python "Dedicated speech-age baseline/compare_llm_numeric.py" --repo .
```

Rerunning the model needs two things that are not in this repository:

- The Kids VA stimulus audio named in the manifests in the Study 1A and
  Study 1B folders, built from a licensed copy of the corpus
- The model weights, downloaded from Hugging Face at the revision above

```
python baseline_wav2vec2_age.py --model-dir PATH --audio-1a PATH --audio-1b PATH
```

`--model-dir` is the folder with the downloaded model files. `--audio-1a` and
`--audio-1b` are the folders with the stimulus audio for each study. The script
reads the manifests from this repository and writes to `results` beside itself.
The `--repo`, `--manifest-1a`, `--manifest-1b` and `--out` options override
those defaults. The script stops if a manifest or any clip it names is missing.

The run used Python 3.10.12, torch 2.6.0, transformers 5.20.2, numpy 2.2.6,
scipy 1.15.3, pandas 2.3.3 and soundfile 0.14.0. Intervals come from a
percentile bootstrap that resamples speakers over 10,000 draws with a fixed
seed.
