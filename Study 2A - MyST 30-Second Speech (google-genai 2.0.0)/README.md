# Study 2a: Age Estimation from Children's Speech and Text (MyST Corpus)

## Overview

This study tests whether three language models can estimate a child's age from conversational speech, and whether the audio signal helps or hurts compared to text alone. It uses a larger and different corpus than the Kids VA studies, so its job is to test whether those conclusions hold at scale. 250 students from grades 3 to 5, roughly ages 8 to 11, were drawn from the MyST Children's Conversational Speech Corpus (Pradhan et al., 2021). Each produced about 30 seconds of speech responding to a virtual science tutor. Every student was tested in two conditions. The first was raw audio and the second was a transcript of the same audio. Three questions were asked three times each, Q1 over 13, Q2 over 18, and Q3 estimate age, for 13,500 API calls in total.

This study ran on the pinned google-genai 2.0.0 setup, so its Gemini estimates are reliable.

## Models

- Gemini 2.5 Flash, via the google-genai SDK
- GPT-4o Audio Preview, via the OpenAI SDK
- Nemotron 3 Nano Omni, via fal.ai for audio and DeepInfra for text

GPT-4o Audio Preview has since been retired by OpenAI. Its successor gpt-audio was not run on MyST. On Kids VA it refused most threshold questions but answered some once the text part of each request came before the audio. The OpenAI results here reflect a model that is no longer available.

## What we found

### Audio is more accurate than text for every model

The central result is that the audio condition produced younger and more accurate estimates than the text condition for the same students. This held for all three models and was significant in every case at p below .0001. Gemini estimated 9.2 from audio against 11.3 from text. GPT-4o estimated 8.9 against 9.9. Nemotron estimated 7.9 against 14.9. The acoustic signal pulls estimates toward the true range rather than adding noise.

### Text alone overestimates children's ages

From transcripts only, Gemini and GPT-4o each classified roughly 52 percent of these 8 to 11 year olds as over 13. Nemotron's text estimates averaged 14.9. The likely reason is the science vocabulary in these transcripts. Words like solute and respiratory system read as linguistically mature to a model trained mostly on adult text.

### Audio is more reliable for the regulatory thresholds

For the over 13 question, audio produced 2.5 to 11.6 percent incorrect yes responses, while text produced 1.6 to 51.9 percent. For over 18, audio was nearly perfect, with every model under 2 percent. Text reached as high as 29.8 percent for Gemini.

### The result reinforces the corrected Kids VA studies

Gemini estimated these children accurately from audio. Its mean estimate was 9.2 against a true range of roughly 8 to 11. That matches its corrected performance on the Kids VA corpus. There it estimates 5 to 10 year olds at about 6.7. Taken together, the two corpora show that an audio-capable model estimates children's ages accurately across different recording conditions, speech types, and clip lengths. This larger sample of 250 children strengthens that conclusion beyond what the smaller Kids VA set could support on its own.

## A note on the earlier cross-study comparison

An earlier version of this README reported that Gemini improved dramatically from Study 1. It put the change at about 35 years down to 9.2. That contrast was an artifact. The original Kids VA run used an older google-genai version that double encoded the audio, so Gemini was estimating from corrupted input and returning adult ages. Once that defect was fixed, Gemini estimated the Kids VA children accurately, so there was no failure on that corpus to recover from. Only Gemini was affected, since the other two models used different SDKs. The cross-study story is therefore one of consistency rather than improvement.

## Caveats

These results reflect one corpus of children discussing science topics. The text overestimation may be amplified by that science vocabulary. The corpus documents students as grades 3 to 5 without per-student grade labels, so accuracy within the age range cannot be assessed. The audio condition produced 1,205 errors out of 6,750 calls. That is about 18 percent, and most came from Nemotron. Whether a model's behavior reflects genuine acoustic processing or alignment-time engineering cannot be determined from these results.

## Files

- `study2a_analysis.ipynb` is the full analysis notebook with code, outputs, and interpretation.
- `study2a_results.csv` holds a coded outcome for each API response. The response text is not included.
- `study2a_q3_parsed.csv` holds the parsed age estimates for Q3.
- `study2a_student_means.csv` holds per-student, per-model mean estimates.
- `study2a_q1_over13.csv` holds the Q1 binary classification rates.
- `study2a_q2_over18.csv` holds the Q2 binary classification rates.
- `Code/run_experiment.py` is the experiment script with checkpointing.
- `Code/select_study2_students.py` is the script that drew the 250 students.
- `Code/build_study2_stimuli.py` is the stimulus preparation script that reads from the MyST source.
- `Code/setup.sh` is the VM setup script.
- `config.env.example` is the API key template. Copy it to `config.env` and fill in your keys before running the experiment script. The real `config.env` is gitignored and must never be committed.
- `SAMPLES_WITHHELD.md` explains why no sample audio or transcripts are included.

## Citations

Speech data from Pradhan, S., Cole, R. A., and Ward, W. MyST Children's Conversational Speech LDC2021S05.
