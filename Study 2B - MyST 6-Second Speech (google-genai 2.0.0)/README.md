# Study 2B: Age Estimation from Truncated Speech (MyST Corpus, 6 seconds)

## Overview

This study uses the same 250 students and the same speech content as Study 2A. The speech is truncated from about 30 seconds to 6 seconds. It runs both audio and text conditions. The purpose is to isolate duration as a factor in age estimation accuracy. It asks whether a model that estimates ages accurately from 30 seconds still does so from 6 seconds of the same speech.

This study ran on the pinned google-genai 2.0.0 setup, so its Gemini estimates are reliable.

## Models

- Gemini 2.5 Flash, via the google-genai SDK
- GPT-4o Audio Preview, via the OpenAI SDK
- Nemotron 3 Nano Omni, via fal.ai for audio and DeepInfra for text

GPT-4o Audio Preview has since been retired by OpenAI. Its successor gpt-audio refuses age estimation on nearly every clip, so the OpenAI results here reflect a model that is no longer available.

## What we found

### Duration is not what drives accuracy

Gemini's audio estimates barely changed when the clip was cut from 30 seconds to 6 seconds. The mean moved from 9.2 to 9.0. Gemini identifies these MyST children as children from just 6 seconds of speech. This matches its corrected performance on the Kids VA corpus. There it estimates 5 to 10 year olds near 6.7 from clips of similar length. Accurate estimation holds at short duration on both corpora.

### Duration does matter for Nemotron

Nemotron's audio mean rose from 7.9 at 30 seconds to 11.6 at 6 seconds. Its over 13 error rate jumped from 5.0 percent to 29.7 percent. Nemotron needs longer audio and loses accuracy on short clips.

### GPT-4o was heavily rate-limited

Rate limiting returned errors on 52 percent of audio calls, which left only 90 of 250 students with valid responses. Among those the mean estimate of 9.6 was reasonable, but the over 13 error rate rose to 44.0 percent. The rate limiting is itself a practical finding, since it made a full scale audio study impractical on the preview model at the time of testing.

### Text degraded more than audio

Shortening the transcript pushed the text estimates up more than the audio. Gemini's text mean went from 11.3 to 16.2 and Nemotron's from 14.9 to 22.4. The audio minus text gap widened for both models. This means the acoustic signal matters even more when there is little text to work from.

### There is no corpus divergence to explain

An earlier version of this analysis treated Gemini's Kids VA estimate near 35 as a corpus specific failure and called it a puzzle. That estimate was an artifact. The original Kids VA run used an older google-genai version that double encoded the audio, so Gemini was estimating from corrupted input. Once the defect was fixed, Gemini estimated the Kids VA children accurately. Gemini is now accurate at 6 seconds on both corpora, MyST near 9.0 and Kids VA near 6.7. Duration does not drive accuracy and speech type does not drive it. The earlier divergence was a software defect, and only Gemini was affected, since the other two models used different SDKs.

## Caveats

GPT-4o audio results rest on 90 of 250 students because of rate limiting and may not be representative. The truncated text transcripts are approximate. They are taken as the first fifth of the words and may not correspond exactly to the first 6 seconds of audio. As with all studies in this project, we cannot fully separate model capability from alignment policy from recording artifacts.

## Files

- `study2b_analysis.ipynb` is the full analysis notebook.
- `study2b_results_combined.csv` holds a coded outcome for each API response from the audio and text checkpoints. The response text is not included.
- `study2b_q3_parsed.csv` holds the parsed age estimates for Q3.
- `study2b_student_means.csv` holds the per-student mean estimates.
- `study2b_age_distributions.png` shows the estimate distributions by model and condition.
- `study2b_duration_comparison.png` compares the 6 second and 30 second audio estimates.
- `study2b_binary_classification.png` shows the over 13 and over 18 rates.
- `Code/run_experiment_2b.py` is the experiment script.
- `Code/rerun_gpt4o_audio.py` is the GPT-4o retry script with rate limit handling.
- `SAMPLES_WITHHELD.md` explains why no sample audio or transcripts are included.

## Citations

Speech data from Pradhan, S., Cole, R. A., and Ward, W. MyST Children's Conversational Speech LDC2021S05. Web Download. Philadelphia. Linguistic Data Consortium, 2021.
