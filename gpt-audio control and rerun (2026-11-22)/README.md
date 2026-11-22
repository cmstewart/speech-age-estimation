# gpt-audio control and rerun, 2026-11-22

The first gpt-audio runs on Kids VA placed the audio part of each request before the text part. A control on 2026-11-22 showed that gpt-audio ignored the audio in that order. The server still received the audio and counted its tokens. No error was returned. Studies 1A and 1B were then rerun for gpt-audio with the text part first. Study 1C was not rerun.

## What was sent and when

All calls went to the OpenAI Chat Completions endpoint with the model `gpt-audio-2025-08-28`. All times are UTC on 2026-11-22.

| Time | Calls | Audio | Request shape |
|---|---|---|---|
| 20:29 to 20:30 | 9 | One adult control clip | Audio first, as in the studies |
| 20:37 to 20:39 | 9 | The same adult control clip | Text first |
| 20:54 to 21:13 | 504 | The 56 Kids VA stimuli of Studies 1A and 1B | Text first |
| 20:54 and 21:13 | 6 | The same adult control clip | Text first |

The adult control clip is a 6.2 second recording of one of the authors. It was sent with a transcript prompt and with the over-18 and open age questions. Its six calls in the rerun ran three before and three after the Kids VA calls. They showed that the text-first shape worked for the whole session.

The Kids VA calls sent each of the 28 children in each study three times for each of the three study questions. This sent the Kids VA stimuli to OpenAI a second time. The authors approved that before the run.

## What changed

The only change from the study requests was the part order. The text part came first and the audio part second. Everything else matched the study code. That covers the model, the prompts, the `modalities` setting, the limit of 300 output tokens, the three repeats, the one second delay between calls, the retry rule and the conversion of each FLAC clip to WAV. No temperature, seed or system message was set, which matches the studies.

## Results in brief

- The adult transcript failed in the audio-first shape. The model asked for the audio three times out of three.
- The adult transcript was correct in the text-first shape three times out of three.
- In the rerun, gpt-audio gave a valid answer to the over-13 question 19 times in Study 1A and 12 times in Study 1B. Every answer was no.
- It gave a valid answer to the over-18 question 10 times in Study 1A and 9 times in Study 1B. Every answer was no.
- It gave an age estimate for 21 of 28 children in Study 1A and 19 of 28 in Study 1B.
- The mean absolute error per response was 0.98 years in Study 1A and 1.59 years in Study 1B.

The analysis scripts in this repository use these rows for gpt-audio on Studies 1A and 1B. They drop gpt-audio on Study 1C from every rate. The audio-first rows stay in the study folders as the record of the failure.

## Files

| File | Contents |
|---|---|
| `rerun_coded.csv` | One coded row for each of the 504 Kids VA calls in the rerun. It holds no response text. |
| `CONTROL_REPORT.md` | The report on the transport control. Its verdict predates the rerun. |
| `study_caller.py` | The study loader, caller and prompts, copied from the Study 1B notebook |
| `offline_check.py` | Builds one study request without sending it and checks the encoded audio |
| `prepare_adult.py` | Converts an adult clip to the Kids VA stimulus format |
| `live_run.py` | Sends the adult clip in the audio-first study shape |
| `followups.py` | Sends the adult clip in the follow-up shapes |
| `rerun_text_first.py` | Sends the text-first rerun and the adult control calls |
| `code_rerun.py` | Codes the rerun log with the rules used for every Kids VA results file |

The columns of `rerun_coded.csv` follow the Kids VA results files. `request_shape` is always `text_first`. `audio_tokens` is the audio token count the server reported for the call. `system_fingerprint` and `request_id` are the identifiers that OpenAI returned.

## What is not included

The raw call logs hold model response text, and the responses about Kids VA clips can repeat what the children said. The logs are therefore not included. The adult control clip is not included either.

To rerun the scripts, set `OPENAI_API_KEY`. Set `KIDSVA_STIMULI_1A` and `KIDSVA_STIMULI_1B` to the stimulus folders built from a licensed copy of the corpus. Record an adult clip and convert it with `prepare_adult.py`. The scripts never write the key to disk.
