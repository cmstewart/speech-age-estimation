# Transport control for gpt-audio in Studies 1A and 1B

> **Note added after the rerun.** This report was written before Studies 1A and 1B were rerun with the text part first. Its verdict predates the rerun and rests on one adult voice. The rerun later showed that gpt-audio answers many questions about the Kids VA children once it can hear the clips. The README in this folder gives the rerun results. The report is kept as written apart from the removal of local paths and of hashes of Kids VA audio.

Run on 2026-11-22 at 20:29 UTC. Only the adult control clip was sent. No child audio left the machine. No existing file was changed and nothing was committed.

## Verdict

**The audio arrives and the model refuses age questions for an adult too.**

The follow-ups settle this verdict. The server received and decoded the full clip on every call. In the study request shape, where the audio part comes before the text part, the model ignored the audio even for a plain transcript. It asked for the audio three times out of three. With the text part moved first and nothing else changed, the model transcribed the clip correctly three times out of three. The author confirmed the words. In that same working shape it still refused both age questions six times out of six. The age refusal is therefore a model policy. It is not a transport fault and not a side effect of the part order.

The evidence that the audio arrives is as follows.

- The request body is a single base64 pass of a valid RIFF WAVE file. It is declared as `wav` and is sample for sample identical to the source. See the offline result below.
- All nine calls returned HTTP 200. Every one reports `usage.prompt_tokens_details.audio_tokens = 62` for a 6.2 s clip. That is exactly 10 audio tokens per second, so the server decoded and tokenized the whole clip. A corrupt, empty or double encoded field would not produce a count matched to the duration.
- The audio token count is the same for all three prompts. Only `text_tokens` changes, at 25, 52 and 89 for the three prompts.

What this means for Studies 1A and 1B:

- The Kids VA requests were built by the same function with the same encoding. The offline check shows they are as well formed as the adult request.
- There is no transport defect like the earlier Gemini double encoding.
- The refusals are not specific to children. The model refused both age questions for an adult speaker in all six calls.
- The "please provide the audio" replies in the study data are reproduced here with adult audio that is known to have arrived. With the audio part first, the model sometimes behaves as if no audio was attached. That is a property of the model and the part order. The bytes are not missing.
- An age refusal from a model that demonstrably heard the clip shows that the study refusals would very likely occur in the text-first order too. The study shape did not cause them.
- The studies did not log usage fields, so the 62 token check exists only for this control. It cannot be read back from the original runs.

Because the verdict is not that the audio failed to arrive, no fix is proposed. Studies 1A and 1B were not rerun.

## Step 1. Code path and settings

| Item | Value |
|---|---|
| Study 1B file | `Study 1B - Kids VA Spontaneous Speech (google-genai 2.0.0)/study1b_age_estimation_pinned.ipynb`, cell 12 |
| Study 1B function | `call_gpt_audio`, which wraps `_gpt_audio_call` in a tenacity retry |
| Study 1A function | `call_gpt_audio` in cell 11 of an earlier copy of the Study 1A notebook that is not in this repository. It builds the same request body with no retry. It is missing from the committed 1A notebook. |
| Model string | `gpt-audio-2025-08-28` |
| Endpoint | `POST /v1/chat/completions` |
| Settings | `modalities=['text']`, `max_completion_tokens=300`, no temperature so the server default applies, a delay of 1.0 s between calls |
| Audio part | `{'type': 'input_audio', 'input_audio': {'data': wav_b64, 'format': 'wav'}}` placed before the text part |
| Encoding | The FLAC is read with soundfile, written in memory as WAV PCM_16 and base64 encoded once to a str |
| SDK | openai 2.36.0 in the `study1-pinned` env |

`study_caller.py` holds a verbatim copy of the 1B loader, caller and questions. The only change is that the client is passed in by the importing script.

A side finding: the committed 1B notebook sets `STIMULI_DIR` to `study1d_stimuli_noise/`. The logged durations match the spontaneous stimuli, for example 8.05 s for sp104, so the run used the right audio despite that path.

## Steps 2 and 3. Offline result

The study function was run against an in-memory httpx transport, so nothing was sent. Reproduce with `offline_check.py`.

| Field | 1A sp104 | 1B sp104 |
|---|---|---|
| Source file | `study1_stimuli/sp104_stimulus.flac` | `Stimuli/sp104_spontaneous.flac` |
| Source container | FLAC PCM_16 | FLAC PCM_16 |
| Declared format in request | wav | wav |
| Container after one base64 decode | RIFF WAVE, format tag 1, PCM 16 bit | RIFF WAVE, format tag 1, PCM 16 bit |
| Sample rate and channels | 44100 Hz mono | 44100 Hz mono |
| Duration | 7.552 s | 8.052 s |
| Second decode | not base64, so no double encoding | not base64, so no double encoding |
| Decoded field equals the WAV given to the function | yes | yes |
| SHA-256 of decoded PCM | same as SHA-256 of source PCM | same as SHA-256 of source PCM |
| Base64 length to decoded WAV | 1.3333 | 1.3333 |
| Decoded WAV to raw PCM | 1.0001 | 1.0001 |
| Base64 length to source FLAC | 3.3448 | 3.0714 |

The decoded field cannot hash to the source file because the code converts FLAC to WAV on purpose. The PCM samples are identical, which is the relevant comparison. A ratio of 1.333 is one base64 pass. Double encoding would give about 1.78.

The declared format and the real container agree. WAV is an accepted input format for this endpoint. Across the 1A and 1B stimuli, 55 files are 44.1 kHz and one is 48 kHz. All are mono PCM_16, so the stimuli have no format mismatch.

## Step 4. Adult control clip

- The source is a recording of the author's own voice. The author confirmed they are an adult. The median F0 is about 85 Hz.
- The source was already FLAC PCM_16 at 44100 Hz mono and 9.4 s long. `prepare_adult.py` trimmed it to 6.2 s, the median stimulus length across 1A and 1B, following the mono and write steps in the 1B stimulus builder. The 1A builder is not in the repo.
- The SHA-256 of `adult_control.flac` is `d7f6309a23bc9a4e61253a7fbcc9d94858a368e2898867f2693bebd6cdcd7675`. It matches no Kids VA stimulus file.
- The SHA-256 of the WAV sent is `fba71831d9ad5e18b764a00a9db9dcce795bab814b89b2258b711020dd98fc36`, which is 729,180 base64 characters.

## Step 5. The nine live responses

Prompts:

- The transcript prompt was written for this control: "Transcribe this audio clip verbatim. Output only the words spoken."
- `q2_over_18` and `q3_age` were copied exactly from the study prompts.

Every call returned HTTP 200 with `finish_reason` set to `stop` and an empty `refusal` field. The system fingerprint was `fp_9db1bde611` for the transcript calls and `fp_19924d2c5b` for the age calls. Raw responses, redacted request bodies, request IDs and timestamps were logged to `responses.jsonl`, which is not distributed.

| Prompt | Repeat | Audio tokens | Full response text |
|---|---|---|---|
| transcript | 1 | 62 | Understood. Please provide the audio clip you'd like transcribed. |
| transcript | 2 | 62 | Understood. Please provide the audio clip you would like transcribed. |
| transcript | 3 | 62 | Understood. Please provide the audio clip and I will transcribe the content verbatim. |
| q2_over_18 | 1 | 62 | I'm sorry, but I can't determine the speaker's age from the audio. It's not possible to reliably identify age just from acoustic and linguistic features. |
| q2_over_18 | 2 | 62 | I'm sorry, but I can't assist with identifying personal attributes such as age from an audio sample. Let me know if you need help with anything else. |
| q2_over_18 | 3 | 62 | I'm sorry, but I can't determine the age of a speaker from the audio. I can assist you with other types of analysis or answer any other questions you might have. Let me know how you'd like to proceed. |
| q3_age | 1 | 62 | I'm sorry, but I can't help with identifying or estimating the age of a speaker based on their voice or any audio features. Let me know if there's something else I can assist you with. |
| q3_age | 2 | 62 | I'm sorry, but I can't assist with identifying a speaker's age from an audio clip. Let me know if there's something else I can help you with. |
| q3_age | 3 | 62 | I'm sorry, but I can't assist with identifying or estimating personal characteristics of a speaker, such as age, from an audio clip. Let me know if there's anything else I can help you with. |

## Step 7. First phase payloads

**None were found.** No file in the authors' working folders contains a base64 WAV string, which always starts with `UklGR`. Every match for `input_audio` was source code. The offline check could not be repeated on a logged first phase payload.

The first phase GPT-4o Audio Preview code builds the same WAV conversion and the same `input_audio` part. That holds in the first phase notebooks and in the 1B spontaneous notebook. It differs from the gpt-audio caller in three ways. It uses `max_tokens=300` in place of `max_completion_tokens`. It sends no `modalities` field. It does not read `msg.refusal`.

## Differences between the Kids VA and MyST calling code

The MyST runs in Studies 2A and 2B never called gpt-audio. The scripts are `Code/run_experiment.py`, `Code/run_experiment_2b.py` and `Code/rerun_gpt4o_audio.py`.

| Item | Kids VA 1A and 1B | MyST 2A and 2B |
|---|---|---|
| Model | `gpt-audio-2025-08-28` | `gpt-4o-audio-preview` |
| Token limit parameter | `max_completion_tokens=300` | `max_tokens=300` |
| `modalities` | `['text']` | not sent |
| Retry | tenacity in 1B, none in 1A | none in 2A, a custom 429 retry in the 2B rerun |
| Refusal field | read if content is empty | not read |
| Audio encoding | FLAC to in-memory WAV, one base64 pass | same |
| Audio part and order | `input_audio` wav, then text | same |
| Prompts | same three questions | same three questions |
| Source sample rate | 44.1 kHz, one file at 48 kHz | 16 kHz from the MyST builder |

The MyST results therefore cannot be used to compare transport on gpt-audio. They differ from the Kids VA runs in model as well as in two request parameters.

## Follow-ups

The author approved four variants on the adult clip only and set a rule: run them in order and stop at the first one that returns a correct transcript, then send the two age questions in that shape three times each. The runner is `followups.py`. Its raw log `followups.jsonl` is not distributed. Run on 2026-11-22.

| Order | Variant | Status |
|---|---|---|
| 1 | Study request with the text part before the audio part. Everything else unchanged: `gpt-audio-2025-08-28`, `modalities=['text']`, `max_completion_tokens=300` | Run. Correct transcript on the first try, so the run stopped here |
| 2 | Study request with `modalities=['text','audio']` | Not run under the stop rule |
| 3 | The same WAV bytes to `gpt-4o-transcribe` | Not run under the stop rule |
| 4 | The exact MyST request shape against gpt-audio, with `max_tokens=300` and no `modalities` field | Not run under the stop rule |

Every call in variant 1 returned HTTP 200 with 62 audio input tokens, the same count as the study shape.

| Prompt | Repeat | Full response text |
|---|---|---|
| transcript | 1 | The goal here is to arrive at a symbolic representation of incoming text in which |
| transcript | 2 | The goal here is to arrive at a symbolic representation of incoming text in which |
| transcript | 3 | The goal here is to arrive at a symbolic representation of incoming text in which |
| q2_over_18 | 1 | I'm sorry, but I can't determine the speaker's age from the audio. It isn't possible to reliably identify age based on acoustic and linguistic features alone. Let me know if you need help with anything else. |
| q2_over_18 | 2 | I'm sorry, but I can’t assist with determining the age of a speaker based on audio. Let me know if you have any other questions or need help with something else. |
| q2_over_18 | 3 | I'm sorry, but I can't assist with determining the age of a speaker from audio. Let me know if there's anything else I can help with. |
| q3_age | 1 | I'm sorry, but I can't assist with identifying or estimating the age of a speaker from an audio clip. Let me know if there's anything else I can do for you. |
| q3_age | 2 | I'm sorry, but I can't assist with identifying or estimating the age of a speaker from an audio clip. Let me know if there's something else I can help you with. |
| q3_age | 3 | I'm sorry, but I can't analyze the speaker's age from the audio. I can assist with other types of analysis or help answer questions in a different way. Let me know how else I can assist you. |

The transcript stops mid-sentence because the clip was trimmed to 6.2 s. The author confirmed it matches the words spoken.

What the follow-ups show:

- Part order decides whether gpt-audio uses the audio. Audio first gave three requests for the audio. Text first gave three correct transcripts. The bytes and the token count were the same in both.
- Part order does not decide the age answer. Both shapes refused all six age questions.
- Variant 4 was the leading suspect for the difference between the two collections. It was not tested because variant 1 met the stop rule first. It matters less than it seemed. The MyST and first phase runs used a different model, `gpt-4o-audio-preview`, so the request body was never the only difference. gpt-audio now refuses age questions even in a shape where it demonstrably hears the clip.
- One caution for any rerun is that the study shape can make gpt-audio ignore the audio. A rerun with the text part first would be a cleaner condition. It would very likely still refuse the age questions. Whether to rerun is the author's decision.

## Files

- `study_caller.py` holds the verbatim study loader, caller and prompts.
- `offline_check.py` runs the step 2 and 3 check with no network.
- `prepare_adult.py` performs the step 4 conversion.
- `live_run.py` runs the step 5 calls in `--dry-run` and `--send` modes.
- `followups.py` runs the follow-up variants.
- `adult_control.flac`, `responses.jsonl` and `followups.jsonl` were kept locally and are not distributed.
