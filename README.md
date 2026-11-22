# Three Invisible Failure Modes in Speech-Based Age Estimation for Child Safety

This repository holds the code, coded model responses, and figures for a poster 
presentation at the NeurIPS 2026 Workshop on Child Safety in AI in Atlanta in December 
2026.

| Author | Affiliation |
|---|---|
| Christopher M. Stewart | University of Memphis |
| Kelsey Kraus | San José State University |
| J. Elliott Casal | University of Memphis |
| Geoffrey L. Brown | University of Wisconsin-Madison |

A preprint can be found [here](https://arxiv.org/pdf/2611.18971). Questions 
can go to the corresponding author at cmstwrt1@memphis.edu.

## High-level overview

AI products increasingly tailor what they say to the age of the user. Major
developers now build age prediction into their products to decide which
safeguards apply. A safeguard cannot fire for a child that the system predicts
is an adult. In the United States, the Children’s Online Privacy Protection Act 
protects  users under 13. European and UK rules add safeguards for users under 
18. None of  these instruments defines an acceptable error rate. Unlike adults,
young children are often still learning to type, so they may often prefer to
speak to AI systems. So, in practice, child-facing conversational systems with
large language model components may use speech from children (among other 
information) to predict a user's age. Although speech-based age estimation has a
long history in speech processing, we found no public test of whether
audio-native language models can tell that a speaker is a child under 13 or
under 18. 

## What we did

We drew speech samples from two corpora of American English children’s speech:

- **Kids VA** is the [Kids Voice Assistant corpus](https://www.sciencedirect.com/science/article/abs/pii/S2212868922000587) and it features speech from 28 5-10 year-olds.
- **MyST** is the [My Science Tutor corpus](https://arxiv.org/abs/2309.13347). It contains classroom recordings of students in grades 3 to 5, who are roughly 8 to 11 years old. We sampled 250 of them at random.

We tested three audio-capable language models, i.e. models that don't have to 
turn speech audio into text for further processing. Those were Gemini 2.5 Flash, 
GPT-4o Audio Preview with its successor gpt-audio, and Nemotron 3 Nano Omni. The
speech samples drawn from these two resources were inserted into prompts 
with questions about whether speakers were over 13 and over 18. A third question
asks for an age in years with a confidence rating and a one-sentence rationale. 
Each question was asked three times with no system prompt. The full prompt text 
is in Section 2 of the paper and in each study notebook.

Every speaker is under 13, so every positive response to the over-13 and over-18 
threshold questions is incorrect. Our design measures false positives only. To 
make our analysis more tractable, and because no quality threshold is specified 
in existing regulations, we set two reference points of our own. We consider a 
false-positive rate above 5% on the over-13 question to be incompatible with 
effective age gating. We consider any rate above 1% on the over-18 question 
to be a failure.

Five studies varied the speech type, the clip length, the background noise, and
the input modality.

| Study | Corpus | Children | Clip | What varies | Modality |
|---|---|---|---|---|---|
| 1A | Kids VA | 28, ages 5 to 10 | About 6 s | Scripted speech, baseline | Audio |
| 1B | Kids VA | 28, ages 5 to 10 | About 6 s | Spontaneous speech | Audio |
| 1C | Kids VA | 28, ages 5 to 10 | About 6 s | Scripted speech under nine background-noise conditions | Audio |
| 2A | MyST | 250, grades 3 to 5 | About 30 s | Conversational speech, baseline | Audio and text |
| 2B | MyST | 250, grades 3 to 5 | About 6 s | The 2A speech truncated | Audio and text |

The text conditions gave the model a verbatim transcript of the same speech in
place of the audio.

The models queried are not dedicated speech models. Rather they are general-purpose models 
that are increasingly being adopted in a wide variety of child-facing AI deployments. To get
a sense for how the generalist models compare with a dedicated speech-age model on age
prediction for children, we presented the 56 Kids VA clips of Studies 1A and 1B to the 24-layer
wav2vec 2.0 age and gender model released by audEERING.

## What we found

At a very high level, there is some usable age signal in the clips, but not enough.

- Models classified children as over 13 on 2.5% to 42.5% of audio responses.
- Only 3 of 12 model and condition combinations met the 5% reference point on the over-13 question.
- Every model that answered placed some 5 to 11 year olds over 18. Ten of 12 model ~ study combinations exceeded the 1% reference point on the over-18 question.
- The models' responses are inconsistent. Gemini’s over-13 rate under noise was 6.1% per response, yet 67.9% of children were misclassified on at least one of their 27 queries.
- There is more signal in audio than in transcripts. From transcripts, Gemini and GPT-4o each placed more than half of the MyST children over 13. From the audio of the same speech the rates were 11.6% and 2.5%, respectively.
- The dedicated speech-age model did worse than the audio-capable large language models. It placed 42.9% of scripted clips and 64.3% of spontaneous clips at 13 or older.

## The three failure modes

While measuring those rates, we hit three failures. None produced a warning, a
status code or a change in output format. Each would have passed standard
monitoring.

1. **The pipeline corrupted the input.** Version 0.4.0 of the google-genai client library encoded inline audio in base64 twice. Gemini received an unreadable string of characters in place of each recording, but it answered anyway. Its mean age estimate for children with a true mean age of 7.4 was 36.8 years. A capture of the outgoing request confirmed the defect. We subsequently upgraded the library to version 2.0.0, at which point this behavior vanished.
2. **The model changed underneath the tool.** OpenAI retired GPT-4o Audio Preview in May 2026 and replaced it with gpt-audio. Our requests placed the audio before the text. Preview had answered requests in that order. gpt-audio ignored the audio in that order. The server still received the audio and counted its tokens. No error or warning came back. On Kids VA, gpt-audio refused every threshold question or asked for the audio it had already been sent. A control with an adult voice on November 22, 2026 found the cause. Moving the text before the audio was the only change needed. In a rerun of Studies 1A and 1B in that order, gpt-audio gave 31 valid answers to the over-13 question and 19 to the over-18 question. Every one of those answers was no. It gave an age estimate for 40 of the 56 children.
3. **The model’s answers disagreed with each other.** A speaker over 18 is necessarily over 13. Of 9,687 paired answers, 329 indicated that speakers were both under 13 and over 18 *for the same speech sample*. Nemotron also answered that 39.5% of spontaneous Kids VA clips came from speakers over 13, even though its own numeric estimates put only 8.9% of them there.

## What we recommend

The paper draws five practices from these failures:

- Evaluate the whole pipeline and not only the model.
- Report library versions, endpoint identifiers and collection dates.
- Treat a safety validation as perishable, since the provider can replace an endpoint without notice.
- Ask redundant questions and state how refusals are counted. Pairing the over-13 answer with a numeric estimate cut the audio error rate below 3% in ten of twelve cases. The treatment of refusals alone moved one rate from 14.7% to 80.1%.
- Inspect what an inference platform keeps.

## Limitations

These studies measure false positives only. No speaker is above either
threshold, and the oldest children are about 11. Both corpora are American
English. The baseline ran only on Kids VA. Each judgment came from one prompt
about one clip, so the results describe one component and not a full age
assurance system. Section 9 of the paper gives the complete list.

## Repository layout

| Folder | Contents |
|---|---|
| `Study 1A - Kids VA Scripted Speech (google-genai 2.0.0)` | Collection notebook, coded outcomes, parsed estimates, figures, pinned environment, and the `google_genai regression` folder documenting the double-encoding defect at the transport layer |
| `Study 1B - Kids VA Spontaneous Speech (google-genai 2.0.0)` | Collection notebook, stimulus builder, coded outcomes, parsed estimates, figures |
| `Study 1C - Kids VA Scripted Speech w- Background Noise Injection (google-genai 2.0.0)` | Noise injection script, collection notebook, coded outcomes, parsed estimates |
| `Study 2A - MyST 30-Second Speech (google-genai 2.0.0)` | Student selection script, stimulus builder, collection script, analysis notebook, coded outcomes, parsed estimates, figures |
| `Study 2B - MyST 6-Second Speech (google-genai 2.0.0)` | Collection script, GPT-4o retry script, analysis notebook, coded outcomes, parsed estimates, figures |
| `Dedicated speech-age baseline` | The baseline model’s per-clip estimates on the Kids VA clips of Studies 1A and 1B and the comparison with the language models’ numeric estimates |
| `Gating rule and refusal sensitivity` | The script that reproduces every threshold table in the paper, its output CSVs, and the table rows as LaTeX |
| `Binary question analysis` | An earlier consistency check between threshold answers and numeric estimates, with its figures |
| `gpt-audio control and rerun (2026-11-22)` | The transport control on gpt-audio, the coded text-first rerun of Studies 1A and 1B and the scripts that ran them |
| `First collection phase (superseded, google-genai 0.4.0)` | Runs under the defective client library. They are kept as evidence of the defect and as the source of the GPT-4o Audio Preview refusal rates on Kids VA |

Each study folder has its own README.

## What the response files contain

Every results file holds a coded outcome for each response and not the
response text. The models often quoted the children’s words in their answers.
The MyST license from the Linguistic Data Consortium prohibits redistribution
of corpus content. The Kids VA authors asked that none of their corpus be
distributed with this code. We therefore replaced the text with codes. Every
rate in the paper can be recomputed from the codes.

The columns work as follows:

- `outcome` holds `yes`, `no`, `refusal` or `unclear` for a threshold question and `answered` for the age question
- `outcome` in the Kids VA files also takes `no_age` when a response to the age question held no age
- Errors are coded as an `error` class with its cause
- `pred_age` holds the parsed numeric estimate
- `refusal_check` appears in the Kids VA files only and takes the values `refused`, `no_audio`, `error` and `other`

The `refusal_check` column sorts each response by its wording. A response
with refusal language is `refused`. A response that asks for the audio is
`no_audio`. The column does not show whether the audio arrived. The paper
uses it to compare refusals across the two OpenAI endpoints. The gpt-audio
rows in the Study 1A and 1B folders come from requests that placed the audio
before the text. Those rows record what gpt-audio said while it ignored the
audio. The text-first rerun carries the same column in
`gpt-audio control and rerun (2026-11-22)/rerun_coded.csv`.

## Getting the corpora

This repository redistributes no audio and no transcripts. Each corpus comes
under its distributor’s own terms.

- **Kids VA** can be requested from its authors through Kim et al. 2022, Examining voice assistants in the context of children’s speech, International Journal of Child-Computer Interaction 34.
- **MyST** is available from the Linguistic Data Consortium as LDC2021S05.

The stimulus construction scripts rebuild the stimuli from licensed copies of
the corpora. Manifests in the Kids VA folders pin the exact clips used. The
script that drew the 250 MyST students is in the Study 2A `Code` folder. The
list of selected students holds corpus identifiers and file paths, so it is
not distributed here.

## Reproducing the analyses

Every table in the paper recomputes from the committed CSV files without
contacting any API.

`Gating rule and refusal sensitivity/gating_and_refusals.py` reproduces every
threshold table in the paper. It reads the coded outcomes for every study. It
keeps each student’s first presentation in the Study 2B GPT-4o audio
condition. It writes the rates under three treatments of excluded responses,
the per-child rates under both rules, the gating-rule analysis and
`table_rows.tex`. For gpt-audio on Studies 1A and 1B it uses the text-first
rerun. It drops gpt-audio on Study 1C, which was not rerun. Confidence
intervals come from a percentile bootstrap that
resamples speakers with replacement over 10,000 draws with a fixed seed. The
script runs in a few seconds.

The study notebooks reproduce the numeric estimates, the calibration figures
and the audio-versus-text comparisons. The Kids VA collection notebooks and
the binary analysis notebook were run on the response text. Their saved
outputs are cleared where they quoted the children. They do not rerun from
the coded files.

`Dedicated speech-age baseline/compare_llm_numeric.py` reproduces the baseline
comparison table from the committed CSV files. Rerunning the baseline model
itself needs the Kids VA audio and the model weights. That folder’s README
gives the model revision and the steps.

The Kids VA collection runs used the pinned environment in
`Study 1A - Kids VA Scripted Speech (google-genai 2.0.0)/requirements_pinned.txt`.
The MyST collection ran under google-genai 2.0.0. Where a notebook can
collect fresh data, API keys are entered at runtime and never written to
disk.

## Ethics and data handling

This work analyzed two existing corpora and collected no new data from
children. Both corpora are deidentified and we made no attempt to reidentify
any speaker. San José State University and the University of Memphis each
determined that the work does not require institutional review board
oversight. We used no adversarial prompting or refusal bypass, and we report
refusals as findings.

Reaching one model required uploading each clip to an inference platform’s
object storage. The platform retained those uploads by default and nothing in
its responses said so. All uploaded objects were deleted and the platform
confirmed the deletion in writing. The paper’s ethics statement in Appendix A
describes the episode and its remediation in full.

## Citation

```bibtex
@misc{stewart2026three,
  title  = {Three Invisible Failure Modes in Speech-Based Age Estimation for Child Safety},
  author = {Stewart, Christopher M. and Kraus, Kelsey and Casal, J. Elliott and Brown, Geoffrey L.},
  year   = {2026},
  note   = {Poster at the NeurIPS 2026 Workshop on Child Safety in AI},
  eprint = {2611.18971},
  archivePrefix = {arXiv}
}
```

## License

MIT. See `LICENSE`.
