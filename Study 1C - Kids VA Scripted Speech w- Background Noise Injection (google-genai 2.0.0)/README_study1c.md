# Study 1c: Kids VA Speech with Injected Background Noise

Real classroom audio is noisy, so a model that estimates a child's age cleanly in a quiet recording may not hold up where it would actually be deployed. This study tests that directly. It adds controlled background noise to clean Kids VA recordings and asks whether each model's age estimates survive the degradation, and which kinds of noise pull them off the most. The corpus is 28 children aged 5 to 10, run across three models and nine acoustic conditions, with three repeats per question.

## Headline finding

Background noise does not meaningfully degrade age estimation for the models that perform the task. Both Gemini 2.5 Flash and Nemotron 3 Nano Omni stayed within about a year of their own clean estimates across all nine conditions, and most conditions were statistically indistinguishable from clean in paired tests. The OpenAI model refused the task on nearly every clip regardless of noise, so it produced no usable estimates.

## How we got here

The study began as a different question. An earlier run appeared to show Gemini judging the clean studio recordings as adults, near 36 years, and the original aim was to find which kind of noise corrected that overestimation. The overestimation turned out to be a client library defect that mangled the audio before it reached the model, not an acoustic effect. Once the defect was fixed, there was no overestimation left to correct, so the study became a straightforward robustness test. Does realistic background noise move accurate age estimates off the mark. The results here are the corrected rerun on the fixed setup, across all three models.

## Design

Three families of noise were injected into the clean stimuli, with a same-session clean baseline run in the same batch. All stimuli were loudness normalized so volume is not a confound. Voice-free controls are white and pink noise. Adult-voiced ambience is coffeeshop and downtown. Kid-coded ambience is classroom. The ambience conditions run at two noise levels, 10 and 20 dB SNR, where the lower level is the harsher and more classroom-like.

## Results

Gemini estimates ages accurately and holds up under noise. Its estimates stayed close to the true range in every condition, and where noise shifted them at all it nudged them slightly upward toward older ages, never by more than about a year. Only two conditions, white noise and the louder coffeeshop setting, shifted significantly from clean, and both shifts were small.

Nemotron is also robust to noise, with no condition differing significantly from its clean baseline. Its estimates drifted slightly downward under noise rather than upward. It carries more scatter than Gemini and labels children as over 13 somewhat more often, so it is the noisier of the two even though noise itself does not break it.

gpt-audio refused between 90 and 96 percent of calls. Its refusal rate was flat across every condition. These requests placed the audio part before the text part. A control on 2026-11-22 showed that gpt-audio ignores the audio in that order. The model did not use the audio, so the flat rate says nothing about noise. Study 1C was not rerun. gpt-audio is excluded from every rate for this study. The control is in `../gpt-audio control and rerun (2026-11-22)`.

The practical takeaway is reassuring for deployment. The models that attempt the task estimate children's ages accurately, and that accuracy survives the kind of background a classroom would add.

## Files

- `study1c_noise_injection_pinned.ipynb` is the analysis notebook. It runs all three models across every condition and produces the tables and figure. Its saved outputs are cleared because they quoted the children.
- `study1c_stimuli_noise` holds the manifest for the full grid of 28 children across nine conditions. The audio is not included.
- The `study1c_*.csv` files hold a coded outcome for each response and the parsed age estimates. The response text is not included.
- The pinned package versions are in `../Study 1A - Kids VA Scripted Speech (google-genai 2.0.0)/requirements_pinned.txt`.

## Caveat

The classroom condition contained faint background child voices, so it is not a pure test of classroom ambience. This does not change the conclusion, since the condition produced no meaningful shift in either model.

## Footnote on the corrected setup

Because the study includes a clean condition run in the same batch, it doubles as a check on the fix. Gemini's clean estimates land near the true ages rather than near 36, which confirms the audio now reaches the model intact. The earlier adult-range estimates were the client library defect and not a property of the recordings.
