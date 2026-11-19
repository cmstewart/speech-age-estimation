# Study 1A: Kids VA Scripted Speech, google-genai 2.0.0

This study tests whether audio capable language models can estimate a child's age from a short clip of scripted speech. It also documents and reproduces a client side defect in an older audio SDK that silently corrupted the audio and produced a false result in an earlier run. The corpus is 28 children between the ages of 5 and 10, and each clip is run three times per question.

## Running the notebook

1. Use Python 3.11 or newer, since the pinned packages require it.
2. Create a fresh environment and install the pinned versions from `requirements_pinned.txt`.
3. Register that environment as a Jupyter kernel and select it in the notebook.
4. Open `study1a_age_estimation_audio_version_demo.ipynb` and run it from the top.

You will enter API keys for Gemini, OpenAI, and fal.ai at runtime. The keys are never written to disk.

## Reproducing the google-genai defect

You do not need the full run to see the bug. Section 3 of the notebook builds two small throwaway environments on its own, one holding google-genai 0.4.0 and one holding 2.0.0. Section 7 then sends the same clip through both and prints a side by side table. The broken version shows a payload about one third larger that decodes to text rather than audio. The fixed version shows a payload that decodes straight to audio. That single cell is the whole demonstration, and it runs in well under a minute.

The throwaway environments live under `demo_envs` and are rebuilt automatically, so they are not tracked in this repository.

## Folder contents

- `study1a_age_estimation_audio_version_demo.ipynb` is the main notebook. It runs all conditions, demonstrates the defect, and produces the figures and tables. Its saved outputs are cleared because they quoted the children.
- `google_genai regression` holds the worker script and the captured request data behind the defect demonstration, including the wire forensics that prove the double encoding. Responses in the captured output files are replaced with coded outcomes.
- `Results` holds a coded outcome for each response, the parsed age estimates and the figures. The response text is not included.
- `study1_stimuli/manifest.json` lists the clips with their ages and durations. The audio and transcripts are not included.

## Results

### The original overestimation was a client bug, not a model failure

The first version of this study reported that Gemini judged these children to be adults. That result was an artifact of the client library rather than a property of the model. When Gemini 2.5 Flash was given stimuli packaged by an older version of the request building package, it never actually received the audio. The model was working from our text prompt plus a long meaningless string of characters.

Section 7 sends the identical request under both SDK versions and reads what each one placed in the API call. Under google-genai 0.4.0 the audio field is 472,080 characters and decodes to base64 text instead of audio. Under 2.0.0 the same field is 354,060 characters and decodes straight to FLAC. The ratio between them is 1.33, which is the signature of encoding the same bytes one extra time. On the same clip the 0.4.0 build estimated age 35 while the 2.0.0 build estimated age 6. The defect starved the model of audio and left it guessing.

### Gemini 2.5 Flash is accurate once the audio arrives intact

Under google-genai 2.0.0 Gemini produced valid age estimates for all 28 children across three repeats. The mean signed error is -0.64 years and the mean absolute error is 1.52 years. The slight negative bias is marginally significant at t = -2.06 and p = 0.049. Accuracy holds across most of the range and weakens only for the oldest children. Ages 9 and 10 are underestimated by roughly 1.5 to 2 years. The downward trend with age is present but is not statistically reliable in this sample at r = -0.23 and p = 0.24. On the binary questions Gemini correctly answered no to over 13 and over 18 for nearly every child.

### gpt-audio declines the task

The earlier study used GPT-4o Audio Preview, which has since been retired. We probed the successor models gpt-audio, gpt-audio-1.5, gpt-audio-2025-08-28, and gpt-audio-mini-2025-10-06 on a small set of clips. Across the whole family the models either refuse age estimation outright or stall by claiming no audio was provided, and the rare genuine estimate is unreliable and nonsensical. None is usable for this task.

We used the dated snapshot gpt-audio-2025-08-28 as the named condition, since a dated snapshot maps to a fixed model more reliably than a floating alias. In the full run it refused 221 of 252 calls and produced zero usable age estimates, so it does not appear in the age analysis. The earlier age correlated refusal pattern does not reproduce. Younger children were refused at least as often as older ones. The age to refusal correlation is -0.33 and is not statistically significant at p = 0.09. This shift shows how changing provider policy can alter behavior over a short window, which matters for any application that depends on these models in a classroom.

### Nemotron underestimates and compresses with age

Nemotron answered 251 of 252 calls and drew a single content policy refusal. Its mean signed error is -1.44 years and its mean absolute error is 1.61 years. The underestimate is strongly significant at t = -6.23 and p below 0.0001. The defining feature is compression. The age to error correlation is -0.71 and p is below 0.0001. Nemotron is accurate for the youngest children and underestimates the oldest by about three years. This may signal a reluctance to read children's ages accurately as they approach the COPPA threshold.

### The methodological point

These results show three distinct kinds of failure. Gemini shows how using these models in real settings depends on engineering plumbing that can introduce silent faults with real consequences, such as labeling a child an adult. GPT shows how the retirement and replacement of model versions can swing behavior, since a very similar model now refuses the task or claims the audio is missing, plausibly because a provider policy on biometric inference shifted within a single month. Nemotron shows poor performance on the task itself, underestimating ages and compressing the scale. The data sent to a model and the policy a model is trained to follow can each degrade a deployed application, and neither announces itself in the output.
