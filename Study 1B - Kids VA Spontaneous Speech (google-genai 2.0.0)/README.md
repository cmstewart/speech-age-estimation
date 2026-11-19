# Study 1b - Kids VA Spontaneous Speech with google-genai 2.0.0

This study tests whether audio capable language models can estimate a child's age from a short clip of *spontaneous* speech. The corpus is 28 children between the ages of 5 and 10, and each clip is run three times per question under fully pinned package versions.

## Running the notebook

The pinned requirements live one folder up, in `Study 1A - Kids VA Scripted Speech (google-genai 2.0.0)`, and the two studies share the same environment. Install those pinned versions into a fresh Python 3.11 or newer environment, register it as a Jupyter kernel, then open `study1b_age_estimation_pinned.ipynb` and run it from the top. The first cell verifies that every package version matches. You enter API keys for Gemini, OpenAI, and fal.ai at runtime and they are never written to disk.

The requirements file is at `../Study 1A - Kids VA Scripted Speech (google-genai 2.0.0)/requirements_pinned.txt`.

## Folder contents

- `study1b_age_estimation_pinned.ipynb` is the main notebook. It runs all three models and produces the figures and tables. Its saved outputs are cleared because they quoted the children.
- The `study1b_pinned_*.csv` files hold a coded outcome for each response and the parsed age estimates. The response text is not included.
- The `study1b_pinned_*.png` files hold the figures.
- `manifest.json` lists the clips with their ages and durations.
- The audio stimuli and their transcripts are not included. The corpus can be requested from its authors.

## Results

Each child was asked whether the speaker is over 13, whether the speaker is over 18, and to estimate the speaker's age.

### Gemini 2.5 Flash is accurate

In this study, Gemini produced valid estimates for all 28 children and refused nothing. Its mean signed error is -0.48 years and its mean absolute error is 1.50 years. The small negative bias is not statistically significant in this sample.

### gpt-audio declines the task

gpt-audio refused about 75 percent of the age questions and produced no usable estimates, so it does not appear in the age analysis. This is in keeping with Study 1A with scripted speech.

### Nemotron overestimates

Nemotron answered across the full age range but overestimated by about 2.6 years on average. Its mean absolute error is near 3.6 years. The overestimation is significant and is largest in the middle of the range. The oldest children at ages 9 and 10 are estimated almost exactly right. A few clips drew very large overestimates, and twelve calls failed a content filter.

### Scripted versus spontaneous

This reverses what Nemotron did on scripted speech. There it underestimated and compressed the scale and read the oldest children about three years young. That pattern suggested a possible reluctance to age children near the COPPA threshold. Here the bias flips to overestimation and the oldest children are read accurately, so that reluctance does not hold across speech style.