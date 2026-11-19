"""Dedicated speech-age baseline for the Kids VA studies.

Runs the audeering wav2vec2 age and gender model on the exact Kids VA clips
that the language models received in Studies 1A and 1B. Everything runs
locally. No audio leaves the machine.

The stimulus manifests ship with this repository. The audio and the model
weights do not. Pass their locations on the command line.

Usage
    python baseline_wav2vec2_age.py --model-dir PATH --audio-1a PATH --audio-1b PATH

Arguments
    --model-dir    Folder holding the downloaded model files
    --audio-1a     Folder holding the Study 1A stimulus audio
    --audio-1b     Folder holding the Study 1B stimulus audio
    --repo         Repository root. Defaults to the parent of this folder
    --manifest-1a  Study 1A manifest. Defaults to the copy in the repository
    --manifest-1b  Study 1B manifest. Defaults to the copy in the repository
    --out          Output folder. Defaults to results beside this script
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import transformers
from scipy.signal import resample_poly
from scipy.stats import pearsonr
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2PreTrainedModel

MODEL_NAME = "audeering/wav2vec2-large-robust-24-ft-age-gender"
TARGET_SR = 16000
THRESHOLDS = (13, 18)
CLASSES = ("female", "male", "child")
N_BOOT = 10000
SEED = 20261119
CONSTANT_AGE = 7.0

# Manifest locations relative to the repository root.
DEFAULT_MANIFESTS = {
    "1A": "Study 1A - Kids VA Scripted Speech (google-genai 2.0.0)/study1_stimuli/manifest.json",
    "1B": "Study 1B - Kids VA Spontaneous Speech (google-genai 2.0.0)/Stimuli/manifest.json",
}


# Model definition from the audeering model card.
class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.post_init()

    def forward(self, input_values):
        hidden = self.wav2vec2(input_values)[0]
        hidden = torch.mean(hidden, dim=1)
        age = self.age(hidden)
        gender = torch.softmax(self.gender(hidden), dim=1)
        return age, gender


def parse_args():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Dedicated speech-age baseline on Kids VA clips.")
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--audio-1a", type=Path, required=True)
    ap.add_argument("--audio-1b", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=here.parent)
    ap.add_argument("--manifest-1a", type=Path, default=None)
    ap.add_argument("--manifest-1b", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=here / "results")
    return ap.parse_args()


def resolve_studies(args):
    """Return the manifest path and audio folder for each study."""
    manifests = {"1A": args.manifest_1a, "1B": args.manifest_1b}
    audio = {"1A": args.audio_1a, "1B": args.audio_1b}
    studies = {}
    for study, relative in DEFAULT_MANIFESTS.items():
        manifest = manifests[study] or args.repo / relative
        studies[study] = {"manifest": manifest, "audio_dir": audio[study]}
    return studies


def build_jobs(studies):
    """List every clip to score and stop if a manifest or clip is missing."""
    jobs = []
    for study, spec in studies.items():
        if not spec["manifest"].exists():
            sys.exit(f"Manifest not found for Study {study} at {spec['manifest']}")
        for item in json.loads(spec["manifest"].read_text()):
            jobs.append((study, item, spec["audio_dir"] / item["audio_file"]))
    missing = [str(path) for _, _, path in jobs if not path.exists()]
    if missing:
        sys.exit(f"Missing audio files {missing}")
    return jobs


def load_model(model_dir):
    """Load the feature extractor and model and stop if any weight is missing."""
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
    model, info = AgeGenderModel.from_pretrained(model_dir, output_loading_info=True)
    report = {key: sorted(map(str, value)) for key, value in info.items()}
    print(f"Loading info {json.dumps(report)}", flush=True)
    missing = [key for key in info["missing_keys"] if not key.endswith("masked_spec_embed")]
    if missing:
        sys.exit(f"Weights did not load {missing}")
    model.eval()
    return processor, model


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_audio(path):
    """Read a clip as mono float32 at the model's sample rate."""
    signal, native_sr = sf.read(path, dtype="float32", always_2d=True)
    signal = signal.mean(axis=1)
    if native_sr != TARGET_SR:
        g = np.gcd(native_sr, TARGET_SR)
        signal = resample_poly(signal, TARGET_SR // g, native_sr // g).astype(np.float32)
    return signal, native_sr


def predict(processor, model, signal):
    """Return the age estimate in years and the three class probabilities."""
    values = processor(signal, sampling_rate=TARGET_SR)["input_values"][0]
    batch = torch.from_numpy(np.asarray(values, dtype=np.float32).reshape(1, -1))
    with torch.inference_mode():
        age, gender = model(batch)
    return 100 * float(age[0, 0]), gender[0].numpy()


def score_clip(processor, model, study, item, path):
    signal, native_sr = load_audio(path)
    est_age, probs = predict(processor, model, signal)
    return {
        "study": study,
        "child": item["child"],
        "true_age": item["age"],
        "gender": item.get("gender"),
        "audio_file": item["audio_file"],
        "audio_sha256": sha256(path),
        "native_sr": native_sr,
        "seconds": len(signal) / TARGET_SR,
        "est_age": est_age,
        "p_female": float(probs[0]),
        "p_male": float(probs[1]),
        "p_child": float(probs[2]),
        "top_class": CLASSES[int(probs.argmax())],
    }


def score_all(processor, model, jobs):
    """Score every clip and print progress with the time remaining."""
    total = len(jobs)
    rows = []
    start = time.time()
    for done, (study, item, path) in enumerate(jobs, 1):
        row = score_clip(processor, model, study, item, path)
        rows.append(row)
        eta = (time.time() - start) / done * (total - done)
        print(f"[{done}/{total}] {total - done} remaining, ETA {eta:.0f} s, "
              f"{study} {item['child']} true {item['age']} est {row['est_age']:.1f}", flush=True)
    return pd.DataFrame(rows)


def is_deterministic(processor, model, job, first_estimate):
    """Score the first clip again and compare with its first estimate."""
    _, _, path = job
    signal, _ = load_audio(path)
    again, _ = predict(processor, model, signal)
    return abs(again - first_estimate) < 1e-4


def boot_rate(flags, rng):
    flags = np.asarray(flags, dtype=float)
    idx = rng.integers(0, len(flags), size=(N_BOOT, len(flags)))
    return np.percentile(flags[idx].mean(axis=1), [2.5, 97.5])


def boot_r(est, true, rng):
    est, true = np.asarray(est), np.asarray(true)
    draws = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(est), len(est))
        if np.std(true[i]) == 0 or np.std(est[i]) == 0:
            continue
        draws.append(np.corrcoef(est[i], true[i])[0, 1])
    return np.percentile(draws, [2.5, 97.5])


def perm_p(est, true, rng, n=N_BOOT):
    est, true = np.asarray(est), np.asarray(true)
    observed = abs(np.corrcoef(est, true)[0, 1])
    hits = sum(abs(np.corrcoef(est, rng.permutation(true))[0, 1]) >= observed for _ in range(n))
    return (hits + 1) / (n + 1)


def summarize(clips, label, rng):
    """Compute rates, error and correlation for one set of clips."""
    est, true = clips["est_age"].to_numpy(), clips["true_age"].to_numpy()
    row = {"study": label, "n_clips": len(clips), "n_children": clips["child"].nunique()}
    for threshold in THRESHOLDS:
        flags = (est >= threshold).astype(float)
        lo, hi = boot_rate(flags, rng)
        row[f"over{threshold}_count"] = int(flags.sum())
        row[f"over{threshold}_pct"] = 100 * flags.mean()
        row[f"over{threshold}_ci_lo"] = 100 * lo
        row[f"over{threshold}_ci_hi"] = 100 * hi
    row["mean_est"] = est.mean()
    row["mean_true"] = true.mean()
    row["mae"] = np.abs(est - true).mean()
    row["mae_constant7"] = np.abs(CONSTANT_AGE - true).mean()
    row["r"] = pearsonr(est, true)[0]
    row["r_ci_lo"], row["r_ci_hi"] = boot_r(est, true, rng)
    row["r_perm_p"] = perm_p(est, true, rng)
    row["min_est"] = est.min()
    row["max_est"] = est.max()
    row["pct_child_argmax"] = 100 * (clips["top_class"] == "child").mean()
    row["mean_p_child"] = clips["p_child"].mean()
    return row


def summarize_all(clips, study_labels):
    """Summarize each study and then all clips together."""
    rng = np.random.default_rng(SEED)
    rows = [summarize(clips[clips["study"] == label], label, rng) for label in study_labels]
    rows.append(summarize(clips, "+".join(study_labels), rng))
    return pd.DataFrame(rows)


def read_revision(model_dir):
    """Return the Hugging Face revision recorded with the download if present."""
    for meta in (model_dir / ".cache" / "huggingface" / "download").glob("*.metadata"):
        return meta.read_text().splitlines()[0]
    return None


def build_run_info(model_dir, deterministic, runtime_seconds):
    weights = sorted(model_dir.glob("*.safetensors")) + sorted(model_dir.glob("*.bin"))
    return {
        "model": MODEL_NAME,
        "model_revision": read_revision(model_dir),
        "weights_sha256": {w.name: sha256(w) for w in weights},
        "threshold_rule": "estimate >= 13 and estimate >= 18",
        "constant_baseline_age": CONSTANT_AGE,
        "bootstrap_draws": N_BOOT,
        "seed": SEED,
        "deterministic_rerun_matches": bool(deterministic),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "runtime_seconds": round(runtime_seconds, 1),
    }


def write_outputs(out_dir, clips, summary, run_info):
    out_dir.mkdir(parents=True, exist_ok=True)
    clips.to_csv(out_dir / "baseline_clip_estimates.csv", index=False)
    summary.to_csv(out_dir / "baseline_summary.csv", index=False)
    (out_dir / "baseline_run_info.json").write_text(json.dumps(run_info, indent=2))


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    studies = resolve_studies(args)
    jobs = build_jobs(studies)
    processor, model = load_model(args.model_dir)

    print(f"{len(jobs)} clips to score across {len(studies)} studies", flush=True)
    start = time.time()
    clips = score_all(processor, model, jobs)
    deterministic = is_deterministic(processor, model, jobs[0], clips.loc[0, "est_age"])
    summary = summarize_all(clips, list(studies))
    run_info = build_run_info(args.model_dir, deterministic, time.time() - start)
    write_outputs(args.out, clips, summary, run_info)

    pd.set_option("display.width", 200)
    print(summary.round(2).T.to_string(), flush=True)
    print(f"Deterministic rerun matches {deterministic}", flush=True)


if __name__ == "__main__":
    main()
