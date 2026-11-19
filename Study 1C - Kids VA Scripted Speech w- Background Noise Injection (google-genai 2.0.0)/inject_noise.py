"""
Study 1d: Inject background noise into clean Kids VA stimuli.

Tests whether adding background noise to the clean, studio-quality Kids VA
recordings moves Gemini's age estimates toward the correct range, and what
TYPE of noise is corrective.

Three families of noise:
  Voice-free controls:   white, pink        (pure spectral degradation)
  Adult-voiced ambience: coffeeshop, downtown  (human activity, not kid-coded)
  Kid-coded ambience:    classroom          (suspected corrective)

Plus a same-session 'clean' condition (resampled, loudness-normalized, no noise)
so noise is compared against clean in the same run, guarding against drift.

All final mixes are loudness-normalized to a common RMS so loudness is not a
confound. Existing output files are left in place, so a re-run only generates
genuinely new conditions.

Usage:
    pip install soundfile numpy librosa
    python inject_noise.py
"""

import json
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

# ── Configuration ──
# INPUT_DIR pointed at a machine-specific absolute path in the original run.
# It is set here to the clean Study 1A stimuli within this repository.
INPUT_DIR = Path('../Study 1A - Kids VA Scripted Speech (google-genai 2.0.0)/study1_stimuli')
# The original run wrote to study1d_stimuli_noise. The identical stimulus set
# served Study 1C, and this repository stores its manifest under
# study1c_stimuli_noise, so the output path is set to match.
OUTPUT_DIR = Path('study1c_stimuli_noise')

# 'clean' adds no noise. Others are a path to a noise file or 'white'/'pink'.
NOISE_SOURCES = {
    'clean': 'clean',
    'classroom': 'background_childrens-classroom.flac',
    'coffeeshop': 'background_coffeeshop.flac',
    'downtown': 'downtown_trier_background.flac',
    'white': 'white',
    'pink': 'pink',
}

# Signal-to-noise ratios in dB. 'clean' takes none. Lower dB means more noise.
SNR_LEVELS_DB = {
    'classroom': [10, 20],
    'coffeeshop': [10, 20],
    'downtown': [10, 20],
    'white': [10],
    'pink': [10],
}

TARGET_SR = 16000   # match MyST so sample rate is held constant
TARGET_RMS = 0.05   # common loudness target for all final mixes
SEED = 42

OUTPUT_DIR.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

CHILD_META = {
    'sp104': {'age': 7, 'gender': 'M'}, 'sp150': {'age': 8, 'gender': 'F'},
    'sp169': {'age': 7, 'gender': 'F'}, 'sp232': {'age': 6, 'gender': 'M'},
    'sp243': {'age': 7, 'gender': 'F'}, 'sp262': {'age': 6, 'gender': 'F'},
    'sp316': {'age': 7, 'gender': 'F'}, 'sp358': {'age': 8, 'gender': 'M'},
    'sp367': {'age': 8, 'gender': 'M'}, 'sp376': {'age': 7, 'gender': 'M'},
    'sp396': {'age': 7, 'gender': 'F'}, 'sp413': {'age': 6, 'gender': 'M'},
    'sp440': {'age': 7, 'gender': 'F'}, 'sp463': {'age': 8, 'gender': 'M'},
    'sp464': {'age': 7, 'gender': 'F'}, 'sp492': {'age': 8, 'gender': 'F'},
    'sp519': {'age': 8, 'gender': 'M'}, 'sp567': {'age': 10, 'gender': 'F'},
    'sp590': {'age': 9, 'gender': 'M'}, 'sp593': {'age': 6, 'gender': 'F'},
    'sp605': {'age': 9, 'gender': 'F'}, 'sp611': {'age': 10, 'gender': 'F'},
    'sp626': {'age': 5, 'gender': 'M'}, 'sp795': {'age': 7, 'gender': 'M'},
    'sp846': {'age': 5, 'gender': 'F'}, 'sp965': {'age': 10, 'gender': 'M'},
    'sp979': {'age': 9, 'gender': 'F'}, 'sp993': {'age': 6, 'gender': 'M'},
}


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def normalize_loudness(x, target_rms=TARGET_RMS):
    x = x * (target_rms / rms(x))
    peak = np.max(np.abs(x))
    if peak > 0.99:
        x = x * (0.99 / peak)
    return x


def make_white(n):
    return rng.standard_normal(n)


def make_pink(n):
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]
    spectrum = spectrum / np.sqrt(freqs)
    return np.fft.irfft(spectrum, n=n)


# Cache loaded noise files so we don't reread them for every child
_noise_cache = {}


def get_noise_full(source):
    if source in _noise_cache:
        return _noise_cache[source]
    path = Path(source)
    if not path.exists():
        print(f'  WARNING: noise file not found: {path}')
        _noise_cache[source] = None
        return None
    noise, sr = sf.read(path)
    if noise.ndim > 1:
        noise = noise.mean(axis=1)
    if sr != TARGET_SR:
        noise = librosa.resample(noise, orig_sr=sr, target_sr=TARGET_SR)
    _noise_cache[source] = noise
    return noise


def load_noise(source, length):
    if source == 'white':
        return make_white(length)
    if source == 'pink':
        return make_pink(length)
    full = get_noise_full(source)
    if full is None:
        return None
    if len(full) < length:
        reps = int(np.ceil(length / len(full)))
        full = np.tile(full, reps)
    start = rng.integers(0, max(1, len(full) - length))
    return full[start:start + length]


def mix_at_snr(signal, noise, snr_db):
    sig_rms = rms(signal)
    noise_rms = rms(noise)
    target_noise_rms = sig_rms / (10 ** (snr_db / 20))
    scaled = noise * (target_noise_rms / noise_rms)
    return signal + scaled


# ── Load clean stimuli ──
with open(INPUT_DIR / 'manifest.json') as f:
    manifest = json.load(f)

print(f'Loaded {len(manifest)} clean stimuli from {INPUT_DIR}')
print(f'Conditions: {list(NOISE_SOURCES.keys())}')
print()

new_manifest = []
written = 0
skipped = 0

for entry in manifest:
    child = entry['child']
    audio_path = INPUT_DIR / entry['audio_file']
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
    duration = round(len(data) / TARGET_SR, 2)
    meta = CHILD_META.get(child, {})

    def record(cond, noise_type, snr, label, fname):
        new_manifest.append({
            'child': child, 'age': meta.get('age'), 'gender': meta.get('gender'),
            'condition': cond, 'noise_type': noise_type, 'snr_db': snr,
            'condition_label': label, 'duration': duration,
            'sample_rate': TARGET_SR, 'audio_file': fname,
        })

    for cond, source in NOISE_SOURCES.items():

        # Clean condition: no noise, one version
        if source == 'clean':
            fname = f'{child}_clean.flac'
            out_path = OUTPUT_DIR / fname
            if out_path.exists():
                skipped += 1
            else:
                sf.write(str(out_path), normalize_loudness(data), TARGET_SR)
                written += 1
            record('clean', 'clean', None, 'clean', fname)
            continue

        for snr in SNR_LEVELS_DB.get(cond, [10]):
            label = f'{cond}_snr{snr}'
            fname = f'{child}_{label}.flac'
            out_path = OUTPUT_DIR / fname

            # Dedup: skip generation if the file already exists
            if out_path.exists():
                skipped += 1
                record(cond, cond, snr, label, fname)
                continue

            noise = load_noise(source, len(data))
            if noise is None:
                continue
            mixed = normalize_loudness(mix_at_snr(data, noise, snr))
            sf.write(str(out_path), mixed, TARGET_SR)
            written += 1
            record(cond, cond, snr, label, fname)

with open(OUTPUT_DIR / 'manifest.json', 'w') as f:
    json.dump(new_manifest, f, indent=2)

print(f'Wrote {written} new files, skipped {skipped} existing files.')
print(f'Manifest now lists {len(new_manifest)} stimuli.')
print('\nConditions in manifest:')
from collections import Counter
for label, n in sorted(Counter(m['condition_label'] for m in new_manifest).items()):
    print(f'  {label}: {n} children')