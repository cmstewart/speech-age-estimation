"""Convert the adult control clip to the Kids VA stimulus format.

Target, measured from the 1A and 1B stimuli: FLAC, PCM_16, 44100 Hz, mono,
6.2 s which is the median stimulus length across 1A and 1B.
Processing follows Study 1B Stimuli/build_task1_stimuli.py: mono by channel mean,
trim to length, then sf.write to FLAC at the stimulus sample rate.
The 1A builder is not in the repo.

Usage: python prepare_adult.py <ADULT_CLIP_PATH> [start_seconds]
Writes adult_control.flac in this folder
"""
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 44100
TARGET_DUR = 6.2
OUT = Path(__file__).parent / "adult_control.flac"

src = sys.argv[1]
start = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

data, sr = sf.read(src, always_2d=False)
if data.ndim > 1:
    data = data.mean(axis=1)
if sr != TARGET_SR:
    g = gcd(sr, TARGET_SR)
    data = resample_poly(data, TARGET_SR // g, sr // g)
s0 = int(start * TARGET_SR)
data = data[s0:s0 + int(TARGET_DUR * TARGET_SR)]
if len(data) < int(TARGET_DUR * TARGET_SR):
    print(f"warning: clip shorter than target, {len(data) / TARGET_SR:.2f} s")
data = np.clip(data, -1.0, 1.0)
sf.write(str(OUT), data, TARGET_SR, format="FLAC", subtype="PCM_16")

i = sf.info(str(OUT))
print(f"wrote {OUT}: {i.format} {i.subtype} {i.samplerate} Hz {i.channels} ch {i.duration:.3f} s "
      f"from source at {sr} Hz")
