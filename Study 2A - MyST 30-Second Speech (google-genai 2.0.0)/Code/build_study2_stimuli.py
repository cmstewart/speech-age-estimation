"""
Study 2: Build audio and text stimuli directly from the MyST zip file.

No need to unzip the full corpus. This script reads files directly
from the zip and writes only the final stimuli.

Usage:
    pip install soundfile numpy
    python build_study2_stimuli.py

Inputs:
    - study2_manifest.json (250 selected students with file paths)
    - myst_child_conv_speech_LDC2021S05.zip (the corpus zip, can be on USB)

Outputs:
    - study2_stimuli/{student_id}_audio.flac
    - study2_stimuli/{student_id}_transcript.txt
    - study2_stimuli/manifest.json
"""

import json
import io
import random
import zipfile
import numpy as np
import soundfile as sf
from pathlib import Path

# ── Configuration ──
# Update this path to point to your zip file (e.g., on USB)
ZIP_PATH = Path('/Volumes/EMTEC T260C/myst_child_conv_speech_LDC2021S05.zip')
MANIFEST_PATH = Path('study2_manifest.json')
OUTPUT_DIR = Path('study2_stimuli')

TARGET_DURATION = 30.0
SILENCE_DURATION = 0.5
SAMPLE_RATE = 16000
MAX_UTTERANCES = 50

random.seed(42)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Load manifest ──
with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

print(f'Manifest: {len(manifest)} students')
print(f'Zip file: {ZIP_PATH}')
print(f'Output:   {OUTPUT_DIR}')
print()

# ── Open zip and build stimuli ──
results = []
skipped = 0

ZIP_PREFIX = 'myst_child_conv_speech/'

with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    zip_contents = set(zf.namelist())
    print(f'Zip contains {len(zip_contents):,} entries')
    print()

    for i, entry in enumerate(manifest):
        student_id = entry['student_id']
        utterances = entry['utterances']

        shuffled = list(utterances)
        random.shuffle(shuffled)

        audio_segments = []
        transcript_parts = []
        total_duration = 0.0
        used_count = 0
        errors = 0

        for utt in shuffled:
            if used_count >= MAX_UTTERANCES:
                break
            if total_duration >= TARGET_DURATION:
                break

            flac_zip_path = ZIP_PREFIX + utt['flac_path']
            trn_zip_path = ZIP_PREFIX + utt['trn_path']

            if flac_zip_path not in zip_contents or trn_zip_path not in zip_contents:
                errors += 1
                continue

            # Read transcript
            try:
                with zf.open(trn_zip_path) as tf:
                    text = tf.read().decode('utf-8', errors='replace').strip()
            except Exception as e:
                errors += 1
                continue

            # Skip silence, unintelligible, and very short utterances
            if len(text) < 2:
                continue
            if text.startswith('<') or text == '<SILENCE>':
                continue
            # Strip unintelligible markers
            text = text.replace('(())', '').strip()
            if len(text) < 2:
                continue

            # Convert from ALL CAPS to sentence case
            text = text[0] + text[1:].lower()

            # Read audio from zip into memory
            try:
                with zf.open(flac_zip_path) as af:
                    audio_bytes = af.read()
                data, sr = sf.read(io.BytesIO(audio_bytes))
            except Exception as e:
                errors += 1
                continue

            if data.ndim > 1:
                data = data.mean(axis=1)

            if sr != SAMPLE_RATE:
                errors += 1
                continue

            utt_duration = len(data) / sr

            if utt_duration < 0.3 or utt_duration > 15.0:
                continue

            if total_duration + utt_duration > TARGET_DURATION + 5.0:
                continue

            audio_segments.append(data)
            transcript_parts.append(text)
            total_duration += utt_duration
            used_count += 1

        # Check minimum duration
        if total_duration < 5.0:
            print(f'  SKIP {student_id}: only {total_duration:.1f}s of usable audio')
            skipped += 1
            continue

        # Concatenate audio with silence gaps
        silence = np.zeros(int(SILENCE_DURATION * SAMPLE_RATE))
        concatenated = []
        for j, seg in enumerate(audio_segments):
            concatenated.append(seg)
            if j < len(audio_segments) - 1:
                concatenated.append(silence)
        concatenated = np.concatenate(concatenated)
        final_duration = len(concatenated) / SAMPLE_RATE

        # Save audio
        audio_path = OUTPUT_DIR / f'{student_id}_audio.flac'
        sf.write(str(audio_path), concatenated, SAMPLE_RATE)

        # Save transcript
        transcript_text = ' '.join(transcript_parts)
        transcript_path = OUTPUT_DIR / f'{student_id}_transcript.txt'
        with open(transcript_path, 'w') as f:
            f.write(f'Student: {student_id}\n')
            f.write(f'Modules: {", ".join(entry["modules"])}\n')
            f.write(f'Utterances used: {used_count}\n')
            f.write(f'Audio duration: {final_duration:.2f}s\n')
            f.write(f'Read errors: {errors}\n')
            f.write(f'\n--- Concatenated transcript ---\n')
            f.write(transcript_text)
            f.write(f'\n\n--- Individual utterances ---\n')
            for t in transcript_parts:
                f.write(f'  - {t}\n')

        results.append({
            'student_id': student_id,
            'modules': entry['modules'],
            'partition': entry['partition'],
            'audio_file': f'{student_id}_audio.flac',
            'transcript_file': f'{student_id}_transcript.txt',
            'duration': round(final_duration, 2),
            'n_utterances': used_count,
            'total_paired_available': entry['total_paired_utterances'],
            'transcript_text': transcript_text,
        })

        if (i + 1) % 25 == 0:
            print(f'  {i+1}/{len(manifest)} students processed...')

# Save output manifest
with open(OUTPUT_DIR / 'manifest.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nDone!')
print(f'  Students processed: {len(results)}')
print(f'  Students skipped: {skipped}')
print(f'  Output directory: {OUTPUT_DIR}')

if results:
    durations = [r['duration'] for r in results]
    utt_counts = [r['n_utterances'] for r in results]
    print(f'\n  Duration: min={min(durations):.1f}s, max={max(durations):.1f}s, '
          f'mean={sum(durations)/len(durations):.1f}s')
    print(f'  Utterances: min={min(utt_counts)}, max={max(utt_counts)}, '
          f'mean={sum(utt_counts)/len(utt_counts):.0f}')