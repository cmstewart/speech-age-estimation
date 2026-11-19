"""
Build spontaneous speech stimuli from Kids VA Task 1.

For each child, concatenates their longest Task 1 utterances (child-only)
until matching the duration of their Study 1 scripted stimulus.
Prioritizes longer utterances to maximize developmental signal.

Usage:
    pip install soundfile numpy pandas openpyxl
    python build_task1_stimuli.py
"""

import json
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from glob import glob

# ── Configuration ──
STUDY1_AUDIO = Path('/Volumes/EMTEC T260C/children-assistant-corpus/study1_audio')
TASK1_DIR = Path('/Volumes/EMTEC T260C/children-assistant-corpus/Task1')
TASK1_TRANSCRIPTION = Path('Task_1-Transcription.xlsx')
OUTPUT_DIR = Path('followup_task1_spontaneous')

SILENCE_DURATION = 0.5
MIN_UTT_LENGTH = 0.3  # seconds

OUTPUT_DIR.mkdir(exist_ok=True)

# Age and gender metadata
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


# ── Step 1: Get target durations from Study 1 ──

print('Step 1: Reading Study 1 durations...')
target_durations = {}
for child_dir in sorted(STUDY1_AUDIO.iterdir()):
    if not child_dir.is_dir() or child_dir.name.startswith('.'):
        continue
    child = child_dir.name
    total_dur = 0.0
    for flac_file in child_dir.glob('*.flac'):
        if flac_file.name.startswith('.'):
            continue
        data, sr = sf.read(flac_file)
        total_dur += len(data) / sr
    total_dur += SILENCE_DURATION * 2  # gaps between 3 utterances
    target_durations[child] = total_dur

print(f'  {len(target_durations)} children, durations {min(target_durations.values()):.1f}s - {max(target_durations.values()):.1f}s')


# ── Step 2: Load Task 1 transcriptions ──

print('\nStep 2: Loading Task 1 transcriptions...')
t1 = pd.read_excel(TASK1_TRANSCRIPTION)
t1['uid'] = t1['User ID'].str.lower()
t1['fname'] = t1['File Name'].str.strip()
t1['fname_clean'] = t1['fname'].str.replace(r'\s*\[.*\]', '', regex=True).str.strip()
t1['is_child'] = ~t1['fname_clean'].str.contains('-[ab]', regex=True, na=False)

text_col = 'Child Responses Transcription (non-fonetic)'
t1['text'] = t1[text_col].astype(str).str.strip()
t1['text_len'] = t1['text'].str.len()

child_utts = t1[t1['is_child'] & (t1['text_len'] > 5)].copy()
print(f'  {len(child_utts)} usable child utterances across {child_utts["uid"].nunique()} children')


# ── Step 3: Build stimuli ──

print('\nStep 3: Building stimuli...')

manifest = []
skipped = 0

for child in sorted(target_durations.keys()):
    target = target_durations[child]
    meta = CHILD_META[child]
    utts = child_utts[child_utts['uid'] == child].copy()

    if len(utts) == 0:
        print(f'  SKIP {child}: no Task 1 utterances')
        skipped += 1
        continue

    # Find audio directory
    possible_dirs = [
        TASK1_DIR / 'Audios' / child,
        TASK1_DIR / 'Audios' / child / 'Final_files',
        TASK1_DIR / child,
    ]
    audio_dir = None
    for d in possible_dirs:
        if d.exists():
            audio_dir = d
            break

    if audio_dir is None:
        print(f'  SKIP {child}: audio directory not found')
        skipped += 1
        continue

    # Read all audio files and pair with transcriptions
    utt_data = []
    for _, row in utts.iterrows():
        fname = row['fname_clean']
        if '/' in fname:
            fname = fname.split('/')[-1]

        # Search for the audio file (transcription has no extension)
        audio_path = None
        for ext in ['.wav', '.flac']:
            candidate = audio_dir / (fname + ext)
            if candidate.exists():
                audio_path = candidate
                break

        # Fallback: search recursively
        if audio_path is None:
            for ext in ['.wav', '.flac']:
                matches = list(audio_dir.rglob(fname + ext))
                if matches:
                    audio_path = matches[0]
                    break
        if audio_path is None:
            matches = list(audio_dir.rglob(fname))
            if matches:
                audio_path = matches[0]
        if audio_path is None:
            continue

        try:
            data, sr = sf.read(audio_path)
        except Exception:
            continue

        if data.ndim > 1:
            data = data.mean(axis=1)

        utt_dur = len(data) / sr
        if utt_dur < MIN_UTT_LENGTH or utt_dur > 15.0:
            continue

        utt_data.append({
            'audio': data,
            'sr': sr,
            'duration': utt_dur,
            'text': row['text'],
            'text_len': row['text_len'],
            'utt_type': row.get('Type of Utterance', ''),
        })

    if len(utt_data) == 0:
        print(f'  SKIP {child}: no readable audio files')
        skipped += 1
        continue

    # Sort by text length (longest first) to prioritize developmental speech
    utt_data.sort(key=lambda x: x['text_len'], reverse=True)

    # Concatenate until target duration
    segments = []
    transcripts = []
    total_dur = 0.0
    sample_rate = utt_data[0]['sr']

    for utt in utt_data:
        if total_dur >= target:
            break
        if total_dur + utt['duration'] > target + 2.0:
            if total_dur >= target * 0.7:
                break
            continue

        segments.append(utt['audio'])
        transcripts.append(utt['text'])
        total_dur += utt['duration']

    if len(segments) == 0 or total_dur < 2.0:
        print(f'  SKIP {child}: insufficient audio ({total_dur:.1f}s)')
        skipped += 1
        continue

    # Concatenate with silence gaps
    silence = np.zeros(int(SILENCE_DURATION * sample_rate))
    concatenated = []
    for j, seg in enumerate(segments):
        concatenated.append(seg)
        if j < len(segments) - 1:
            concatenated.append(silence)
    concatenated = np.concatenate(concatenated)

    # Trim if over target
    max_samples = int((target + 0.5) * sample_rate)
    if len(concatenated) > max_samples:
        concatenated = concatenated[:max_samples]

    final_dur = len(concatenated) / sample_rate
    transcript_text = ' '.join(transcripts)

    # Save audio
    out_path = OUTPUT_DIR / f'{child}_spontaneous.flac'
    sf.write(str(out_path), concatenated, sample_rate)

    # Save transcript
    transcript_path = OUTPUT_DIR / f'{child}_transcript.txt'
    with open(transcript_path, 'w') as f:
        f.write(f'Speaker: {child}\n')
        f.write(f'Age: {meta["age"]}\n')
        f.write(f'Gender: {meta["gender"]}\n')
        f.write(f'Target duration: {target:.2f}s (matched to Study 1)\n')
        f.write(f'Actual duration: {final_dur:.2f}s\n')
        f.write(f'Utterances used: {len(segments)}\n')
        f.write(f'Source: Task 1 (spontaneous requests)\n')
        f.write(f'\n--- Concatenated transcript ---\n')
        f.write(transcript_text)
        f.write(f'\n\n--- Individual utterances ---\n')
        for t in transcripts:
            f.write(f'  - {t}\n')

    manifest.append({
        'child': child,
        'age': meta['age'],
        'gender': meta['gender'],
        'target_duration': round(target, 2),
        'actual_duration': round(final_dur, 2),
        'n_utterances': len(segments),
        'audio_file': f'{child}_spontaneous.flac',
        'transcript_file': f'{child}_transcript.txt',
        'transcript_text': transcript_text,
    })

    print(f'  {child} (age {meta["age"]}): target={target:.1f}s, actual={final_dur:.1f}s, '
          f'{len(segments)} utterances')

# Save manifest
with open(OUTPUT_DIR / 'manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)


# ── Summary ──

print(f'\nDone!')
print(f'  Stimuli created: {len(manifest)}')
print(f'  Skipped: {skipped}')

if manifest:
    durs = [m['actual_duration'] for m in manifest]
    print(f'  Duration: {min(durs):.1f}s - {max(durs):.1f}s (mean {np.mean(durs):.1f}s)')

    from collections import Counter
    age_counts = Counter(m['age'] for m in manifest)
    print(f'  Age distribution:')
    for age in sorted(age_counts):
        print(f'    Age {age}: {age_counts[age]}')
