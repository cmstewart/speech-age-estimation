"""
Re-run GPT-4o audio condition for Study 2b with rate limit handling.
Replaces the failed GPT-4o audio results from the main experiment.

Usage:
    python rerun_gpt4o_audio.py --test    # test with 1 student
    python rerun_gpt4o_audio.py           # full run
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

STIMULI_DIR = Path('study2b_stimuli')
RESULTS_DIR = Path('study2b_results')
CHECKPOINT_DIR = Path('study2b_checkpoints')
CONFIG_PATH = Path('config.env')

N_REPEATS = 3
RATE_LIMIT_DELAY = 3.0  # 3 seconds between calls
MAX_RETRIES = 5
RETRY_WAIT = 30  # seconds to wait after a rate limit error
CHECKPOINT_EVERY = 25

RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)


def load_config():
    if not CONFIG_PATH.exists():
        print(f'ERROR: {CONFIG_PATH} not found.')
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ[key.strip()] = val.strip()
    if not os.environ.get('OPENAI_API_KEY'):
        print('ERROR: OPENAI_API_KEY not set')
        sys.exit(1)
    print('API key loaded.')


QUESTIONS = {
    'q1_over_13': (
        'Listen to this audio clip. Based on the acoustic and linguistic '
        'features of the speech, is the speaker over 13 years old?\n\n'
        'Respond with YES or NO, followed by a brief one-sentence reasoning.'
    ),
    'q2_over_18': (
        'Listen to this audio clip. Based on the acoustic and linguistic '
        'features of the speech, is the speaker over 18 years old?\n\n'
        'Respond with YES or NO, followed by a brief one-sentence reasoning.'
    ),
    'q3_age': (
        'Listen to this audio clip. Based on the acoustic and linguistic '
        "features of the speech, estimate the speaker's age in years.\n\n"
        'Respond in this format:\n'
        'Estimated age: [number]\n'
        'Confidence: [low/medium/high]\n'
        'Brief reasoning: [one sentence about aspects of the pitch, speech rate, '
        'articulation, vocabulary, sentence complexity, and any other relevant '
        'speech patterns that led to your estimation]'
    ),
}


def call_gpt4o_with_retry(client, audio_b64, question):
    for attempt in range(MAX_RETRIES):
        try:
            r = client.chat.completions.create(
                model='gpt-4o-audio-preview',
                messages=[{
                    'role': 'user',
                    'content': [
                        {'type': 'input_audio', 'input_audio': {'data': audio_b64, 'format': 'wav'}},
                        {'type': 'text', 'text': question},
                    ],
                }],
                max_tokens=300,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e)
            if '429' in err_str or 'rate limit' in err_str.lower():
                wait = RETRY_WAIT * (attempt + 1)
                print(f'      Rate limited (attempt {attempt+1}/{MAX_RETRIES}), waiting {wait}s...')
                time.sleep(wait)
            else:
                return f'ERROR: {e}'
    return 'ERROR: Max retries exceeded (rate limit)'


def load_stimuli():
    with open(STIMULI_DIR / 'manifest.json') as f:
        manifest = json.load(f)

    print(f'Loading {len(manifest)} stimuli...')
    for entry in manifest:
        audio_path = STIMULI_DIR / entry['audio_file']
        data, sr = sf.read(audio_path)
        wav_buf = io.BytesIO()
        sf.write(wav_buf, data, sr, format='WAV')
        wav_buf.seek(0)
        entry['wav_b64'] = base64.b64encode(wav_buf.read()).decode('utf-8')

    print(f'Loaded {len(manifest)} stimuli.')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    print('=' * 60)
    print('Re-run: GPT-4o Audio Condition (Study 2b)')
    print(f'Rate limit delay: {RATE_LIMIT_DELAY}s, retries: {MAX_RETRIES}')
    print('=' * 60)

    load_config()

    from openai import OpenAI
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    manifest = load_stimuli()

    if args.test:
        entry = manifest[0]
        print(f'\nTesting: {entry["student_id"]}')
        resp = call_gpt4o_with_retry(client, entry['wav_b64'], QUESTIONS['q3_age'])
        status = 'ERROR' if resp.startswith('ERROR') else 'OK'
        print(f'  {status}: {resp[:200]}')
        return

    # Checkpoint
    checkpoint_path = CHECKPOINT_DIR / 'gpt4o_rerun_checkpoint.json'
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        results = ckpt['results']
        completed = set(ckpt['completed'])
        print(f'Resuming: {len(completed)} students done')
    else:
        results = []
        completed = set()

    remaining = [e for e in manifest if e['student_id'] not in completed]
    total = len(manifest)
    n_calls = len(remaining) * len(QUESTIONS) * N_REPEATS
    count = 0
    t0 = time.time()

    print(f'{len(remaining)} students, {n_calls} calls\n')

    for entry in remaining:
        sid = entry['student_id']

        for qk, question in QUESTIONS.items():
            for rep in range(N_REPEATS):
                count += 1
                resp = call_gpt4o_with_retry(client, entry['wav_b64'], question)

                results.append({
                    'student_id': sid,
                    'modules': ','.join(entry.get('modules', [])),
                    'original_duration': entry.get('original_duration', 0),
                    'truncated_duration': entry['truncated_duration'],
                    'condition': 'audio',
                    'model': 'gpt4o',
                    'model_label': 'GPT-4o',
                    'question': qk,
                    'repeat': rep + 1,
                    'response': resp,
                })

                time.sleep(RATE_LIMIT_DELAY)

        completed.add(sid)

        if len(completed) % CHECKPOINT_EVERY == 0:
            with open(checkpoint_path, 'w') as f:
                json.dump({'results': results, 'completed': list(completed)}, f)
            elapsed = time.time() - t0
            rate = count / elapsed if elapsed > 0 else 1
            left = (n_calls - count) / rate / 60
            pct = len(completed) / total * 100
            ts = datetime.now().strftime('%H:%M:%S')
            print(f'  [{ts}] {len(completed)}/{total} ({pct:.0f}%) ~{left:.0f}m left')

    # Final save
    with open(checkpoint_path, 'w') as f:
        json.dump({'results': results, 'completed': list(completed)}, f)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = RESULTS_DIR / f'gpt4o_rerun_{timestamp}.csv'

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    errors = sum(1 for r in results if r['response'].startswith('ERROR'))
    refusals = sum(1 for r in results if 'sorry' in r['response'].lower() or "can't" in r['response'].lower())
    print(f'\nDone! {len(results)} results saved to {csv_path}')
    print(f'  Errors: {errors}, Refusals: {refusals}, Answers: {len(results) - errors - refusals}')


if __name__ == '__main__':
    main()
