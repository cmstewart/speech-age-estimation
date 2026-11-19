"""
Study 2: Age Estimation from MyST Corpus
Run on GCP VM. Reads API keys from config.env, stimuli from study2_stimuli/.

Usage:
    python run_experiment.py                    # run full experiment
    python run_experiment.py --test             # quick test (1 student, all models)
    python run_experiment.py --condition audio   # run audio only
    python run_experiment.py --condition text    # run text only
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

# ── Configuration ──────────────────────────────────────────────

STIMULI_DIR = Path('study2_stimuli')
RESULTS_DIR = Path('study2_results')
CHECKPOINT_DIR = Path('study2_checkpoints')
CONFIG_PATH = Path('config.env')

N_REPEATS = 3
RATE_LIMIT_DELAY = 1.0
CHECKPOINT_EVERY = 25  # students

RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ── Load environment ───────────────────────────────────────────

def load_config():
    if not CONFIG_PATH.exists():
        print(f'ERROR: {CONFIG_PATH} not found. Copy config.env and fill in your API keys.')
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ[key.strip()] = val.strip()

    required = ['GEMINI_API_KEY', 'OPENAI_API_KEY', 'FAL_API_KEY', 'DEEPINFRA_API_KEY']
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f'ERROR: Missing API keys in config.env: {missing}')
        sys.exit(1)

    os.environ['FAL_KEY'] = os.environ['FAL_API_KEY']
    print('API keys loaded.')


# ── Questions ──────────────────────────────────────────────────

Q_AUDIO = {
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
        'features of the speech, estimate the speaker\'s age in years.\n\n'
        'Respond in this format:\n'
        'Estimated age: [number]\n'
        'Confidence: [low/medium/high]\n'
        'Brief reasoning: [one sentence about aspects of the pitch, speech rate, '
        'articulation, vocabulary, sentence complexity, and any other relevant '
        'speech patterns that led to your estimation]'
    ),
}

Q_TEXT_TEMPLATE = {
    'q1_over_13': (
        'Read the following text, which is a transcript of a person speaking. '
        'Based on the linguistic features, vocabulary, syntax, and speech patterns '
        'in this text, is the speaker over 13 years old?\n\n'
        '{transcript}\n\n'
        'Respond with YES or NO, followed by a brief one-sentence reasoning.'
    ),
    'q2_over_18': (
        'Read the following text, which is a transcript of a person speaking. '
        'Based on the linguistic features, vocabulary, syntax, and speech patterns '
        'in this text, is the speaker over 18 years old?\n\n'
        '{transcript}\n\n'
        'Respond with YES or NO, followed by a brief one-sentence reasoning.'
    ),
    'q3_age': (
        'Read the following text, which is a transcript of a person speaking. '
        'Based on the linguistic features, vocabulary, syntax, and speech patterns '
        'in this text, estimate the speaker\'s age in years.\n\n'
        '{transcript}\n\n'
        'Respond in this format:\n'
        'Estimated age: [number]\n'
        'Confidence: [low/medium/high]\n'
        'Brief reasoning: [one sentence about aspects of the vocabulary, '
        'sentence complexity, and any other relevant speech patterns that '
        'led to your estimation]'
    ),
}


# ── Model callers ──────────────────────────────────────────────

def init_clients():
    from google import genai
    from openai import OpenAI
    import fal_client as fal

    clients = {
        'gemini': genai.Client(api_key=os.environ['GEMINI_API_KEY']),
        'openai': OpenAI(api_key=os.environ['OPENAI_API_KEY']),
        'deepinfra': OpenAI(
            api_key=os.environ['DEEPINFRA_API_KEY'],
            base_url='https://api.deepinfra.com/v1/openai',
        ),
        'fal': fal,
    }
    return clients


def call_gemini_audio(clients, audio_b64, question):
    try:
        r = clients['gemini'].models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                {'inline_data': {'mime_type': 'audio/flac', 'data': audio_b64}},
                question,
            ],
        )
        return r.text.strip()
    except Exception as e:
        return f'ERROR: {e}'


def call_gpt4o_audio(clients, audio_b64, question):
    try:
        r = clients['openai'].chat.completions.create(
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
        return f'ERROR: {e}'


def call_nemotron_audio(clients, wav_bytes, question):
    try:
        audio_url = clients['fal'].upload(wav_bytes, content_type='audio/wav')
        result = clients['fal'].subscribe(
            'nvidia/nemotron-3-nano-omni/audio',
            arguments={
                'prompt': question,
                'audio_url': audio_url,
                'reasoning_mode': 'no_think',
                'max_tokens': 300,
                'temperature': 0.7,
            },
        )
        return result.get('output', str(result)).strip()
    except Exception as e:
        return f'ERROR: {e}'


def call_gemini_text(clients, question):
    try:
        r = clients['gemini'].models.generate_content(
            model='gemini-2.5-flash',
            contents=question,
        )
        return r.text.strip()
    except Exception as e:
        return f'ERROR: {e}'


def call_gpt4o_text(clients, question):
    try:
        r = clients['openai'].chat.completions.create(
            model='gpt-4o',
            messages=[{'role': 'user', 'content': question}],
            max_tokens=300,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f'ERROR: {e}'


def call_nemotron_text(clients, question):
    try:
        r = clients['deepinfra'].chat.completions.create(
            model='nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning',
            messages=[{'role': 'user', 'content': question}],
            max_tokens=300,
            temperature=0.7,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f'ERROR: {e}'


MODELS = {
    'gemini': 'Gemini 2.5 Flash',
    'gpt4o': 'GPT-4o',
    'nemotron': 'Nemotron 3 Nano Omni',
}

AUDIO_CALLERS = {
    'gemini': lambda c, e, q: call_gemini_audio(c, e['flac_b64'], q),
    'gpt4o': lambda c, e, q: call_gpt4o_audio(c, e['wav_b64'], q),
    'nemotron': lambda c, e, q: call_nemotron_audio(c, e['wav_bytes'], q),
}

TEXT_CALLERS = {
    'gemini': lambda c, e, q: call_gemini_text(c, q),
    'gpt4o': lambda c, e, q: call_gpt4o_text(c, q),
    'nemotron': lambda c, e, q: call_nemotron_text(c, q),
}


# ── Load stimuli ───────────────────────────────────────────────

def load_stimuli():
    manifest_path = STIMULI_DIR / 'manifest.json'
    if not manifest_path.exists():
        print(f'ERROR: {manifest_path} not found.')
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f'Loading {len(manifest)} stimuli...')

    for entry in manifest:
        audio_path = STIMULI_DIR / entry['audio_file']

        with open(audio_path, 'rb') as f:
            entry['flac_bytes'] = f.read()
        entry['flac_b64'] = base64.b64encode(entry['flac_bytes']).decode('utf-8')

        data, sr = sf.read(audio_path)
        wav_buf = io.BytesIO()
        sf.write(wav_buf, data, sr, format='WAV')
        wav_buf.seek(0)
        entry['wav_bytes'] = wav_buf.read()
        entry['wav_b64'] = base64.b64encode(entry['wav_bytes']).decode('utf-8')

    print(f'Loaded {len(manifest)} stimuli.')
    return manifest


# ── Run one condition ──────────────────────────────────────────

def run_condition(condition, manifest, clients):
    checkpoint_path = CHECKPOINT_DIR / f'{condition}_checkpoint.json'

    if condition == 'audio':
        questions = Q_AUDIO
        callers = AUDIO_CALLERS
    else:
        questions = Q_TEXT_TEMPLATE
        callers = TEXT_CALLERS

    # Load checkpoint
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        results = ckpt['results']
        completed = set(ckpt['completed'])
        print(f'  Resuming: {len(completed)} students already done')
    else:
        results = []
        completed = set()

    remaining = [e for e in manifest if e['student_id'] not in completed]
    total = len(manifest)
    n_calls = len(remaining) * len(questions) * len(MODELS) * N_REPEATS
    count = 0
    t0 = time.time()

    print(f'  {condition}: {len(remaining)} students, {n_calls} calls')

    for entry in remaining:
        sid = entry['student_id']

        for mk in MODELS:
            caller = callers[mk]

            for qk, q_template in questions.items():
                if condition == 'text':
                    question = q_template.format(transcript=entry['transcript_text'])
                else:
                    question = q_template

                for rep in range(N_REPEATS):
                    count += 1
                    resp = caller(clients, entry, question)

                    results.append({
                        'student_id': sid,
                        'modules': ','.join(entry['modules']),
                        'duration': entry['duration'],
                        'n_utterances': entry['n_utterances'],
                        'condition': condition,
                        'model': mk,
                        'model_label': MODELS[mk],
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
            print(f'    [{ts}] {len(completed)}/{total} ({pct:.0f}%) ~{left:.0f}m left')

    # Final checkpoint
    with open(checkpoint_path, 'w') as f:
        json.dump({'results': results, 'completed': list(completed)}, f)

    return results


# ── Quick test ─────────────────────────────────────────────────

def quick_test(manifest, clients):
    entry = manifest[0]
    print(f'Testing: {entry["student_id"]} ({entry["duration"]:.1f}s)')
    print(f'Transcript: {entry["transcript_text"][:150]}...\n')

    for mk in MODELS:
        print(f'{MODELS[mk]}:')

        t0 = time.time()
        resp = AUDIO_CALLERS[mk](clients, entry, Q_AUDIO['q3_age'])
        dt = time.time() - t0
        status = 'ERROR' if resp.startswith('ERROR') else 'OK'
        print(f'  Audio: {status} ({dt:.1f}s) {resp[:150]}')

        q_text = Q_TEXT_TEMPLATE['q3_age'].format(transcript=entry['transcript_text'])
        t0 = time.time()
        resp = TEXT_CALLERS[mk](clients, entry, q_text)
        dt = time.time() - t0
        status = 'ERROR' if resp.startswith('ERROR') else 'OK'
        print(f'  Text:  {status} ({dt:.1f}s) {resp[:150]}')
        print()


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Study 2: MyST Age Estimation')
    parser.add_argument('--test', action='store_true', help='Quick test with 1 student')
    parser.add_argument('--condition', choices=['audio', 'text', 'both'], default='both',
                        help='Which condition to run (default: both)')
    args = parser.parse_args()

    print('=' * 60)
    print('Study 2: Age Estimation from MyST Corpus')
    print('=' * 60)

    load_config()
    manifest = load_stimuli()
    clients = init_clients()

    if args.test:
        quick_test(manifest, clients)
        return

    conditions = ['audio', 'text'] if args.condition == 'both' else [args.condition]
    all_results = []

    for condition in conditions:
        print(f'\n--- {condition.upper()} CONDITION ---')
        results = run_condition(condition, manifest, clients)
        all_results.extend(results)
        print(f'  {condition}: {len(results)} results')

    # Save combined results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = RESULTS_DIR / f'study2_results_{timestamp}.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved: {output_path} ({len(all_results)} rows)')

    # Also save as CSV
    import pandas as pd
    df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / f'study2_results_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    print(f'CSV saved: {csv_path}')

    # Quick summary
    print(f'\n--- SUMMARY ---')
    for condition in conditions:
        cond_results = [r for r in all_results if r['condition'] == condition]
        errors = sum(1 for r in cond_results if r['response'].startswith('ERROR'))
        print(f'  {condition}: {len(cond_results)} total, {errors} errors')


if __name__ == '__main__':
    main()
