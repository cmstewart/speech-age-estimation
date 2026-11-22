"""Approved follow-up variants. Adult clip only.

  python followups.py <variant> <prompt_key> [n_repeats]

variants
  v1_text_first      study request with the text part before the audio part
  v2_text_and_audio  study request with modalities text and audio
  v3_transcribe      same WAV bytes to gpt-4o-transcribe
  v4_myst_shape      the exact MyST request shape against gpt-audio, max_tokens and no modalities
prompt_key: transcript, q2_over_18 or q3_age

The key is read by the OpenAI SDK from OPENAI_API_KEY. It is never printed or saved.
Appends to followups.jsonl in this folder.
"""
import io
import sys
import time
from datetime import datetime, timezone

import httpx
import openai
from importlib.metadata import version

import study_caller as sc
from live_run import ADULT, CAPTURE, PROMPTS, _on_request, _on_response, guard, sha
import json
import base64

OUT = ADULT.parent / "followups.jsonl"
MODEL = sc.GPT_AUDIO_MODEL


def audio_part(entry):
    return {'type': 'input_audio', 'input_audio': {'data': entry['wav_b64'], 'format': 'wav'}}


def v1_text_first(client, entry, prompt):
    r = client.chat.completions.create(
        model=MODEL, modalities=['text'],
        messages=[{'role': 'user', 'content': [{'type': 'text', 'text': prompt}, audio_part(entry)]}],
        max_completion_tokens=300)
    m = r.choices[0].message
    return (m.content or getattr(m, 'refusal', None) or '').strip()


def v2_text_and_audio(client, entry, prompt):
    r = client.chat.completions.create(
        model=MODEL, modalities=['text', 'audio'], audio={'voice': 'alloy', 'format': 'wav'},
        messages=[{'role': 'user', 'content': [audio_part(entry), {'type': 'text', 'text': prompt}]}],
        max_completion_tokens=300)
    m = r.choices[0].message
    if m.content:
        return m.content.strip()
    if getattr(m, 'audio', None) is not None and getattr(m.audio, 'transcript', None):
        return m.audio.transcript.strip()
    return (getattr(m, 'refusal', None) or '').strip()


def v3_transcribe(client, entry, prompt):
    r = client.audio.transcriptions.create(
        model='gpt-4o-transcribe', file=('adult_control.wav', io.BytesIO(entry['wav_bytes']), 'audio/wav'))
    return r.text.strip()


def v4_myst_shape(client, entry, prompt):
    # Study 2A run_experiment.py call_gpt4o_audio with only the model string changed.
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{'role': 'user', 'content': [audio_part(entry), {'type': 'text', 'text': prompt}]}],
        max_tokens=300)
    return r.choices[0].message.content.strip()


VARIANTS = {f.__name__: f for f in (v1_text_first, v2_text_and_audio, v3_transcribe, v4_myst_shape)}


def redact(req):
    ctype = req.headers.get('content-type', '')
    if 'json' not in ctype:
        return {'content_type': ctype.split(';')[0], 'body_bytes': len(req.content),
                'note': 'multipart upload of the same WAV bytes, body not stored'}
    body = json.loads(req.content)
    for msg in body.get('messages', []):
        for p in msg.get('content', []):
            if p.get('type') == 'input_audio':
                d = p['input_audio']['data']
                p['input_audio']['data'] = f"<{len(d)} base64 chars, sha256 of decoded {sha(base64.b64decode(d))}>"
    return body


def strip_audio_out(body):
    # Drop the spoken reply audio from v2 responses to keep the log small. Keep its length.
    try:
        a = body['choices'][0]['message'].get('audio')
        if a and a.get('data'):
            a['data'] = f"<{len(a['data'])} base64 chars of reply audio, not stored>"
    except Exception:
        pass
    return body


def main():
    variant, pkey = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    adult_hash = guard()
    entry = sc.load_entry(str(ADULT))
    client = openai.OpenAI(http_client=httpx.Client(
        event_hooks={'request': [_on_request], 'response': [_on_response]}, timeout=120))
    fn = VARIANTS[variant]
    with open(OUT, 'a') as out:
        for rep in range(1, n + 1):
            CAPTURE.clear()
            t0 = datetime.now(timezone.utc).isoformat()
            try:
                text = fn(client, entry, PROMPTS[pkey])
            except Exception as e:
                text = f'ERROR: {e}'
            body = CAPTURE.get('body')
            if isinstance(body, dict):
                body = strip_audio_out(body)
            rec = {
                'sent_at_utc': t0, 'variant': variant, 'prompt_key': pkey, 'repeat': rep,
                'prompt_text': PROMPTS[pkey], 'openai_sdk': version('openai'),
                'audio_file_sha256': adult_hash, 'wav_sha256': sha(entry['wav_bytes']),
                'request_url': str(CAPTURE['request'].url) if 'request' in CAPTURE else None,
                'request_body_redacted': redact(CAPTURE['request']) if 'request' in CAPTURE else None,
                'http_status': CAPTURE.get('status'), 'response_headers': CAPTURE.get('headers'),
                'raw_response': body, 'function_return': text,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out.flush()
            usage = body.get('usage') if isinstance(body, dict) else None
            print(f"{variant} {pkey} rep {rep} status {CAPTURE.get('status')} usage {usage}\n  {text!r}")
            time.sleep(1.0)


if __name__ == '__main__':
    main()
