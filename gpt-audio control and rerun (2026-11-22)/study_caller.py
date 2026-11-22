"""Verbatim copies of the Study 1B gpt-audio loader and caller.

Source: Study 1B - Kids VA Spontaneous Speech (google-genai 2.0.0)/study1b_age_estimation_pinned.ipynb
  cell 8   WAV conversion inside the manifest loop
  cell 12  GPT_AUDIO_MODEL, _RETRY, _gpt_audio_call, call_gpt_audio

The Study 1A caller was found in an earlier copy of the Study 1A notebook that is not in this
repository. It builds the same request body without the retry wrapper.

The only change is that openai_client is injected by the caller of this module
instead of being a notebook global, so the offline check can supply a client
whose transport never touches the network.
"""
import base64
import io

import soundfile as sf
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

GPT_AUDIO_MODEL = 'gpt-audio-2025-08-28'   # successor to the retired gpt-4o-audio-preview

openai_client = None  # set by the importing script


def load_entry(audio_path):
    """Study 1B cell 8, the WAV part of the loop body."""
    entry = {}
    with open(audio_path, 'rb') as f:
        entry['flac_bytes'] = f.read()
    entry['flac_b64'] = base64.b64encode(entry['flac_bytes']).decode('utf-8')

    data, sr = sf.read(audio_path)
    wav_buf = io.BytesIO()
    sf.write(wav_buf, data, sr, format='WAV')
    wav_buf.seek(0)
    entry['wav_bytes'] = wav_buf.read()
    entry['wav_b64'] = base64.b64encode(entry['wav_bytes']).decode('utf-8')
    return entry


_RETRYABLE_SUBSTRINGS = (
    '429', 'rate limit', 'rate_limit', 'too many requests',
    'timeout', 'timed out', 'temporarily unavailable', 'overloaded',
    'service unavailable', '502', '503', '504', 'connection reset',
    'connection error', 'remote disconnected',
)


def _is_retryable(exc):
    try:
        import openai
        if isinstance(exc, (openai.RateLimitError, openai.APITimeoutError,
                            openai.APIConnectionError, openai.InternalServerError)):
            return True
        if isinstance(exc, openai.BadRequestError):
            return False
    except Exception:
        pass
    try:
        import httpx
        if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return any(s in msg for s in _RETRYABLE_SUBSTRINGS)


_RETRY = retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential_jitter(initial=2, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)


@_RETRY
def _gpt_audio_call(entry, question):
    response = openai_client.chat.completions.create(
        model=GPT_AUDIO_MODEL,
        modalities=['text'],
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'input_audio', 'input_audio': {'data': entry['wav_b64'], 'format': 'wav'}},
                {'type': 'text', 'text': question},
            ],
        }],
        max_completion_tokens=300,
    )
    msg = response.choices[0].message
    text = msg.content
    if text is None and getattr(msg, 'refusal', None):
        text = msg.refusal
    return (text or '').strip()


def call_gpt_audio(entry, question):
    try:
        return _gpt_audio_call(entry, question)
    except Exception as e:
        return f'ERROR: {e}'


# Study 1B cell 10, verbatim. Identical text in Study 1A cell 12.
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
        'features of the speech, estimate the speaker\'s age in years.\n\n'
        'Respond in this format:\n'
        'Estimated age: [number]\n'
        'Confidence: [low/medium/high]\n'
        'Brief reasoning: [one sentence about aspects of the pitch, speech rate, '
        'articulation, vocabulary, sentence complexity, and any other relevant '
        'speech patterns that led to your estimation]'
    ),
}
