"""Live transport control. Sends ONLY adult_control.flac in this folder to gpt-audio
through the Study 1B call_gpt_audio function. Three prompts, three repeats each.

  python live_run.py --dry-run   prints the exact request and sends nothing
  python live_run.py --send      sends the nine calls

The key is read by the OpenAI SDK from OPENAI_API_KEY. It is never printed or saved.
KIDSVA_STIMULI_1A and KIDSVA_STIMULI_1B must name the Kids VA stimulus folders, so the
guard can check that the adult clip is not a Kids VA file. No Kids VA file is ever sent.
adult_control.flac is not distributed. Build one with prepare_adult.py.
"""
import argparse
import base64
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import openai
from importlib.metadata import version

import study_caller as sc

HERE = Path(__file__).parent
ADULT = HERE / "adult_control.flac"
OUT = HERE / "responses.jsonl"
KIDSVA_DIRS = [os.environ.get("KIDSVA_STIMULI_1A", ""), os.environ.get("KIDSVA_STIMULI_1B", "")]
RATE_LIMIT_DELAY = 1.0  # study setting
N_REPEATS = 3

PROMPTS = {
    "transcript": "Transcribe this audio clip verbatim. Output only the words spoken.",
    "q2_over_18": sc.QUESTIONS["q2_over_18"],
    "q3_age": sc.QUESTIONS["q3_age"],
}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def guard():
    """Refuse to run on anything that is not the prepared adult clip."""
    assert ADULT.exists(), "run prepare_adult.py first"
    assert all(KIDSVA_DIRS), "set KIDSVA_STIMULI_1A and KIDSVA_STIMULI_1B"
    adult_hash = sha(ADULT.read_bytes())
    for d in KIDSVA_DIRS:
        for f in Path(d).glob("*.flac"):
            assert sha(f.read_bytes()) != adult_hash, f"adult clip matches Kids VA file {f}"
    return adult_hash


CAPTURE = {}


def _on_request(request):
    CAPTURE["request"] = request


def _on_response(response):
    response.read()
    CAPTURE["status"] = response.status_code
    CAPTURE["headers"] = {k: v for k, v in response.headers.items()
                          if k.lower().startswith(("x-request-id", "openai-", "x-ratelimit"))
                          or k.lower() == "date"}
    try:
        CAPTURE["body"] = response.json()
    except Exception:
        CAPTURE["body"] = response.text


def redacted_body(req):
    body = json.loads(req.content)
    part = body["messages"][0]["content"][0]["input_audio"]
    data = part["data"]
    part["data"] = f"<{len(data)} base64 chars, sha256 of decoded {sha(base64.b64decode(data))}, starts {data[:16]}>"
    return body


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--send", action="store_true")
    args = ap.parse_args()

    adult_hash = guard()
    entry = sc.load_entry(str(ADULT))

    if args.dry_run:
        def stub(request):
            CAPTURE["request"] = request
            return httpx.Response(200, json={"id": "dry", "object": "chat.completion", "created": 0,
                                             "model": sc.GPT_AUDIO_MODEL, "choices": [{"index": 0,
                                             "finish_reason": "stop", "message": {"role": "assistant",
                                             "content": "dry"}}]})
        sc.openai_client = openai.OpenAI(api_key="dry-run-no-key", base_url="https://api.openai.com/v1",
                                         http_client=httpx.Client(transport=httpx.MockTransport(stub)))
        for key, prompt in PROMPTS.items():
            sc.call_gpt_audio(entry, prompt)
            req = CAPTURE["request"]
            print(f"### prompt {key}  x{N_REPEATS}")
            print(f"{req.method} {req.url}")
            print(json.dumps(redacted_body(req), indent=2, ensure_ascii=False))
        print(f"\nadult_control.flac sha256 {adult_hash}")
        print(f"decoded audio field is byte identical to the in memory WAV: "
              f"{base64.b64decode(json.loads(CAPTURE['request'].content)['messages'][0]['content'][0]['input_audio']['data']) == entry['wav_bytes']}")
        return

    sc.openai_client = openai.OpenAI(  # key from OPENAI_API_KEY
        http_client=httpx.Client(event_hooks={"request": [_on_request], "response": [_on_response]},
                                 timeout=120))
    with open(OUT, "a") as out:
        for key, prompt in PROMPTS.items():
            for rep in range(1, N_REPEATS + 1):
                CAPTURE.clear()
                t0 = datetime.now(timezone.utc).isoformat()
                text = sc.call_gpt_audio(entry, prompt)
                rec = {
                    "sent_at_utc": t0,
                    "prompt_key": key,
                    "repeat": rep,
                    "prompt_text": prompt,
                    "function": "study_caller.call_gpt_audio, verbatim from study1b_age_estimation_pinned.ipynb cell 12",
                    "openai_sdk": version("openai"),
                    "audio_file": ADULT.name,
                    "audio_file_sha256": adult_hash,
                    "wav_sha256": sha(entry["wav_bytes"]),
                    "request_url": str(CAPTURE["request"].url) if "request" in CAPTURE else None,
                    "request_body_redacted": redacted_body(CAPTURE["request"]) if "request" in CAPTURE else None,
                    "http_status": CAPTURE.get("status"),
                    "response_headers": CAPTURE.get("headers"),
                    "raw_response": CAPTURE.get("body"),
                    "function_return": text,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                print(f"{key} rep {rep}: {text[:100]!r}")
                time.sleep(RATE_LIMIT_DELAY)


if __name__ == "__main__":
    sys.exit(main())
