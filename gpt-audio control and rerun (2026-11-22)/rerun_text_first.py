"""Rerun of the Study 1A and 1B gpt-audio condition with the text part before the audio part.
Approved by the author on 2026-11-22. Everything else matches the study:
gpt-audio-2025-08-28, modalities text, max_completion_tokens 300, no temperature,
the same three questions, 3 repeats, 1.0 s between calls, the tenacity retry from Study 1B,
and the same FLAC to in-memory WAV conversion.

An adult transcript control runs before and after the children, three calls each.
Resumable: calls already present in the log are skipped.

  python rerun_text_first.py --dry-run   builds one request offline and prints it redacted
  python rerun_text_first.py --send

The key is read by the OpenAI SDK from OPENAI_API_KEY. It is never printed or saved.
No audio is written to disk. KIDSVA_STIMULI_1A and KIDSVA_STIMULI_1B must name the
Kids VA stimulus folders built from a licensed copy of the corpus. The manifests are
read from this repository. The log rerun_calls.jsonl holds response text and is not
distributed.
"""
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

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import study_caller as sc  # noqa: E402

REPO = HERE.parent
STUDIES = {
    "1A": (Path(os.environ.get("KIDSVA_STIMULI_1A", "")),
           REPO / "Study 1A - Kids VA Scripted Speech (google-genai 2.0.0)/study1_stimuli/manifest.json"),
    "1B": (Path(os.environ.get("KIDSVA_STIMULI_1B", "")),
           REPO / "Study 1B - Kids VA Spontaneous Speech (google-genai 2.0.0)/Stimuli/manifest.json"),
}
ADULT = HERE / "adult_control.flac"
LOG = HERE / "rerun_calls.jsonl"
N_REPEATS = 3
DELAY = 1.0
TRANSCRIPT = "Transcribe this audio clip verbatim. Output only the words spoken."

CAP = {}


def _req(r):
    CAP["request"] = r


def _resp(r):
    r.read()
    CAP["status"] = r.status_code
    CAP["request_id"] = r.headers.get("x-request-id")
    try:
        CAP["body"] = r.json()
    except Exception:
        CAP["body"] = r.text


def sha(b):
    return hashlib.sha256(b).hexdigest()


@sc._RETRY
def _call(client, entry, question):
    r = client.chat.completions.create(
        model=sc.GPT_AUDIO_MODEL,
        modalities=['text'],
        messages=[{'role': 'user', 'content': [
            {'type': 'text', 'text': question},
            {'type': 'input_audio', 'input_audio': {'data': entry['wav_b64'], 'format': 'wav'}},
        ]}],
        max_completion_tokens=300,
    )
    m = r.choices[0].message
    text = m.content
    if text is None and getattr(m, 'refusal', None):
        text = m.refusal
    return (text or '').strip()


def call(client, entry, question):
    try:
        return _call(client, entry, question)
    except Exception as e:
        return f'ERROR: {e}'


def jobs():
    out = []
    for i in range(1, N_REPEATS + 1):
        out.append(("control_before", "adult", None, "transcript", TRANSCRIPT, i))
    for study, (d, man) in STUDIES.items():
        for e in json.load(open(man)):
            for qk, qt in sc.QUESTIONS.items():
                for rep in range(1, N_REPEATS + 1):
                    out.append((study, e["child"], str(d / e["audio_file"]), qk, qt, rep))
    for i in range(1, N_REPEATS + 1):
        out.append(("control_after", "adult", None, "transcript", TRANSCRIPT, i))
    return out


def main():
    dry = "--dry-run" in sys.argv
    if not dry and "--send" not in sys.argv:
        sys.exit("pass --dry-run or --send")
    if not (os.environ.get("KIDSVA_STIMULI_1A") and os.environ.get("KIDSVA_STIMULI_1B")):
        sys.exit("set KIDSVA_STIMULI_1A and KIDSVA_STIMULI_1B to the Kids VA stimulus folders")

    done = set()
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            r = json.loads(line)
            if not r["response"].startswith("ERROR"):
                done.add((r["study"], r["child"], r["question"], r["repeat"]))

    if dry:
        def stub(request):
            CAP["request"] = request
            return httpx.Response(200, json={"id": "dry", "object": "chat.completion", "created": 0,
                                             "model": sc.GPT_AUDIO_MODEL, "choices": [{"index": 0,
                                             "finish_reason": "stop", "message": {"role": "assistant", "content": "dry"}}]})
        client = openai.OpenAI(api_key="dry", http_client=httpx.Client(transport=httpx.MockTransport(stub)), max_retries=0)
    else:
        client = openai.OpenAI(http_client=httpx.Client(event_hooks={"request": [_req], "response": [_resp]}, timeout=120))

    all_jobs = jobs()
    todo = [j for j in all_jobs if (j[0], j[1], j[3], j[5]) not in done]
    print(f"{len(all_jobs)} calls planned, {len(todo)} remaining", flush=True)
    if dry:
        todo = [j for j in todo if j[0] == "1A"][:1]

    cache = {}
    t0 = time.time()
    with open(LOG, "a") as log:
        for n, (study, child, path, qk, qt, rep) in enumerate(todo, 1):
            src = str(ADULT) if path is None else path
            if src not in cache:
                cache[src] = sc.load_entry(src)
            entry = cache[src]
            CAP.clear()
            sent = datetime.now(timezone.utc).isoformat()
            text = call(client, entry, qt)
            if dry:
                body = json.loads(CAP["request"].content)
                d = body["messages"][0]["content"][1]["input_audio"]["data"]
                body["messages"][0]["content"][1]["input_audio"]["data"] = f"<{len(d)} base64 chars, sha256 of decoded {sha(base64.b64decode(d))}>"
                print(f"{study} {child} {qk} rep {rep}")
                print(f"POST {CAP['request'].url}")
                print(json.dumps(body, indent=2))
                print("decoded equals in-memory WAV:", base64.b64decode(d) == entry["wav_bytes"])
                return
            b = CAP.get("body") if isinstance(CAP.get("body"), dict) else {}
            ch = (b.get("choices") or [{}])[0]
            rec = {
                "sent_at_utc": sent, "study": study, "child": child, "question": qk, "repeat": rep,
                "audio_file": Path(src).name, "source_sha256": sha(entry["flac_bytes"]), "wav_sha256": sha(entry["wav_bytes"]),
                "request_shape": "text_first", "model_requested": sc.GPT_AUDIO_MODEL,
                "http_status": CAP.get("status"), "request_id": CAP.get("request_id"),
                "model_returned": b.get("model"), "system_fingerprint": b.get("system_fingerprint"),
                "finish_reason": ch.get("finish_reason"), "refusal_field": (ch.get("message") or {}).get("refusal"),
                "usage": b.get("usage"), "response": text,
            }
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
            if n % 20 == 0 or study.startswith("control"):
                el = time.time() - t0
                print(f"{n}/{len(todo)} {study} {child} {qk} {rep} | {el/60:.1f} min | {text[:70]!r}", flush=True)
            time.sleep(DELAY)
    print("done", flush=True)


if __name__ == "__main__":
    main()
