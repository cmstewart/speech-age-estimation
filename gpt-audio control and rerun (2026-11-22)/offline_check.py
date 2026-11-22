"""Offline transport check. Builds the gpt-audio request with the study function
and captures it with an in-memory httpx transport. Nothing is sent to any network.

Usage: python offline_check.py <audio.flac> [<audio.flac> ...]
Writes JSON results to stdout. No audio or request body is written to disk.
"""
import base64
import hashlib
import io
import json
import struct
import sys

import httpx
import numpy as np
import openai
import soundfile as sf

import study_caller as sc

CAPTURED = []


def _handler(request: httpx.Request) -> httpx.Response:
    CAPTURED.append(request)
    return httpx.Response(200, json={
        "id": "offline", "object": "chat.completion", "created": 0, "model": sc.GPT_AUDIO_MODEL,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "offline stub"}}],
    })


def sha(b):
    return hashlib.sha256(b).hexdigest()


def container(b):
    if b[:4] == b"RIFF" and b[8:12] == b"WAVE":
        return "RIFF WAVE"
    if b[:4] == b"fLaC":
        return "FLAC"
    if b[:3] == b"ID3" or b[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "MP3"
    if b[:4] == b"OggS":
        return "Ogg"
    return f"unknown, first bytes {b[:8]!r}"


def wav_fmt(b):
    # Walk RIFF chunks to the fmt chunk.
    i = 12
    while i + 8 <= len(b):
        cid, size = b[i:i + 4], struct.unpack("<I", b[i + 4:i + 8])[0]
        if cid == b"fmt ":
            tag, ch, sr, _, _, bits = struct.unpack("<HHIIHH", b[i + 8:i + 24])
            return {"format_tag": tag, "channels": ch, "sample_rate": sr, "bits_per_sample": bits}
        i += 8 + size + (size & 1)
    return None


def check(path):
    CAPTURED.clear()
    sc.openai_client = openai.OpenAI(
        api_key="offline-no-key",
        base_url="https://offline.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
        max_retries=0,
    )
    entry = sc.load_entry(path)
    stub = sc.call_gpt_audio(entry, sc.QUESTIONS["q2_over_18"])
    assert stub == "offline stub", stub
    assert len(CAPTURED) == 1
    req = CAPTURED[0]
    body = json.loads(req.content)
    part = body["messages"][0]["content"][0]
    field = part["input_audio"]["data"]

    decoded = base64.b64decode(field, validate=True)
    src = open(path, "rb").read()
    src_info = sf.info(path)
    dec_info = sf.info(io.BytesIO(decoded))
    src_pcm, _ = sf.read(path, dtype="int16")
    dec_pcm, _ = sf.read(io.BytesIO(decoded), dtype="int16")

    second = None
    try:
        d2 = base64.b64decode(decoded, validate=True)
        second = container(d2)
    except Exception:
        second = "not valid base64, so not double encoded"

    return {
        "source_path": path,
        "request_url_path": req.url.path,
        "request_keys": sorted(body.keys()),
        "model": body.get("model"),
        "modalities": body.get("modalities"),
        "max_completion_tokens": body.get("max_completion_tokens"),
        "temperature_in_body": body.get("temperature", "absent"),
        "content_part_types": [p["type"] for p in body["messages"][0]["content"]],
        "declared_format": part["input_audio"]["format"],
        "field_is_str": isinstance(field, str),
        "field_prefix": field[:12],
        "base64_len": len(field),
        "decoded_len": len(decoded),
        "decoded_container": container(decoded),
        "decoded_wav_fmt": wav_fmt(decoded) if decoded[:4] == b"RIFF" else None,
        "decoded_subtype": dec_info.subtype,
        "decoded_sample_rate": dec_info.samplerate,
        "decoded_channels": dec_info.channels,
        "decoded_duration_s": round(dec_info.duration, 4),
        "second_decode": second,
        "sha256_decoded": sha(decoded),
        "sha256_wav_bytes_given_to_function": sha(entry["wav_bytes"]),
        "decoded_equals_wav_bytes": decoded == entry["wav_bytes"],
        "sha256_source_file": sha(src),
        "source_container": container(src),
        "source_subtype": src_info.subtype,
        "source_sample_rate": src_info.samplerate,
        "source_channels": src_info.channels,
        "source_duration_s": round(src_info.duration, 4),
        "source_bytes": len(src),
        "pcm_samples_identical": bool(src_pcm.shape == dec_pcm.shape and np.array_equal(src_pcm, dec_pcm)),
        "sha256_source_pcm16": sha(src_pcm.tobytes()),
        "sha256_decoded_pcm16": sha(dec_pcm.tobytes()),
        "ratio_base64_to_decoded_wav": round(len(field) / len(decoded), 4),
        "ratio_base64_to_source_flac": round(len(field) / len(src), 4),
        "ratio_decoded_wav_to_raw_pcm": round(len(decoded) / (dec_info.frames * dec_info.channels * 2), 4),
        "request_body_bytes": len(req.content),
    }


if __name__ == "__main__":
    print(json.dumps([check(p) for p in sys.argv[1:]], indent=2))
