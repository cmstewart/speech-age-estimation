
import sys, os, json, base64, gzip, zlib, time
import re as _re

JOBS = sys.argv[1]
OUT  = sys.argv[2]
API_KEY = os.environ["GEMINI_API_KEY"]
MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Record every outgoing request body, across both HTTP libraries the SDK might use.
CAPTURED = []

try:
    import httpx
    _h_send = httpx.Client.send
    def _h_spy(self, request, *a, **k):
        try: CAPTURED.append(request.content)
        except Exception: pass
        return _h_send(self, request, *a, **k)
    httpx.Client.send = _h_spy
except Exception:
    pass

try:
    import requests
    _r_send = requests.sessions.Session.send
    def _r_spy(self, request, *a, **k):
        try: CAPTURED.append(request.body)
        except Exception: pass
        return _r_send(self, request, *a, **k)
    requests.sessions.Session.send = _r_spy
except Exception:
    pass

import google.genai as genai
from importlib.metadata import version as _v
sdk_version = _v("google-genai")

client = genai.Client(api_key=API_KEY)
with open(JOBS) as jf:
    jobs = json.loads(jf.read())


def run_one(audio_path, question):
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    flac_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    # ===== verbatim original call_gemini path =====
    resp = client.models.generate_content(
        model=MODEL,
        contents=[
            {"inline_data": {"mime_type": "audio/flac", "data": flac_b64}},
            question,
        ],
    )
    return resp.text.strip()
    # ===== end call block =====


def decompress(raw):
    if isinstance(raw, str):
        raw = raw.encode("utf-8", "ignore")
    if not isinstance(raw, (bytes, bytearray)):
        return b""
    raw = bytes(raw)
    for attempt in (lambda b: gzip.decompress(b),
                    lambda b: zlib.decompress(b),
                    lambda b: zlib.decompress(b, -zlib.MAX_WBITS),
                    lambda b: b):
        try:
            return attempt(raw)
        except Exception:
            continue
    return raw


# Longest base64-ish run in the raw body, standard or url-safe, padded or not.
_b64_run = _re.compile(rb"[A-Za-z0-9+/_-]{200,}={0,2}")


def best_audio_field(bodies):
    best = b""
    for body in bodies:
        if body is None:
            continue
        text = decompress(body)
        for m in _b64_run.findall(text):
            if len(m) > len(best):
                best = m
    return best


def try_decode(b):
    if isinstance(b, (bytes, bytearray)):
        s = bytes(b).decode("ascii", "ignore")
    else:
        s = str(b)
    s = s.replace("-", "+").replace("_", "/")
    s += "=" * (-len(s) % 4)
    return base64.b64decode(s)


with open(OUT, "w") as out:
    for i, job in enumerate(jobs):
        try:
            ans = run_one(job["audio_path"], job["question"])
        except Exception as e:
            ans = f"ERROR: {e}"
        out.write(json.dumps({"tag": job["tag"], "response": ans}) + "\n")
        out.flush()
        print(f"PROGRESS {i+1}/{len(jobs)}", file=sys.stderr, flush=True)
        time.sleep(0.5)

    forensic = {"captured": False, "bodies_seen": len(CAPTURED)}
    field = best_audio_field(CAPTURED)
    if field:
        try:
            dec = try_decode(field)
            forensic = {
                "captured": True,
                "bodies_seen": len(CAPTURED),
                "wire_b64_len": len(field),
                "decoded_len": len(dec),
                "decoded_head_ascii": dec[:8].decode("latin1"),
                "decoded_is_flac": dec[:4] == b"fLaC",
            }
            if dec[:4] != b"fLaC":
                try:
                    d2 = try_decode(dec)
                    forensic["double_decoded_is_flac"] = d2[:4] == b"fLaC"
                except Exception:
                    forensic["double_decoded_is_flac"] = False
        except Exception as e:
            forensic = {"captured": False, "bodies_seen": len(CAPTURED), "decode_error": repr(e)}

    out.write(json.dumps({"type": "meta", "sdk_version": sdk_version, "forensic": forensic}) + "\n")
