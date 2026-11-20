"""Turn table_rows.tex into JSON, dropping Study 1C and pooled rows. Usage: python3 table_rows_to_json.py TABLE_ROWS_TEX OUT_JSON"""
import json, re, sys
src, out = sys.argv[1], sys.argv[2]
sec, sub, T = None, None, {}
for line in open(src):
    line = line.strip()
    if line.startswith("%%"):
        tag = line[2:].strip()
        if tag in ("over 13", "over 18"):
            sub = tag
        else:
            sec = tag
            sub = None
        continue
    if not line:
        continue
    cells = [c.strip() for c in line.rstrip("\\").strip().split("&")]
    cells = [c.replace("--", "–") for c in cells]
    if cells[0] == "1C" or "pooled" in cells:
        continue
    key = sec + ("|" + sub if sub else "")
    T.setdefault(key, []).append(cells)
for k, v in T.items():
    print(k, len(v))
json.dump(T, open(out, "w"), indent=1, ensure_ascii=False)
