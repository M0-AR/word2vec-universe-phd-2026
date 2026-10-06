"""Download helpers (text8 + analogy test)."""
from __future__ import annotations
import os, urllib.request, zipfile

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

def ensure_text8():
    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "text8.txt")
    if os.path.exists(out) and os.path.getsize(out) > 1_000_000:
        return out
    zpath = os.path.join(DATA, "text8.zip")
    if not os.path.exists(zpath):
        print("downloading text8 (16MB)...")
        urllib.request.urlretrieve("https://mattmahoney.net/dc/text8.zip", zpath)
    with zipfile.ZipFile(zpath) as z:
        raw = z.read("text8").decode("utf-8", errors="ignore")
    open(out, "w", encoding="utf-8").write(raw)
    return out
