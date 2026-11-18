"""Fails if any site-identifying string appears in the tracked files. The
patterns are stored base64-encoded so this file does not match itself."""
import base64
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_ENC = ("bG9naW4xfGRvcXxxZG98cmF0YW58a2FyYXNoY2h1a3wvZ3JvdXBzL3wvbW50L3wvaG9tZS91c2VyfGRvc2F8"
        "MTkyXC4xNjh8cHJmc3xoaG1pfGphbmVsaWF8Z2hvX3xwYXNzd29yZA==")
PATTERN = re.compile(base64.b64decode(_ENC).decode(), re.I)


def _files():
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-co", "--exclude-standard"],
                             capture_output=True, text=True, check=True).stdout.split()
        return [ROOT / f for f in out]
    except Exception:
        return [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts
                and "__pycache__" not in p.parts]


def test_no_site_strings():
    hits = []
    for p in _files():
        if p.name == "test_scrub.py" or not p.is_file():
            continue
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if PATTERN.search(line):
                hits.append(f"{p.relative_to(ROOT)}:{n}: {line.strip()[:80]}")
    assert not hits, "site strings found:\n" + "\n".join(hits)
