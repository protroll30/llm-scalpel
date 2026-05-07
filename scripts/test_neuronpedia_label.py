"""Probe Neuronpedia + network: fetch a well-known GPT-2 Small layer-8 latent label.

Run from repo root (needs ``neuronpedia`` + ``requests`` + ``python-dotenv``; key in ``.env``):
  python scripts/test_neuronpedia_label.py

On WSL without Linux Python, you can still use host Windows Python via e.g.::
  /mnt/host/c/Python313/python.exe scripts/test_neuronpedia_label.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dotenv import load_dotenv

_ = load_dotenv(_REPO / ".env", override=True, encoding="utf-8-sig")

from discovery.labels import fetch_feature_label


def main() -> None:
    # Public page: https://neuronpedia.org/gpt2-small/8-res-jb/4052
    label = fetch_feature_label("gpt2-small", "8-res-jb", 4052, allow_missing_key=True)
    print("status:", label.status)
    print("density:", label.density)
    print("url:", label.neuronpedia_url)
    print("n_explanations:", len(label.explanations))
    print("primary_text:", label.primary_text)
    if label.detail:
        print("detail:", label.detail)


if __name__ == "__main__":
    main()
