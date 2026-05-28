"""Download the three BDG2 files we need: metadata, weather, electricity meters.

The buds-lab/building-data-genome-project-2 repo stores its CSVs in Git LFS.
We pull them from GitHub's LFS media endpoint with a streamed download so we
don't load the whole file into memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

BASE = "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data"

FILES = {
    "metadata.csv": f"{BASE}/metadata/metadata.csv",
    "weather.csv": f"{BASE}/weather/weather.csv",
    "electricity_cleaned.csv": f"{BASE}/meters/cleaned/electricity_cleaned.csv",
}

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with dest.open("wb") as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 / total
                    print(f"  {dest.name}: {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.0f}%)", end="\r")
        print()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = OUT_DIR / name
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"already have {name} ({dest.stat().st_size / 1e6:.1f} MB), skipping")
            continue
        print(f"downloading {name}")
        download(url, dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
