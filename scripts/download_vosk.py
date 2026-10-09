"""Download the small English Vosk model (~40 MB) for offline speech recognition.

    python scripts/download_vosk.py
then set STT_ENGINE=vosk (fully offline) or WAKE_ENGINE=vosk (offline wake word,
Google for commands) in .env.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = "vosk-model-small-en-us-0.15"
URL = f"https://alphacephei.com/vosk/models/{MODEL}.zip"


def main() -> int:
    target = ROOT / "models" / MODEL
    if (target / "conf" / "model.conf").exists():
        print(f"Already downloaded: {target}")
        return 0
    if target.exists():
        print(f"Removing incomplete download in {target}")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    try:
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            archive = Path(tmp) / f"{MODEL}.zip"
            with urllib.request.urlopen(URL, timeout=60) as response, open(archive, "wb") as out:
                shutil.copyfileobj(response, out)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(tmp)
            shutil.move(str(Path(tmp) / MODEL), str(target))  # only appears once complete
    except Exception as exc:
        print(f"Download failed: {exc}\nGet it from https://alphacephei.com/vosk/models and unzip it into models/.")
        return 1
    print(f"Saved to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
