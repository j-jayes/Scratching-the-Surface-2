"""Download Severstal, NEU-DET, GC10-DET from Kaggle into data/raw/.

Requires KAGGLE_USERNAME / KAGGLE_KEY in .env (already set). Skips datasets
whose target directory is already populated.
"""
from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DIR

# Dataset slug → (kind, kaggle-id, target-subdir, marker-file-or-dir)
DATASETS = [
    ("competition", "severstal-steel-defect-detection", "severstal", "train.csv"),
    ("dataset",     "kaustubhdikshit/neu-surface-defect-database", "neu_det", "NEU-DET"),
    ("dataset",     "alex000kim/gc10det", "gc10", "ds"),
]


def _kaggle_auth() -> None:
    """Make sure the Kaggle SDK can find creds — it expects KAGGLE_USERNAME/KEY env vars."""
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        from dotenv import load_dotenv
        load_dotenv()
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        raise SystemExit("KAGGLE_USERNAME / KAGGLE_KEY missing from environment.")


def _unzip_all(target: Path) -> None:
    """Unzip every *.zip in target (recursively, one pass)."""
    for z in list(target.glob("*.zip")):
        print(f"  unzip {z.name}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(target)
        z.unlink()


def download(kind: str, slug: str, sub: str, marker: str) -> None:
    target = RAW_DIR / sub
    target.mkdir(parents=True, exist_ok=True)
    if (target / marker).exists() or any(target.iterdir()):
        # crude "looks populated" check beyond the marker, to allow rerun
        if (target / marker).exists():
            print(f"[skip] {slug} → {target} (marker '{marker}' present)")
            return
    print(f"[fetch] {slug} → {target}")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    if kind == "competition":
        api.competition_download_files(slug, path=str(target), quiet=False)
    elif kind == "dataset":
        api.dataset_download_files(slug, path=str(target), quiet=False, unzip=False)
    else:
        raise ValueError(kind)
    _unzip_all(target)
    # Some Severstal-style archives contain nested zips
    _unzip_all(target)


def main() -> int:
    _kaggle_auth()
    for kind, slug, sub, marker in DATASETS:
        try:
            download(kind, slug, sub, marker)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR downloading {slug}: {type(e).__name__}: {e}")
    print("\nDone. Contents of data/raw/:")
    for child in sorted(RAW_DIR.iterdir()):
        if child.is_dir():
            n = sum(1 for _ in child.rglob("*") if _.is_file())
            print(f"  {child.name}/  ({n} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
