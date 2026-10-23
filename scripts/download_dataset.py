"""
Script to download and unpack the authoritative US Data.gov / OpenEI Gearbox Fault Diagnosis Dataset.
Source: https://data.openei.org/files/623/gearboxdata.zip
"""

import os
import sys
import zipfile
from pathlib import Path
import urllib.request
import shutil

DATASET_URL = "https://data.openei.org/files/623/gearboxdata.zip"


def download_and_extract(target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "gearboxdata.zip"
    extracted_dir = target_dir / "gearbox_extracted"

    if extracted_dir.exists() and any(extracted_dir.iterdir()):
        print(f"[INFO] Dataset already extracted at: {extracted_dir}")
        return extracted_dir

    if not zip_path.exists():
        print(f"[INFO] Downloading dataset from {DATASET_URL} ...")
        # Download with stream and progress
        urllib.request.urlretrieve(DATASET_URL, zip_path)
        print(f"[INFO] Download complete: {zip_path.stat().st_size / (1024 * 1024):.2f} MB")

    print(f"[INFO] Extracting archive to {extracted_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_dir)
    print(f"[INFO] Extraction complete.")
    return extracted_dir


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent.parent
    raw_path = root_dir / "data" / "raw"
    try:
        download_and_extract(raw_path)
    except Exception as e:
        print(f"[ERROR] Failed to download or extract dataset: {e}", file=sys.stderr)
        sys.exit(1)
