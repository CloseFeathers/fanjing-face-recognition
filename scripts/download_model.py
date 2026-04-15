#!/usr/bin/env python3
"""
Download SCRFD face detection model.

Downloads model package from InsightFace official GitHub Releases and extracts
the detection model ONNX file.

Usage:
  python scripts/download_model.py                     # Download buffalo_l (recommended)
  python scripts/download_model.py --pack buffalo_sc   # Download lightweight model
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# Windows UTF-8
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

BASE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7"

# Model package -> detection model filename mapping
PACK_DET_MAP = {
    "buffalo_l":  "det_10g.onnx",    # scrfd_10g_bnkps, 10GF, ~16MB
    "buffalo_sc": "det_500m.onnx",   # scrfd_500m_bnkps, 500MF, ~2MB
}


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        mb = downloaded / 1048576
        total_mb = total_size / 1048576
        sys.stdout.write(f"\r  Downloading: {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)")
    else:
        mb = downloaded / 1048576
        sys.stdout.write(f"\r  Downloading: {mb:.1f} MB")
    sys.stdout.flush()


def download(pack_name: str, output_dir: str = "models") -> Path:
    if pack_name not in PACK_DET_MAP:
        raise ValueError(f"Unsupported model pack: {pack_name}, options: {list(PACK_DET_MAP.keys())}")

    det_filename = PACK_DET_MAP[pack_name]
    out_path = Path(output_dir) / det_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1048576
        print(f"[Model] Model already exists: {out_path} ({size_mb:.1f} MB), skipping download.")
        return out_path

    zip_url = f"{BASE_URL}/{pack_name}.zip"
    zip_path = Path(output_dir) / f"{pack_name}.zip"

    print(f"[Model] Downloading model package: {zip_url}")
    print(f"[Model] Saving to: {zip_path}")

    try:
        urlretrieve(zip_url, str(zip_path), reporthook=_progress_hook)
        print()  # 换行
    except Exception as e:
        print(f"\n[Model] Download failed: {e}")
        print(f"[Model] Please download manually: {zip_url}")
        print(f"[Model] And extract {det_filename} to {output_dir}/ directory")
        sys.exit(1)

    # Extract detection model from zip
    print(f"[Model] Extracting {det_filename} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # File path in zip may be buffalo_l/det_10g.onnx
        target_member = None
        for name in zf.namelist():
            if name.endswith(det_filename):
                target_member = name
                break

        if target_member is None:
            print(f"[Model] Error: {det_filename} not found in zip")
            print(f"[Model] zip contents: {zf.namelist()}")
            sys.exit(1)

        # Extract to output_dir
        data = zf.read(target_member)
        out_path.write_bytes(data)

    size_mb = out_path.stat().st_size / 1048576
    print(f"[Model] Saved: {out_path} ({size_mb:.1f} MB)")

    # Clean up zip
    zip_path.unlink()
    print(f"[Model] Deleted temporary zip.")
    print(f"[Model] Done! Model path: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SCRFD face detection model")
    parser.add_argument(
        "--pack", type=str, default="buffalo_l",
        choices=list(PACK_DET_MAP.keys()),
        help="Model package name (default: buffalo_l)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models",
        help="Model save directory (default: models/)",
    )
    args = parser.parse_args()
    download(args.pack, args.output_dir)


if __name__ == "__main__":
    main()
