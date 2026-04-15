#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download ArcFace ONNX model.

Model: w600k_r50.onnx (InsightFace buffalo_l series)
- Input: [1, 3, 112, 112] (RGB, normalized to [-1, 1])
- Output: [1, 512] (L2 normalized embedding vector)
"""

import os
import sys
import zipfile
from urllib.request import urlretrieve

# Windows console encoding fix
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def progress_hook(count, block_size, total_size):
    """Download progress callback."""
    percent = int(count * block_size * 100 / total_size)
    bar = "=" * (percent // 2) + ">" + " " * (50 - percent // 2)
    print(f"\r[{bar}] {percent}%", end="", flush=True)


def download_arcface(model_dir: str = "models") -> str:
    """
    Download ArcFace ONNX model.

    Args:
        model_dir: Model save directory

    Returns:
        Model file path
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "w600k_r50.onnx")

    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f"[Info] Model already exists: {model_path} ({size_mb:.1f} MB)")
        return model_path

    # InsightFace buffalo_l model download URL
    # Contains: det_10g.onnx, w600k_r50.onnx, etc.
    url = (
        "https://github.com/deepinsight/insightface/releases/download/"
        "v0.7/buffalo_l.zip"
    )

    zip_path = os.path.join(model_dir, "buffalo_l.zip")

    print("[Info] Downloading ArcFace model (buffalo_l.zip)...")
    print(f"[Info] URL: {url}")
    print("[Info] This may take a few minutes, please wait...")

    try:
        urlretrieve(url, zip_path, reporthook=progress_hook)
        print()  # 换行
    except Exception as e:
        print(f"\n[Error] Download failed: {e}")
        print("[Info] Please download manually and extract to models/ directory")
        return None

    print("[Info] Extracting...")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Only extract w600k_r50.onnx
            for name in zf.namelist():
                if name.endswith("w600k_r50.onnx"):
                    data = zf.read(name)
                    with open(model_path, "wb") as f:
                        f.write(data)
                    print(f"[Info] Extracted: {model_path}")
                    break
    except Exception as e:
        print(f"[Error] Extraction failed: {e}")
        return None

    # Clean up zip file
    try:
        os.remove(zip_path)
        print("[Info] Cleaned up temporary files")
    except:
        pass

    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f"[Success] Model download complete: {model_path} ({size_mb:.1f} MB)")
        return model_path
    else:
        print("[Error] Model file not found")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download ArcFace ONNX model")
    parser.add_argument(
        "--model-dir", default="models", help="Model save directory (default: models)"
    )
    args = parser.parse_args()

    path = download_arcface(args.model_dir)
    if path:
        print("\n[Info] You can run the following command to test:")
        print(f"  python -c \"from src.embedding import ArcFaceEmbedder; e = ArcFaceEmbedder('{path}'); print('OK')\"")
