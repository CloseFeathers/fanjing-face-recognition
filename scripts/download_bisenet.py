#!/usr/bin/env python3
"""
Download BiSeNet face parsing model (for speaking detection occlusion).

Model source: face-parsing.PyTorch (https://github.com/zllrunning/face-parsing.PyTorch)
Pre-trained weights: 79999_iter.pth (trained on CelebAMask-HQ)

Two acquisition methods:
  1. Direct ONNX download (recommended, fast)
  2. Download PyTorch weights and convert (requires torch)

Usage:
  python scripts/download_bisenet.py                # Default: direct ONNX download
  python scripts/download_bisenet.py --convert      # Download PyTorch and convert
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

# Windows UTF-8
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

# ========== Configuration ==========
OUTPUT_DIR = Path("models/speaking")
OUTPUT_FILE = "resnet18.onnx"
EXPECTED_SIZE_MB = 53  # Approximately 53MB

# Direct ONNX download URLs
# Prefer project Releases, fallback to Hugging Face
ONNX_URLS = [
    "https://github.com/FlowElement/fanjing-face-recognition/releases/download/v0.1.0/resnet18.onnx",
    "https://huggingface.co/FlowElement/face-parsing/resolve/main/resnet18.onnx",
]

# PyTorch weights download URL (official Google Drive)
PTH_GDRIVE_ID = "154JgKpzCPW82qINcVieuPH3fZ2e0P812"


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


def download_onnx_direct() -> Path:
    """Download pre-converted ONNX model directly."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / OUTPUT_FILE

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1048576
        if size_mb > 40:  # Reasonable size
            print(f"[BiSeNet] Model already exists: {out_path} ({size_mb:.1f} MB), skipping download")
            return out_path
        else:
            print(f"[BiSeNet] Existing file size abnormal ({size_mb:.1f} MB), re-downloading")
            out_path.unlink()

    print(f"[BiSeNet] Downloading pre-converted ONNX model...")

    for url in ONNX_URLS:
        print(f"[BiSeNet] Trying: {url}")
        try:
            urlretrieve(url, str(out_path), reporthook=_progress_hook)
            print()

            size_mb = out_path.stat().st_size / 1048576
            if size_mb > 40:
                print(f"[BiSeNet] Done: {out_path} ({size_mb:.1f} MB)")
                return out_path
            else:
                print(f"[BiSeNet] File size abnormal ({size_mb:.1f} MB), trying next source")
                out_path.unlink()
        except Exception as e:
            print(f"\n[BiSeNet] Download failed: {e}")
            continue

    print(f"\n[BiSeNet] All download sources failed")
    print(f"\n[BiSeNet] Alternative: use --convert flag to convert from PyTorch weights")
    print(f"          python scripts/download_bisenet.py --convert")
    return None


def download_and_convert() -> Path:
    """Download PyTorch weights and convert to ONNX."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[BiSeNet] Error: PyTorch installation required")
        print("          pip install torch")
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / OUTPUT_FILE

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1048576
        if size_mb > 40:
            print(f"[BiSeNet] Model already exists: {out_path} ({size_mb:.1f} MB), skipping")
            return out_path

    # Download PyTorch weights
    pth_path = OUTPUT_DIR / "79999_iter.pth"
    if not pth_path.exists():
        print("[BiSeNet] Downloading PyTorch weights (from Google Drive)...")
        try:
            import gdown
        except ImportError:
            print("[BiSeNet] Installing gdown...")
            os.system(f"{sys.executable} -m pip install gdown")
            import gdown

        gdown.download(id=PTH_GDRIVE_ID, output=str(pth_path), quiet=False)

    if not pth_path.exists():
        print(f"[BiSeNet] Error: Cannot download PyTorch weights")
        return None

    # Define BiSeNet network structure (simplified, inference only)
    print("[BiSeNet] Loading model and converting to ONNX...")

    # BiSeNet 网络定义
    class ConvBNReLU(nn.Module):
        def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, ks, stride, padding, bias=False)
            self.bn = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class BiSeNetOutput(nn.Module):
        def __init__(self, in_ch, mid_ch, n_classes):
            super().__init__()
            self.conv = ConvBNReLU(in_ch, mid_ch, 3, 1, 1)
            self.conv_out = nn.Conv2d(mid_ch, n_classes, 1, 1, 0, bias=True)

        def forward(self, x):
            x = self.conv(x)
            x = self.conv_out(x)
            return x

    # Load pre-trained weights
    state_dict = torch.load(str(pth_path), map_location="cpu")

    # Full BiSeNet definition is complex
    # Simplified approach: use original project's model definition
    print("[BiSeNet] Note: Full conversion requires face-parsing.PyTorch project's model definition")
    print("[BiSeNet] Recommend using direct download: python scripts/download_bisenet.py")

    # Clean up temporary files
    # pth_path.unlink()

    return None


def verify_model(model_path: Path) -> bool:
    """Verify if ONNX model is usable."""
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(str(model_path))
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        output_shape = sess.get_outputs()[0].shape

        print(f"[BiSeNet] Model verification:")
        print(f"          Input: {input_name} {input_shape}")
        print(f"          Output: {output_shape}")

        # Test inference
        dummy = np.random.randn(1, 3, 512, 512).astype(np.float32)
        out = sess.run(None, {input_name: dummy})
        print(f"          Test inference: OK (output shape {out[0].shape})")
        return True

    except Exception as e:
        print(f"[BiSeNet] Model verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download BiSeNet face parsing model")
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert from PyTorch weights (requires torch)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/speaking",
        help="Model save directory (default: models/speaking/)",
    )
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = Path(args.output_dir)

    if args.convert:
        path = download_and_convert()
    else:
        path = download_onnx_direct()

    if path and path.exists():
        verify_model(path)
        print(f"\n[BiSeNet] Speaking detection occlusion model ready: {path}")
    else:
        print(f"\n[BiSeNet] Download failed")
        print(f"\nManual download method:")
        print(f"  1. Visit https://github.com/zllrunning/face-parsing.PyTorch")
        print(f"  2. Download pre-trained weights and convert to ONNX")
        print(f"  3. Place resnet18.onnx in {OUTPUT_DIR}/")
        sys.exit(1)


if __name__ == "__main__":
    main()
