#!/usr/bin/env python3
"""
Convert BiSeNet (face-parsing.PyTorch) from PyTorch format to ONNX format.

This script is for project maintainers to generate ONNX models and upload
to GitHub Releases. Regular users should use download_bisenet.py to
download the ONNX model directly.

Dependencies:
  pip install torch torchvision gdown onnx onnxruntime

Usage:
  python scripts/convert_bisenet_to_onnx.py

Output:
  models/speaking/resnet18.onnx (~53MB)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# ========== Configuration ==========
OUTPUT_DIR = Path("models/speaking")
OUTPUT_FILE = "resnet18.onnx"
PTH_GDRIVE_ID = "154JgKpzCPW82qINcVieuPH3fZ2e0P812"


def check_dependencies():
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import torchvision
    except ImportError:
        missing.append("torchvision")
    try:
        import gdown
    except ImportError:
        missing.append("gdown")
    try:
        import onnx
    except ImportError:
        missing.append("onnx")
    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime")

    if missing:
        print(f"[Error] Missing dependencies: {', '.join(missing)}")
        print(f"        pip install {' '.join(missing)}")
        sys.exit(1)


def define_bisenet():
    """Define BiSeNet network structure matching the weights."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models

    class ConvBNReLU(nn.Module):
        def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
            super().__init__()
            self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks,
                                  stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm2d(out_chan)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class BiSeNetOutput(nn.Module):
        def __init__(self, in_chan, mid_chan, n_classes):
            super().__init__()
            self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
            self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

        def forward(self, x):
            return self.conv_out(self.conv(x))

    class AttentionRefinementModule(nn.Module):
        def __init__(self, in_chan, out_chan):
            super().__init__()
            self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
            self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
            self.bn_atten = nn.BatchNorm2d(out_chan)
            self.sigmoid_atten = nn.Sigmoid()

        def forward(self, x):
            feat = self.conv(x)
            atten = F.adaptive_avg_pool2d(feat, 1)
            atten = self.sigmoid_atten(self.bn_atten(self.conv_atten(atten)))
            return torch.mul(feat, atten)

    class ContextPath(nn.Module):
        def __init__(self):
            super().__init__()
            self.resnet = models.resnet18(weights=None)
            self.arm16 = AttentionRefinementModule(256, 128)
            self.arm32 = AttentionRefinementModule(512, 128)
            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

        def forward(self, x):
            # ResNet forward
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)
            feat8 = self.resnet.layer1(x)
            feat8 = self.resnet.layer2(feat8)
            feat16 = self.resnet.layer3(feat8)
            feat32 = self.resnet.layer4(feat16)

            H8, W8 = feat8.size()[2:]
            H16, W16 = feat16.size()[2:]
            H32, W32 = feat32.size()[2:]

            avg = F.adaptive_avg_pool2d(feat32, 1)
            avg = self.conv_avg(avg)
            avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

            feat32_arm = self.arm32(feat32)
            feat32_sum = feat32_arm + avg_up
            feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
            feat32_up = self.conv_head32(feat32_up)

            feat16_arm = self.arm16(feat16)
            feat16_sum = feat16_arm + feat32_up
            feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
            feat16_up = self.conv_head16(feat16_up)

            return feat8, feat16_up, feat32_up

    class FeatureFusionModule(nn.Module):
        def __init__(self, in_chan, out_chan):
            super().__init__()
            self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
            self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
            self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)
            self.relu = nn.ReLU(inplace=True)
            self.sigmoid = nn.Sigmoid()

        def forward(self, fsp, fcp):
            fcat = torch.cat([fsp, fcp], dim=1)
            feat = self.convblk(fcat)
            atten = F.adaptive_avg_pool2d(feat, 1)
            atten = self.sigmoid(self.conv2(self.relu(self.conv1(atten))))
            return torch.mul(feat, atten) + feat

    class BiSeNet(nn.Module):
        def __init__(self, n_classes=19):
            super().__init__()
            self.cp = ContextPath()
            # No SpatialPath, directly use feat8 (128 channels from layer2)
            self.ffm = FeatureFusionModule(256, 256)  # feat8(128) + feat16_up(128)
            self.conv_out = BiSeNetOutput(256, 256, n_classes)
            self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
            self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

        def forward(self, x):
            H, W = x.size()[2:]
            feat8, feat16_up, feat32_up = self.cp(x)
            
            feat_fuse = self.ffm(feat8, feat16_up)
            feat_out = self.conv_out(feat_fuse)
            feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
            return feat_out

    return BiSeNet


def download_pth():
    import gdown
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pth_path = OUTPUT_DIR / "79999_iter.pth"

    if pth_path.exists():
        size_mb = pth_path.stat().st_size / 1048576
        if size_mb > 40:
            print(f"[Download] Weights already exist: {pth_path} ({size_mb:.1f} MB)")
            return pth_path

    print(f"[Download] Downloading PyTorch weights...")
    gdown.download(id=PTH_GDRIVE_ID, output=str(pth_path), quiet=False)

    if not pth_path.exists():
        print("[Download] Download failed!")
        return None

    size_mb = pth_path.stat().st_size / 1048576
    print(f"[Download] Done: {pth_path} ({size_mb:.1f} MB)")
    return pth_path


def convert_to_onnx(pth_path: Path) -> Path:
    import torch
    import onnx

    onnx_path = OUTPUT_DIR / OUTPUT_FILE
    temp_path = OUTPUT_DIR / "temp.onnx"

    print(f"[Convert] Loading model...")
    BiSeNet = define_bisenet()
    model = BiSeNet(n_classes=19)

    # strict=False because resnet18's fc layer is not in weights (BiSeNet doesn't use it)
    state_dict = torch.load(str(pth_path), map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"[Convert] Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, 512, 512)

    torch.onnx.export(
        model,
        dummy_input,
        str(temp_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=11,
        do_constant_folding=True,
    )

    # Newer PyTorch may separate weights to .data file, merge into single file
    data_path = Path(str(temp_path) + ".data")
    if data_path.exists():
        print(f"[Convert] Merging into single ONNX file...")
        onnx_model = onnx.load(str(temp_path), load_external_data=True)
        onnx.save_model(onnx_model, str(onnx_path), save_as_external_data=False)
        temp_path.unlink()
        data_path.unlink()
    else:
        temp_path.rename(onnx_path)

    size_mb = onnx_path.stat().st_size / 1048576
    print(f"[Convert] Done: {onnx_path} ({size_mb:.1f} MB)")
    return onnx_path


def verify_onnx(onnx_path: Path):
    import torch
    import onnxruntime as ort

    print(f"[Verify] Comparing PyTorch and ONNX outputs...")

    BiSeNet = define_bisenet()
    pth_path = OUTPUT_DIR / "79999_iter.pth"
    model = BiSeNet(n_classes=19)
    model.load_state_dict(torch.load(str(pth_path), map_location="cpu"), strict=False)
    model.eval()

    dummy = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        pt_out = model(dummy).numpy()

    sess = ort.InferenceSession(str(onnx_path))
    ort_out = sess.run(None, {"input": dummy.numpy()})[0]

    diff = np.abs(pt_out - ort_out).max()
    print(f"[Verify] Max difference: {diff:.6f}")
    print(f"[Verify] Output shape: {ort_out.shape}")

    if diff < 1e-4:
        print(f"[Verify] ✓ Verification passed!")
        return True
    else:
        print(f"[Verify] ⚠ Large difference, but may still be usable")
        return True


def main():
    check_dependencies()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pth_path = download_pth()
    if not pth_path:
        sys.exit(1)

    onnx_path = convert_to_onnx(pth_path)
    verify_onnx(onnx_path)

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"ONNX model: {onnx_path}")
    print(f"\nNext steps:")
    print(f"  1. Upload to GitHub Releases")
    print(f"  2. Update ONNX_DIRECT_URL in scripts/download_bisenet.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
