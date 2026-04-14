#!/usr/bin/env python3
"""
下载 SCRFD 人脸检测模型。

从 InsightFace 官方 GitHub Releases 下载模型包，提取检测模型 ONNX 文件。

用法:
  python scripts/download_model.py                     # 默认下载 buffalo_l (推荐)
  python scripts/download_model.py --pack buffalo_sc   # 下载轻量模型
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

# 模型包 → 检测模型文件名的映射
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
        sys.stdout.write(f"\r  下载中: {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)")
    else:
        mb = downloaded / 1048576
        sys.stdout.write(f"\r  下载中: {mb:.1f} MB")
    sys.stdout.flush()


def download(pack_name: str, output_dir: str = "models") -> Path:
    if pack_name not in PACK_DET_MAP:
        raise ValueError(f"不支持的模型包: {pack_name}, 可选: {list(PACK_DET_MAP.keys())}")

    det_filename = PACK_DET_MAP[pack_name]
    out_path = Path(output_dir) / det_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1048576
        print(f"[Model] 模型已存在: {out_path} ({size_mb:.1f} MB), 跳过下载.")
        return out_path

    zip_url = f"{BASE_URL}/{pack_name}.zip"
    zip_path = Path(output_dir) / f"{pack_name}.zip"

    print(f"[Model] 下载模型包: {zip_url}")
    print(f"[Model] 保存到: {zip_path}")

    try:
        urlretrieve(zip_url, str(zip_path), reporthook=_progress_hook)
        print()  # 换行
    except Exception as e:
        print(f"\n[Model] 下载失败: {e}")
        print(f"[Model] 请手动下载: {zip_url}")
        print(f"[Model] 并将 {det_filename} 解压到 {output_dir}/ 目录")
        sys.exit(1)

    # 从 zip 中提取检测模型
    print(f"[Model] 提取 {det_filename} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # 文件在 zip 中的路径可能是 buffalo_l/det_10g.onnx
        target_member = None
        for name in zf.namelist():
            if name.endswith(det_filename):
                target_member = name
                break

        if target_member is None:
            print(f"[Model] 错误: zip 中找不到 {det_filename}")
            print(f"[Model] zip 内容: {zf.namelist()}")
            sys.exit(1)

        # 提取到 output_dir
        data = zf.read(target_member)
        out_path.write_bytes(data)

    size_mb = out_path.stat().st_size / 1048576
    print(f"[Model] 已保存: {out_path} ({size_mb:.1f} MB)")

    # 清理 zip
    zip_path.unlink()
    print(f"[Model] 已删除临时 zip.")
    print(f"[Model] 完成! 模型路径: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 SCRFD 人脸检测模型")
    parser.add_argument(
        "--pack", type=str, default="buffalo_l",
        choices=list(PACK_DET_MAP.keys()),
        help="模型包名称 (默认: buffalo_l)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models",
        help="模型保存目录 (默认: models/)",
    )
    args = parser.parse_args()
    download(args.pack, args.output_dir)


if __name__ == "__main__":
    main()
