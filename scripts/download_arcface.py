#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 ArcFace ONNX 模型

模型: w600k_r50.onnx (InsightFace buffalo_l 系列)
- 输入: [1, 3, 112, 112] (RGB, 归一化到 [-1, 1])
- 输出: [1, 512] (L2 归一化嵌入向量)
"""

import os
import sys
import zipfile
from urllib.request import urlretrieve

# Windows 控制台编码修复
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def progress_hook(count, block_size, total_size):
    """下载进度回调"""
    percent = int(count * block_size * 100 / total_size)
    bar = "=" * (percent // 2) + ">" + " " * (50 - percent // 2)
    print(f"\r[{bar}] {percent}%", end="", flush=True)


def download_arcface(model_dir: str = "models") -> str:
    """
    下载 ArcFace ONNX 模型

    Args:
        model_dir: 模型保存目录

    Returns:
        模型文件路径
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "w600k_r50.onnx")

    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f"[Info] 模型已存在: {model_path} ({size_mb:.1f} MB)")
        return model_path

    # InsightFace buffalo_l 模型下载地址
    # 包含: det_10g.onnx, w600k_r50.onnx 等
    url = (
        "https://github.com/deepinsight/insightface/releases/download/"
        "v0.7/buffalo_l.zip"
    )

    zip_path = os.path.join(model_dir, "buffalo_l.zip")

    print("[Info] 正在下载 ArcFace 模型 (buffalo_l.zip)...")
    print(f"[Info] URL: {url}")
    print("[Info] 这可能需要几分钟，请耐心等待...")

    try:
        urlretrieve(url, zip_path, reporthook=progress_hook)
        print()  # 换行
    except Exception as e:
        print(f"\n[Error] 下载失败: {e}")
        print("[Info] 请手动下载并解压到 models/ 目录")
        return None

    print("[Info] 正在解压...")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # 只提取 w600k_r50.onnx
            for name in zf.namelist():
                if name.endswith("w600k_r50.onnx"):
                    data = zf.read(name)
                    with open(model_path, "wb") as f:
                        f.write(data)
                    print(f"[Info] 已提取: {model_path}")
                    break
    except Exception as e:
        print(f"[Error] 解压失败: {e}")
        return None

    # 清理 zip 文件
    try:
        os.remove(zip_path)
        print("[Info] 已清理临时文件")
    except:
        pass

    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f"[Success] 模型下载完成: {model_path} ({size_mb:.1f} MB)")
        return model_path
    else:
        print("[Error] 模型文件未找到")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="下载 ArcFace ONNX 模型")
    parser.add_argument(
        "--model-dir", default="models", help="模型保存目录 (默认: models)"
    )
    args = parser.parse_args()

    path = download_arcface(args.model_dir)
    if path:
        print("\n[Info] 可以运行以下命令测试:")
        print(f"  python -c \"from src.embedding import ArcFaceEmbedder; e = ArcFaceEmbedder('{path}'); print('OK')\"")
