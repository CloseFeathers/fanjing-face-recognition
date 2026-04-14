# Copyright (c) 2024
# Embedding Logger
"""
Embedding Logger

功能:
1. 保存每个 embedding 到磁盘 (npy 格式)
2. 写入 embeddings.jsonl 记录元数据
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


class EmbeddingLogger:
    """
    Embedding 日志记录器

    保存 embedding 到文件并记录元数据
    """

    def __init__(
        self,
        output_dir: str = "output/embeddings",
        log_path: str = "output/embeddings.jsonl",
    ):
        """
        初始化 Embedding Logger

        Args:
            output_dir: embedding 文件保存目录
            log_path: JSONL 日志路径
        """
        self.output_dir = output_dir
        self.log_path = log_path
        self._log_file = None

    def open(self) -> "EmbeddingLogger":
        """打开日志文件"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._log_file = open(self.log_path, "a", encoding="utf-8")
        return self

    def close(self) -> None:
        """关闭日志文件"""
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def log(
        self,
        track_id: int,
        frame_id: int,
        timestamp_ms: float,
        embedding: np.ndarray,
        quality_score: float,
        face_image_path: Optional[str] = None,
        save_embedding: bool = True,
    ) -> Optional[str]:
        """
        记录一个 embedding

        Args:
            track_id: 轨迹 ID
            frame_id: 帧 ID
            timestamp_ms: 时间戳
            embedding: 嵌入向量 [512]
            quality_score: 质量分数
            face_image_path: 人脸图像路径
            save_embedding: 是否保存 embedding 文件

        Returns:
            embedding 文件路径 (如果保存)
        """
        embedding_path = None

        if save_embedding and embedding is not None:
            # 保存 embedding
            track_dir = os.path.join(self.output_dir, f"track_{track_id}")
            os.makedirs(track_dir, exist_ok=True)
            embedding_path = os.path.join(track_dir, f"frame_{frame_id}.npy")
            np.save(embedding_path, embedding)

        # 写入日志
        if self._log_file:
            entry = {
                "track_id": track_id,
                "frame_id": frame_id,
                "timestamp_ms": round(timestamp_ms, 2),
                "quality_score": round(quality_score, 4),
                "embedding_dim": len(embedding) if embedding is not None else 0,
                "embedding_path": embedding_path,
                "face_image_path": face_image_path,
            }
            self._log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._log_file.flush()

        return embedding_path
