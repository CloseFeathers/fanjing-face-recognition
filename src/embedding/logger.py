# Copyright (c) 2024
# Embedding Logger
"""
Embedding Logger

Features:
1. Save each embedding to disk (npy format)
2. Write embeddings.jsonl to record metadata
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


class EmbeddingLogger:
    """
    Embedding logger

    Save embeddings to files and record metadata
    """

    def __init__(
        self,
        output_dir: str = "output/embeddings",
        log_path: str = "output/embeddings.jsonl",
    ):
        """
        Initialize Embedding Logger

        Args:
            output_dir: Directory to save embedding files
            log_path: JSONL log path
        """
        self.output_dir = output_dir
        self.log_path = log_path
        self._log_file = None

    def open(self) -> "EmbeddingLogger":
        """Open log file."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._log_file = open(self.log_path, "a", encoding="utf-8")
        return self

    def close(self) -> None:
        """Close log file."""
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
        Log an embedding

        Args:
            track_id: Track ID
            frame_id: Frame ID
            timestamp_ms: Timestamp
            embedding: Embedding vector [512]
            quality_score: Quality score
            face_image_path: Face image path
            save_embedding: Whether to save embedding file

        Returns:
            Embedding file path (if saved)
        """
        embedding_path = None

        if save_embedding and embedding is not None:
            # Save embedding
            track_dir = os.path.join(self.output_dir, f"track_{track_id}")
            os.makedirs(track_dir, exist_ok=True)
            embedding_path = os.path.join(track_dir, f"frame_{frame_id}.npy")
            np.save(embedding_path, embedding)

        # Write log
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
