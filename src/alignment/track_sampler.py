"""
Track Sampler — Maintains high-quality face sample cache for each track.

Strategy: best-K
  - Each track keeps at most max_samples faces
  - If new sample quality > worst cached sample, replace it
  - Periodically/on-demand query for track's best sample (for embedding)

Also responsible for:
  - Writing aligned face images to disk output/faces/track_{id}/
  - Writing sample records to JSONL log output/track_faces.jsonl
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .quality import QualityResult


@dataclass
class SampleInfo:
    """A single sample record."""
    track_id: int
    frame_id: int
    timestamp_ms: float
    quality_score: float
    save_path: str


class TrackSampler:
    """Per-track face sample manager."""

    def __init__(
        self,
        max_samples: int = 10,
        output_dir: str = "output/faces",
        log_path: str = "output/track_faces.jsonl",
    ) -> None:
        self.max_samples = max_samples
        self.output_dir = Path(output_dir)
        self._samples: Dict[int, List[SampleInfo]] = {}

        # JSONL logger
        self._log_path = Path(log_path)
        self._log_fp = None

        # Statistics
        self.total_evaluated = 0
        self.total_passed = 0
        self.total_saved = 0

    # --- Lifecycle ---

    def open(self) -> "TrackSampler":
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fp = open(self._log_path, "w", encoding="utf-8")
        return self

    def close(self) -> None:
        if self._log_fp:
            self._log_fp.close()
            self._log_fp = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # --- Core ---

    def try_add(
        self,
        track_id: int,
        frame_id: int,
        timestamp_ms: float,
        aligned_face: Optional[np.ndarray],
        quality: QualityResult,
    ) -> Optional[SampleInfo]:
        """Try to add a sample for track.

        Returns:
            SampleInfo if successfully sampled, otherwise None
        """
        self.total_evaluated += 1

        log_entry = {
            "timestamp_ms": round(timestamp_ms, 3),
            "frame_id": frame_id,
            "track_id": track_id,
            "aligned": aligned_face is not None,
            **quality.to_dict(),
            "save_path": None,
        }

        if not quality.passed or aligned_face is None:
            self._write_log(log_entry)
            return None

        self.total_passed += 1

        # Write to disk
        track_dir = self.output_dir / f"track_{track_id}"
        track_dir.mkdir(parents=True, exist_ok=True)
        save_path = track_dir / f"frame_{frame_id}.jpg"
        cv2.imwrite(str(save_path), aligned_face)

        info = SampleInfo(
            track_id=track_id,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            quality_score=quality.score,
            save_path=str(save_path),
        )

        # best-K cache
        samples = self._samples.setdefault(track_id, [])
        if len(samples) < self.max_samples:
            samples.append(info)
            self.total_saved += 1
        else:
            worst_idx = min(range(len(samples)),
                            key=lambda i: samples[i].quality_score)
            if quality.score > samples[worst_idx].quality_score:
                old = samples[worst_idx]
                old_path = Path(old.save_path)
                if old_path.exists():
                    old_path.unlink(missing_ok=True)
                samples[worst_idx] = info
            else:
                # New sample quality not good enough, delete just-saved file
                save_path.unlink(missing_ok=True)

        log_entry["save_path"] = str(save_path)
        self._write_log(log_entry)
        return info

    # --- Query ---

    def get_samples(self, track_id: int) -> List[SampleInfo]:
        return self._samples.get(track_id, [])

    def get_best(self, track_id: int) -> Optional[SampleInfo]:
        samples = self._samples.get(track_id, [])
        if not samples:
            return None
        return max(samples, key=lambda s: s.quality_score)

    def get_all_tracks(self) -> Dict[int, List[SampleInfo]]:
        return dict(self._samples)

    def remove_track(self, track_id: int) -> None:
        """Clean up samples for removed track (optional, to free disk space)."""
        self._samples.pop(track_id, None)

    # --- Internal ---

    def _write_log(self, entry: dict) -> None:
        if self._log_fp:
            self._log_fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._log_fp.flush()
