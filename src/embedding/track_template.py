# Copyright (c) 2024
# Track Template Manager
"""
Track Template Manager

Features:
1. Collect multiple embedding samples for a single track
2. Aggregate to generate track template (mean / quality-weighted mean)
3. Manage template lifecycle

Aggregation strategies:
- simple_mean: Simple average, all samples have equal weight
- quality_weighted: Quality-weighted average, quality_score as weight
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np


@dataclass
class EmbeddingSample:
    """Single embedding sample."""

    frame_id: int
    timestamp_ms: float
    embedding: np.ndarray  # [512]
    quality_score: float
    image_path: Optional[str] = None


@dataclass
class TrackTemplate:
    """Aggregated template for a Track."""

    track_id: int
    template: np.ndarray  # [512] normalized vector
    sample_count: int
    avg_quality: float
    first_frame_id: int
    last_frame_id: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Optional: associated person_id
    person_id: Optional[int] = None
    similarity_to_person: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        return {
            "track_id": self.track_id,
            "template": self.template.tolist(),
            "sample_count": self.sample_count,
            "avg_quality": round(self.avg_quality, 4),
            "first_frame_id": self.first_frame_id,
            "last_frame_id": self.last_frame_id,
            "created_at": self.created_at,
            "person_id": self.person_id,
            "similarity_to_person": (
                round(self.similarity_to_person, 4)
                if self.similarity_to_person is not None
                else None
            ),
        }


class TrackTemplateManager:
    """
    Track Template Manager

    Manage embedding sample collection and template aggregation for multiple tracks
    """

    def __init__(
        self,
        min_samples: int = 3,
        max_samples: int = 10,
        aggregation: str = "quality_weighted",
        log_path: Optional[str] = "output/track_templates.jsonl",
    ):
        """
        Initialize Track Template Manager

        Args:
            min_samples: Minimum samples required to generate template
            max_samples: Maximum samples to retain per track
            aggregation: Aggregation method ("simple_mean" or "quality_weighted")
            log_path: JSONL log path
        """
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.aggregation = aggregation
        self.log_path = log_path

        # track_id -> List[EmbeddingSample]
        self._samples: Dict[int, List[EmbeddingSample]] = {}

        # track_id -> TrackTemplate (generated templates)
        self._templates: Dict[int, TrackTemplate] = {}

        # Log file
        self._log_file = None

    def open(self) -> "TrackTemplateManager":
        """Open log file."""
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            self._log_file = open(self.log_path, "a", encoding="utf-8")
        return self

    def close(self) -> None:
        """Close log file."""
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def add_sample(
        self,
        track_id: int,
        frame_id: int,
        timestamp_ms: float,
        embedding: np.ndarray,
        quality_score: float,
        image_path: Optional[str] = None,
    ) -> Optional[TrackTemplate]:
        """
        Add an embedding sample

        Args:
            track_id: Track ID
            frame_id: Frame ID
            timestamp_ms: Timestamp
            embedding: Embedding vector [512]
            quality_score: Quality score
            image_path: Saved image path

        Returns:
            TrackTemplate if min_samples reached and template updated
        """
        if embedding is None or len(embedding) == 0:
            return None

        # Create sample
        sample = EmbeddingSample(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            embedding=embedding,
            quality_score=quality_score,
            image_path=image_path,
        )

        # Add to sample list
        if track_id not in self._samples:
            self._samples[track_id] = []

        samples = self._samples[track_id]
        samples.append(sample)

        # If exceeds max_samples, keep best by quality
        if len(samples) > self.max_samples:
            samples.sort(key=lambda s: s.quality_score, reverse=True)
            self._samples[track_id] = samples[: self.max_samples]

        # Check if can generate/update template
        if len(self._samples[track_id]) >= self.min_samples:
            template = self._generate_template(track_id)
            if template:
                self._templates[track_id] = template
                self._log_template(template)
                return template

        return None

    def _generate_template(self, track_id: int) -> Optional[TrackTemplate]:
        """
        Aggregate samples to generate template

        Args:
            track_id: Track ID

        Returns:
            TrackTemplate or None
        """
        samples = self._samples.get(track_id, [])
        if len(samples) < self.min_samples:
            return None

        embeddings = np.array([s.embedding for s in samples])  # [N, 512]
        qualities = np.array([s.quality_score for s in samples])  # [N]

        if self.aggregation == "simple_mean":
            # Simple average
            template_vec = np.mean(embeddings, axis=0)
        elif self.aggregation == "quality_weighted":
            # Quality-weighted average
            # Avoid zero weights
            weights = np.clip(qualities, 0.01, None)
            weights = weights / weights.sum()
            template_vec = np.average(embeddings, axis=0, weights=weights)
        else:
            # Default simple average
            template_vec = np.mean(embeddings, axis=0)

        # L2 normalize
        norm = np.linalg.norm(template_vec)
        if norm > 0:
            template_vec = template_vec / norm

        return TrackTemplate(
            track_id=track_id,
            template=template_vec.astype(np.float32),
            sample_count=len(samples),
            avg_quality=float(np.mean(qualities)),
            first_frame_id=min(s.frame_id for s in samples),
            last_frame_id=max(s.frame_id for s in samples),
        )

    def get_template(self, track_id: int) -> Optional[TrackTemplate]:
        """Get template for specified track."""
        return self._templates.get(track_id)

    def get_all_templates(self) -> Dict[int, TrackTemplate]:
        """Get all generated templates."""
        return self._templates.copy()

    def get_sample_count(self, track_id: int) -> int:
        """Get sample count for specified track."""
        return len(self._samples.get(track_id, []))

    def _log_template(self, template: TrackTemplate) -> None:
        """Write to JSONL log."""
        if self._log_file:
            # Don't save full 512-dim vector in log, only metadata
            log_entry = {
                "track_id": template.track_id,
                "sample_count": template.sample_count,
                "avg_quality": round(template.avg_quality, 4),
                "first_frame_id": template.first_frame_id,
                "last_frame_id": template.last_frame_id,
                "created_at": template.created_at,
                "person_id": template.person_id,
                "similarity_to_person": template.similarity_to_person,
                "template_norm": round(float(np.linalg.norm(template.template)), 4),
            }
            self._log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            self._log_file.flush()

    def clear_track(self, track_id: int) -> None:
        """Clear data for specified track."""
        self._samples.pop(track_id, None)
        self._templates.pop(track_id, None)

    def reset(self) -> None:
        """Reset all data."""
        self._samples.clear()
        self._templates.clear()
