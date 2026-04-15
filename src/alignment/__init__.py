"""
Module 3: Face Alignment + Quality Gate + Track Template

Performs 5-point affine alignment, quality assessment, and high-quality sample caching for confirmed tracks.
"""

from .aligner import FaceAligner
from .quality import QualityConfig, QualityGate, QualityResult
from .track_sampler import SampleInfo, TrackSampler

__all__ = [
    "FaceAligner", "QualityGate", "QualityConfig", "QualityResult",
    "TrackSampler", "SampleInfo",
]
