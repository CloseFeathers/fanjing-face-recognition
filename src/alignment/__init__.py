"""
Module 3: Face Alignment + Quality Gate + Track Template
对 confirmed track 进行 5 点仿射对齐、质量评估、高质量样本缓存。
"""

from .aligner import FaceAligner
from .quality import QualityConfig, QualityGate, QualityResult
from .track_sampler import SampleInfo, TrackSampler

__all__ = [
    "FaceAligner", "QualityGate", "QualityConfig", "QualityResult",
    "TrackSampler", "SampleInfo",
]
