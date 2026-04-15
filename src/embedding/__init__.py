# Copyright (c) 2024
# Embedding module for face recognition

from .candidate_pool import (
    CandidateConfig,
    CandidateSummary,
    EmbeddingSample,
    FaceCandidate,
    FaceCandidatePool,
)
from .embedder import ArcFaceEmbedder

# Module 5: Tri-state identity judgment + candidate pool
from .identity_state import (
    IdentityConfig,
    IdentityDecision,
    IdentityJudge,
    IdentityState,
    RegisteredPersonDB,
)
from .logger import EmbeddingLogger
from .person_registry import PersonAssignment, PersonRegistry
from .track_template import TrackTemplate, TrackTemplateManager

__all__ = [
    "ArcFaceEmbedder",
    "TrackTemplateManager",
    "TrackTemplate",
    "PersonRegistry",
    "PersonAssignment",
    "EmbeddingLogger",
    # Module 5
    "IdentityState",
    "IdentityDecision",
    "IdentityConfig",
    "IdentityJudge",
    "RegisteredPersonDB",
    "CandidateConfig",
    "EmbeddingSample",
    "CandidateSummary",
    "FaceCandidate",
    "FaceCandidatePool",
]
