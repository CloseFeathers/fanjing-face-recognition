"""
Module 5: Session-level FaceCandidatePool + Candidate Summary

Features:
1. Candidate pool input objects are session-level persons (not tracks)
2. identity_state = UNKNOWN_STRONG or AMBIGUOUS timeout can enter
3. Candidate binds to session_person_id
4. Maintain candidate samples and summary objects for each candidate

Summary Object contains:
- session_id, source_id, candidate_id, source_session_person_id
- num_samples, num_tracks_covered
- first_timestamp_ms, last_timestamp_ms
- centroid/template (persisted)
- prototype embeddings (max 3)
- internal_consistency, avg_quality_score
- max_similarity_to_known_person, top1/top2 info
- ready, not_ready_reasons
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .identity_state import IdentityDecision, IdentityState


@dataclass
class CandidateConfig:
    """Candidate pool configuration."""
    # Conditions to enter candidate pool
    min_samples_to_enter: int = 3
    min_tracks_to_enter: int = 1
    min_quality_score_to_enter: float = 0.5

    # register_ready composite score
    register_threshold: float = 0.6
    min_samples_absolute: int = 3       # Baseline: below this value directly not ready
    min_consistency_absolute: float = 0.4  # Baseline: below this value directly not ready

    # Prototype selection parameters
    outlier_threshold: float = 0.4
    prototype_diff_threshold: float = 0.15
    max_prototypes: int = 3

    # Internal consistency calculation
    max_samples_for_pairwise: int = 20


@dataclass
class EmbeddingSample:
    """Single embedding sample."""
    embedding: np.ndarray       # [512]
    quality_score: float
    track_id: int
    frame_id: int
    timestamp_ms: float
    image_path: Optional[str] = None


@dataclass
class CandidateSummary:
    """Candidate summary object."""
    session_id: str
    source_id: str
    candidate_id: int
    source_session_person_id: int

    num_samples: int
    num_tracks_covered: int
    first_timestamp_ms: float
    last_timestamp_ms: float

    # Vectors (not serialized to JSON, persisted separately)
    centroid: Optional[np.ndarray] = None
    prototypes: List[np.ndarray] = field(default_factory=list)

    # Paths (serialized to JSON)
    centroid_path: Optional[str] = None
    prototype_paths: List[str] = field(default_factory=list)

    # Statistics
    internal_consistency: float = 0.0
    avg_quality_score: float = 0.0

    # Comparison with known database
    max_similarity_to_known_person: float = -1.0
    top1_known_person_id: Optional[int] = None
    top1_score: float = -1.0
    top2_known_person_id: Optional[int] = None
    top2_score: Optional[float] = None
    margin: Optional[float] = None

    # Ready state
    ready: bool = False
    not_ready_reasons: List[str] = field(default_factory=list)

    # Prototype selection explanation
    prototype_selection_reason: str = ""

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "source_id": self.source_id,
            "candidate_id": self.candidate_id,
            "source_session_person_id": self.source_session_person_id,
            "num_samples": self.num_samples,
            "num_tracks_covered": self.num_tracks_covered,
            "first_timestamp_ms": round(self.first_timestamp_ms, 3),
            "last_timestamp_ms": round(self.last_timestamp_ms, 3),
            "centroid_path": self.centroid_path,
            "prototype_paths": self.prototype_paths,
            "prototype_count": len(self.prototypes),
            "internal_consistency": round(self.internal_consistency, 4),
            "avg_quality_score": round(self.avg_quality_score, 4),
            "max_similarity_to_known_person": round(self.max_similarity_to_known_person, 4),
            "top1_known_person_id": self.top1_known_person_id,
            "top1_score": round(self.top1_score, 4) if self.top1_score is not None else None,
            "top2_known_person_id": self.top2_known_person_id,
            "top2_score": round(self.top2_score, 4) if self.top2_score is not None else None,
            "margin": round(self.margin, 4) if self.margin is not None else None,
            "ready": self.ready,
            "not_ready_reasons": self.not_ready_reasons,
            "prototype_selection_reason": self.prototype_selection_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class FaceCandidate:
    """Single candidate person object."""

    def __init__(
        self,
        candidate_id: int,
        session_person_id: int,
        session_id: str,
        source_id: str,
        config: CandidateConfig,
    ):
        self.candidate_id = candidate_id
        self.session_person_id = session_person_id
        self.session_id = session_id
        self.source_id = source_id
        self.config = config

        # Sample collection
        self.samples: List[EmbeddingSample] = []

        # Covered track_ids
        self.track_ids: Set[int] = set()

        # Timestamps
        self.first_timestamp_ms: Optional[float] = None
        self.last_timestamp_ms: Optional[float] = None

        # Cached statistics
        self._centroid: Optional[np.ndarray] = None
        self._consistency: Optional[float] = None
        self._prototypes: List[np.ndarray] = []
        self._prototype_reason: str = ""

        # Comparison results with known database
        self.identity_decision: Optional[IdentityDecision] = None

        self.created_at = datetime.now().isoformat()

    def add_sample(self, sample: EmbeddingSample):
        """Add sample."""
        self.samples.append(sample)
        self.track_ids.add(sample.track_id)

        if self.first_timestamp_ms is None:
            self.first_timestamp_ms = sample.timestamp_ms
        self.last_timestamp_ms = sample.timestamp_ms

        # Clear cache
        self._centroid = None
        self._consistency = None
        self._prototypes = []

    def compute_centroid(self) -> Optional[np.ndarray]:
        """Compute centroid (average after removing outliers)."""
        if len(self.samples) == 0:
            return None

        if self._centroid is not None:
            return self._centroid

        embeddings = np.array([s.embedding for s in self.samples])

        if len(embeddings) <= 2:
            # Too few samples, simple average
            centroid = embeddings.mean(axis=0)
        else:
            # First compute initial centroid
            initial_centroid = embeddings.mean(axis=0)
            initial_centroid = initial_centroid / (np.linalg.norm(initial_centroid) + 1e-8)

            # Calculate similarity of each sample to initial centroid
            similarities = np.dot(embeddings, initial_centroid)

            # Remove outliers (similarity < threshold)
            mask = similarities >= self.config.outlier_threshold
            if mask.sum() >= 2:
                # Enough non-outliers
                filtered = embeddings[mask]
                centroid = filtered.mean(axis=0)
            else:
                # Too many outliers, use all
                centroid = initial_centroid

        # L2 normalize
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        self._centroid = centroid.astype(np.float32)
        return self._centroid

    def compute_internal_consistency(self) -> float:
        """
        Compute internal consistency

        <= 20 samples: median of pairwise cosine similarity
        > 20 samples: median of sample similarity to centroid
        """
        if len(self.samples) < 2:
            return 1.0  # Single sample defaults to consistent

        if self._consistency is not None:
            return self._consistency

        embeddings = np.array([s.embedding for s in self.samples])
        n = len(embeddings)

        if n <= self.config.max_samples_for_pairwise:
            # Pairwise similarity
            # Use matrix multiplication to compute all pairwise similarities
            sim_matrix = np.dot(embeddings, embeddings.T)
            # Get upper triangle (excluding diagonal)
            triu_indices = np.triu_indices(n, k=1)
            pairwise_sims = sim_matrix[triu_indices]
            self._consistency = float(np.median(pairwise_sims))
        else:
            # Degrade to similarity to centroid
            centroid = self.compute_centroid()
            if centroid is None:
                self._consistency = 0.0
            else:
                sims = np.dot(embeddings, centroid)
                self._consistency = float(np.median(sims))

        return self._consistency

    def select_prototypes(self) -> Tuple[List[np.ndarray], str]:
        """
        Select prototype embeddings

        Strategy:
        1. Remove outliers from high-quality samples
        2. prototype1: high quality and closest to centroid
        3. prototype2: high quality and sufficiently different from prototype1 but not outlier
        4. prototype3: high quality and sufficiently different from both but not outlier

        Returns: (prototypes, reason)
        """
        if len(self.samples) == 0:
            return [], "no_samples"

        if self._prototypes:
            return self._prototypes, self._prototype_reason

        cfg = self.config
        centroid = self.compute_centroid()
        if centroid is None:
            return [], "centroid_unavailable"

        # Sort by quality
        sorted_samples = sorted(self.samples, key=lambda s: s.quality_score, reverse=True)

        # Filter outliers
        valid_samples = []
        for s in sorted_samples:
            sim_to_centroid = float(np.dot(s.embedding, centroid))
            if sim_to_centroid >= cfg.outlier_threshold:
                valid_samples.append((s, sim_to_centroid))

        if len(valid_samples) == 0:
            return [], "all_samples_are_outliers"

        prototypes = []
        reasons = []

        # prototype1: high quality and closest to centroid
        # Select closest to centroid among high-quality samples
        valid_samples_sorted_by_sim = sorted(valid_samples, key=lambda x: x[1], reverse=True)
        p1_sample, p1_sim = valid_samples_sorted_by_sim[0]
        prototypes.append(p1_sample.embedding)
        reasons.append(f"p1:quality={p1_sample.quality_score:.2f},sim_to_centroid={p1_sim:.3f}")

        if len(valid_samples) < 2 or cfg.max_prototypes < 2:
            self._prototypes = prototypes
            self._prototype_reason = ";".join(reasons) + ";only_1_valid_sample_or_max=1"
            return self._prototypes, self._prototype_reason

        # prototype2: sufficiently different from prototype1
        p2_found = False
        for s, sim_to_centroid in valid_samples_sorted_by_sim[1:]:
            diff_to_p1 = 1.0 - float(np.dot(s.embedding, prototypes[0]))
            if diff_to_p1 >= cfg.prototype_diff_threshold:
                prototypes.append(s.embedding)
                reasons.append(f"p2:quality={s.quality_score:.2f},diff_to_p1={diff_to_p1:.3f}")
                p2_found = True
                break

        if not p2_found:
            reasons.append("p2:not_found(all_too_similar_to_p1)")
            self._prototypes = prototypes
            self._prototype_reason = ";".join(reasons)
            return self._prototypes, self._prototype_reason

        if len(valid_samples) < 3 or cfg.max_prototypes < 3:
            self._prototypes = prototypes
            self._prototype_reason = ";".join(reasons) + ";only_2_valid_or_max=2"
            return self._prototypes, self._prototype_reason

        # prototype3: sufficiently different from both
        for s, sim_to_centroid in valid_samples_sorted_by_sim:
            if s.embedding is prototypes[0] or s.embedding is prototypes[1]:
                continue
            diff_to_p1 = 1.0 - float(np.dot(s.embedding, prototypes[0]))
            diff_to_p2 = 1.0 - float(np.dot(s.embedding, prototypes[1]))
            if diff_to_p1 >= cfg.prototype_diff_threshold and diff_to_p2 >= cfg.prototype_diff_threshold:
                prototypes.append(s.embedding)
                reasons.append(f"p3:quality={s.quality_score:.2f},diff_to_p1={diff_to_p1:.3f},diff_to_p2={diff_to_p2:.3f}")
                break
        else:
            reasons.append("p3:not_found(all_too_similar_to_p1_or_p2)")

        self._prototypes = prototypes
        self._prototype_reason = ";".join(reasons)
        return self._prototypes, self._prototype_reason

    def check_register_ready(self) -> Tuple[bool, float, List[str]]:
        """Composite scoring register_ready evaluation.

        Baseline fail directly not ready; baseline pass uses four-dimension weighted score.

        Returns:
            (ready, score, not_ready_reasons)
        """
        cfg = self.config
        n = len(self.samples)

        if n < cfg.min_samples_absolute:
            return (False, 0.0, [f"samples={n}<{cfg.min_samples_absolute}"])

        consistency = self.compute_internal_consistency()
        if consistency < cfg.min_consistency_absolute:
            return (False, 0.0, [f"consistency={consistency:.3f}<{cfg.min_consistency_absolute}"])

        sample_s = min(n / 5.0, 1.0)
        consist_s = min((consistency - 0.4) / 0.4, 1.0)
        avg_q = sum(s.quality_score for s in self.samples) / n
        quality_s = min(max((avg_q - 0.4) / 0.4, 0.0), 1.0)
        duration = (self.last_timestamp_ms - self.first_timestamp_ms) / 1000.0
        duration_s = min(duration / 3.0, 1.0)

        score = 0.25 * sample_s + 0.35 * consist_s + 0.25 * quality_s + 0.15 * duration_s
        ready = score >= cfg.register_threshold
        reasons = [] if ready else [
            f"score={score:.2f}<{cfg.register_threshold}"
            f"(samp={sample_s:.2f},cons={consist_s:.2f},qual={quality_s:.2f},dur={duration_s:.2f})"
        ]
        return (ready, score, reasons)

    def build_summary(
        self,
        output_dir: str = "output/candidate_vectors",
    ) -> CandidateSummary:
        """Build summary object and save vector files."""
        os.makedirs(output_dir, exist_ok=True)

        # Compute statistics
        centroid = self.compute_centroid()
        consistency = self.compute_internal_consistency()
        prototypes, proto_reason = self.select_prototypes()
        ready, _score, not_ready_reasons = self.check_register_ready()

        avg_quality = 0.0
        if self.samples:
            avg_quality = sum(s.quality_score for s in self.samples) / len(self.samples)

        # Save centroid
        centroid_path = None
        if centroid is not None:
            centroid_path = os.path.join(output_dir, f"candidate_{self.candidate_id}_centroid.npy")
            np.save(centroid_path, centroid)

        # Save prototypes
        prototype_paths = []
        for i, proto in enumerate(prototypes):
            proto_path = os.path.join(output_dir, f"candidate_{self.candidate_id}_proto_{i+1}.npy")
            np.save(proto_path, proto)
            prototype_paths.append(proto_path)

        # Get known database comparison results from identity_decision
        max_sim = -1.0
        top1_known_id = None
        top1_score = -1.0
        top2_known_id = None
        top2_score = None
        margin = None

        if self.identity_decision:
            top1_known_id = self.identity_decision.top1_known_person_id
            top1_score = self.identity_decision.top1_score
            top2_known_id = self.identity_decision.top2_known_person_id
            top2_score = self.identity_decision.top2_score
            margin = self.identity_decision.margin
            max_sim = top1_score if top1_score > 0 else -1.0

        summary = CandidateSummary(
            session_id=self.session_id,
            source_id=self.source_id,
            candidate_id=self.candidate_id,
            source_session_person_id=self.session_person_id,
            num_samples=len(self.samples),
            num_tracks_covered=len(self.track_ids),
            first_timestamp_ms=self.first_timestamp_ms or 0.0,
            last_timestamp_ms=self.last_timestamp_ms or 0.0,
            centroid=centroid,
            prototypes=prototypes,
            centroid_path=centroid_path,
            prototype_paths=prototype_paths,
            internal_consistency=consistency,
            avg_quality_score=avg_quality,
            max_similarity_to_known_person=max_sim,
            top1_known_person_id=top1_known_id,
            top1_score=top1_score,
            top2_known_person_id=top2_known_id,
            top2_score=top2_score,
            margin=margin,
            ready=ready,
            not_ready_reasons=not_ready_reasons,
            prototype_selection_reason=proto_reason,
            updated_at=datetime.now().isoformat(),
        )

        return summary


class FaceCandidatePool:
    """
    Session-level candidate pool

    Manage UNKNOWN_STRONG session persons with stable evidence
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        source_id: str = "",
        config: Optional[CandidateConfig] = None,
        candidates_log_path: str = "output/candidates.jsonl",
        summaries_log_path: str = "output/candidate_summaries.jsonl",
    ):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.source_id = source_id
        self.config = config or CandidateConfig()
        self.candidates_log_path = candidates_log_path
        self.summaries_log_path = summaries_log_path

        # session_person_id -> FaceCandidate
        self._candidates: Dict[int, FaceCandidate] = {}

        # Next candidate_id
        self._next_candidate_id: int = 1

        # Log files
        self._candidates_log = None
        self._summaries_log = None

    def open(self) -> "FaceCandidatePool":
        """Open log files."""
        if self.candidates_log_path:
            os.makedirs(os.path.dirname(self.candidates_log_path), exist_ok=True)
            self._candidates_log = open(self.candidates_log_path, "a", encoding="utf-8")
        if self.summaries_log_path:
            os.makedirs(os.path.dirname(self.summaries_log_path), exist_ok=True)
            self._summaries_log = open(self.summaries_log_path, "a", encoding="utf-8")
        return self

    def close(self):
        """Close log files."""
        if self._candidates_log:
            self._candidates_log.close()
            self._candidates_log = None
        if self._summaries_log:
            self._summaries_log.close()
            self._summaries_log = None

    def try_add_or_update(
        self,
        session_person_id: int,
        identity_state: IdentityState,
        identity_decision: IdentityDecision,
        sample: EmbeddingSample,
        current_samples_count: int,
        current_tracks_count: int,
        avg_quality: float,
    ) -> Optional[int]:
        """
        Try to add or update candidate

        Args:
            session_person_id: person_id within session
            identity_state: Current identity state
            identity_decision: Identity decision result
            sample: New embedding sample
            current_samples_count: Current person's total sample count
            current_tracks_count: Current person's covered track count
            avg_quality: Current average quality score

        Returns:
            candidate_id if successfully added/updated to candidate pool, else None
        """
        if identity_state not in (IdentityState.UNKNOWN_STRONG, IdentityState.AMBIGUOUS):
            return None

        cfg = self.config

        # Check if candidate already exists
        if session_person_id in self._candidates:
            # Update existing candidate
            candidate = self._candidates[session_person_id]
            candidate.add_sample(sample)
            candidate.identity_decision = identity_decision

            ready, _score, reasons = candidate.check_register_ready()
            if ready:
                self._log_candidate_event(candidate.candidate_id, "ready", {
                    "session_person_id": session_person_id,
                    "num_samples": len(candidate.samples),
                })

            return candidate.candidate_id

        # Check if meets conditions to enter candidate pool
        if (current_samples_count < cfg.min_samples_to_enter or
            current_tracks_count < cfg.min_tracks_to_enter or
            avg_quality < cfg.min_quality_score_to_enter):
            return None

        # Create new candidate
        candidate_id = self._next_candidate_id
        self._next_candidate_id += 1

        candidate = FaceCandidate(
            candidate_id=candidate_id,
            session_person_id=session_person_id,
            session_id=self.session_id,
            source_id=self.source_id,
            config=cfg,
        )
        candidate.add_sample(sample)
        candidate.identity_decision = identity_decision

        self._candidates[session_person_id] = candidate

        self._log_candidate_event(candidate_id, "created", {
            "session_person_id": session_person_id,
        })

        return candidate_id

    def get_candidate_id(self, session_person_id: int) -> Optional[int]:
        """Get candidate_id for session_person."""
        candidate = self._candidates.get(session_person_id)
        return candidate.candidate_id if candidate else None

    def get_candidate(self, session_person_id: int) -> Optional[FaceCandidate]:
        """Get candidate object."""
        return self._candidates.get(session_person_id)

    def get_counts(self) -> Dict[str, int]:
        """Get statistics."""
        total = len(self._candidates)
        ready = sum(1 for c in self._candidates.values() if c.check_register_ready()[0])
        return {
            "candidate_count": total,
            "ready_candidate_count": ready,
        }

    def flush_summaries(self) -> List[CandidateSummary]:
        """
        Generate all ready=true candidate summaries and persist

        Returns:
            List of all generated summaries
        """
        summaries = []

        for candidate in self._candidates.values():
            ready, _score, _ = candidate.check_register_ready()
            if ready:
                summary = candidate.build_summary()
                summaries.append(summary)

                if self._summaries_log:
                    self._summaries_log.write(
                        json.dumps(summary.to_dict(), ensure_ascii=False) + "\n"
                    )
                    self._summaries_log.flush()

        return summaries

    def _log_candidate_event(self, candidate_id: int, event: str, data: Dict):
        """Log candidate event."""
        if self._candidates_log:
            entry = {
                "candidate_id": candidate_id,
                "event": event,
                "timestamp": datetime.now().isoformat(),
                **data,
            }
            self._candidates_log.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._candidates_log.flush()

    def reset(self):
        """Reset candidate pool."""
        self._candidates.clear()
        self._next_candidate_id = 1
        self.session_id = str(uuid.uuid4())[:8]
