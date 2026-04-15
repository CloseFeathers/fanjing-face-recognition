# Copyright (c) 2024
# Session-level Person Registry
"""
Session-level Person Registry

Features:
1. Manage person identities within a session
2. Receive new track templates, decide to match existing person or create new person
3. Support margin condition: top1_similarity - top2_similarity > margin

Matching logic:
1. Calculate similarity between new template and all person templates
2. If top1_similarity > threshold and margin condition is satisfied:
   - Bind to top1 person
   - Update person template (optional: sliding average)
3. Otherwise create new person

Threshold explanation:
- similarity_threshold: 0.4 (default, cosine similarity)
  - Typical ArcFace threshold is between 0.3-0.5
  - Same person similarity is usually > 0.5
  - Different person similarity is usually < 0.3
- margin_threshold: 0.1 (default)
  - Ensure top1 is clearly better than top2
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from .track_template import TrackTemplate


@dataclass
class PersonTemplate:
    """Aggregated template for a Person."""

    person_id: int
    template: np.ndarray  # [512] normalized vector
    track_ids: List[int]  # Associated track_id list
    sample_count: int
    first_seen_frame: int
    last_seen_frame: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        return {
            "person_id": self.person_id,
            "track_ids": self.track_ids,
            "sample_count": self.sample_count,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "created_at": self.created_at,
        }


@dataclass
class PersonAssignment:
    """Track to Person assignment result."""

    track_id: int
    person_id: int
    is_new_person: bool
    top1_similarity: float
    top2_similarity: Optional[float]
    margin: Optional[float]
    threshold: float
    margin_threshold: float
    decision: str  # "matched", "new_person", "margin_fail"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        return {
            "track_id": self.track_id,
            "person_id": self.person_id,
            "is_new_person": self.is_new_person,
            "top1_similarity": round(self.top1_similarity, 4),
            "top2_similarity": (
                round(self.top2_similarity, 4)
                if self.top2_similarity is not None
                else None
            ),
            "margin": round(self.margin, 4) if self.margin is not None else None,
            "threshold": self.threshold,
            "margin_threshold": self.margin_threshold,
            "decision": self.decision,
            "timestamp": self.timestamp,
        }


class PersonRegistry:
    """
    Session-level Person Registry

    Manage person identities within a session, implement track to person matching
    """

    def __init__(
        self,
        similarity_threshold: float = 0.4,
        margin_threshold: float = 0.1,
        update_template: bool = True,
        update_weight: float = 0.3,
        log_path: Optional[str] = "output/person_assignments.jsonl",
    ):
        """
        Initialize Person Registry

        Args:
            similarity_threshold: Matching threshold (cosine similarity)
            margin_threshold: Margin threshold (top1 - top2 > margin)
            update_template: Whether to update person template after matching
            update_weight: Update weight (new = old * (1-w) + track * w)
            log_path: Assignment log path
        """
        self.similarity_threshold = similarity_threshold
        self.margin_threshold = margin_threshold
        self.update_template = update_template
        self.update_weight = update_weight
        self.log_path = log_path

        # person_id -> PersonTemplate
        self._persons: Dict[int, PersonTemplate] = {}

        # track_id -> person_id (mapping table)
        self._track_to_person: Dict[int, int] = {}

        # Next person_id
        self._next_person_id: int = 1

        # Log file
        self._log_file = None

    def open(self) -> "PersonRegistry":
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

    def assign(self, track_template: TrackTemplate) -> PersonAssignment:
        """
        Assign track template to a person

        Args:
            track_template: Track's aggregated template

        Returns:
            PersonAssignment assignment result
        """
        track_id = track_template.track_id
        track_vec = track_template.template

        # If already assigned, return existing assignment
        if track_id in self._track_to_person:
            person_id = self._track_to_person[track_id]
            return PersonAssignment(
                track_id=track_id,
                person_id=person_id,
                is_new_person=False,
                top1_similarity=-1.0,  # Already assigned, not recalculated
                top2_similarity=None,
                margin=None,
                threshold=self.similarity_threshold,
                margin_threshold=self.margin_threshold,
                decision="already_assigned",
            )

        # Calculate similarity with all persons
        similarities = []
        for pid, person in self._persons.items():
            sim = float(np.dot(track_vec, person.template))
            similarities.append((pid, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top1 and top2
        top1_pid, top1_sim = similarities[0] if similarities else (None, -1.0)
        top2_sim = similarities[1][1] if len(similarities) > 1 else None
        margin = (top1_sim - top2_sim) if top2_sim is not None else None

        # Decision logic
        if top1_sim >= self.similarity_threshold:
            if margin is None or margin >= self.margin_threshold:
                # Match successful
                person_id = top1_pid
                is_new = False
                decision = "matched"

                # Update person template
                if self.update_template:
                    self._update_person_template(person_id, track_template)

                # Add track to person association
                self._persons[person_id].track_ids.append(track_id)
                self._persons[person_id].last_seen_frame = track_template.last_frame_id
            else:
                # Insufficient margin, create new person
                person_id = self._create_person(track_template)
                is_new = True
                decision = "margin_fail"
        else:
            # Insufficient similarity, create new person
            person_id = self._create_person(track_template)
            is_new = True
            decision = "new_person"

        # Record mapping
        self._track_to_person[track_id] = person_id

        # Update person info in track_template
        track_template.person_id = person_id
        track_template.similarity_to_person = top1_sim if not is_new else -1.0

        # Create assignment result
        assignment = PersonAssignment(
            track_id=track_id,
            person_id=person_id,
            is_new_person=is_new,
            top1_similarity=top1_sim if top1_pid is not None else -1.0,
            top2_similarity=top2_sim,
            margin=margin,
            threshold=self.similarity_threshold,
            margin_threshold=self.margin_threshold,
            decision=decision,
        )

        # Write log
        self._log_assignment(assignment)

        return assignment

    def _create_person(self, track_template: TrackTemplate) -> int:
        """Create new person."""
        person_id = self._next_person_id
        self._next_person_id += 1

        person = PersonTemplate(
            person_id=person_id,
            template=track_template.template.copy(),
            track_ids=[track_template.track_id],
            sample_count=track_template.sample_count,
            first_seen_frame=track_template.first_frame_id,
            last_seen_frame=track_template.last_frame_id,
        )
        self._persons[person_id] = person

        return person_id

    def _update_person_template(
        self, person_id: int, track_template: TrackTemplate
    ) -> None:
        """Update person template (sliding average)."""
        person = self._persons.get(person_id)
        if person is None:
            return

        # Sliding average: new = old * (1-w) + track * w
        w = self.update_weight
        new_template = person.template * (1 - w) + track_template.template * w

        # L2 normalize
        norm = np.linalg.norm(new_template)
        if norm > 0:
            new_template = new_template / norm

        person.template = new_template.astype(np.float32)
        person.sample_count += track_template.sample_count

    def get_person_id(self, track_id: int) -> Optional[int]:
        """Get person_id corresponding to track."""
        return self._track_to_person.get(track_id)

    def get_person(self, person_id: int) -> Optional[PersonTemplate]:
        """Get person template."""
        return self._persons.get(person_id)

    def get_all_persons(self) -> Dict[int, PersonTemplate]:
        """Get all persons."""
        return self._persons.copy()

    def get_person_count(self) -> int:
        """Get current person count."""
        return len(self._persons)

    def _log_assignment(self, assignment: PersonAssignment) -> None:
        """Write assignment log."""
        if self._log_file:
            self._log_file.write(
                json.dumps(assignment.to_dict(), ensure_ascii=False) + "\n"
            )
            self._log_file.flush()

    def reset(self) -> None:
        """Reset session."""
        self._persons.clear()
        self._track_to_person.clear()
        self._next_person_id = 1
