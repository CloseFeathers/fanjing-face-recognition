"""
Module 5: Session Identity State Machine

Tri-state:
  KNOWN_STRONG:    Clear match to registered identity (R#y), direct binding
  AMBIGUOUS:       Fuzzy range, continue observing in current session, may transition as template improves
  UNKNOWN_STRONG:  Clearly not an existing identity; auto-register in current session if register_ready

Judgment logic (based on top1/top2 similarity with RegisteredDB):
  KNOWN_STRONG:    top1 >= known_threshold and sufficient margin
  AMBIGUOUS:       top1 in [band_threshold, known_threshold) or insufficient margin
  UNKNOWN_STRONG:  top1 < band_threshold or official DB is empty
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class IdentityState(str, Enum):
    KNOWN_STRONG = "KNOWN_STRONG"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN_STRONG = "UNKNOWN_STRONG"


@dataclass
class IdentityDecision:
    """Identity decision result."""
    session_person_id: int
    identity_state: IdentityState
    top1_known_person_id: Optional[int]
    top1_score: float
    top2_known_person_id: Optional[int]
    top2_score: Optional[float]
    margin: Optional[float]
    decision_reason: str
    timestamp_ms: float
    frame_id: int
    registered_identity_id: Optional[int] = None

    def to_dict(self) -> Dict:
        d = {
            "session_person_id": self.session_person_id,
            "identity_state": self.identity_state.value,
            "top1_known_person_id": self.top1_known_person_id,
            "top1_score": round(self.top1_score, 4) if self.top1_score is not None else None,
            "top2_known_person_id": self.top2_known_person_id,
            "top2_score": round(self.top2_score, 4) if self.top2_score is not None else None,
            "margin": round(self.margin, 4) if self.margin is not None else None,
            "decision_reason": self.decision_reason,
            "timestamp_ms": round(self.timestamp_ms, 3),
            "frame_id": self.frame_id,
        }
        if self.registered_identity_id is not None:
            d["registered_identity_id"] = self.registered_identity_id
        return d


@dataclass
class IdentityConfig:
    known_threshold: float = 0.55
    band_threshold: float = 0.35
    margin_threshold: float = 0.10
    empty_db_as_unknown_strong: bool = True


class RegisteredPersonDB:
    """Registered official identity database. Supports session auto-registration + cross-session persistence."""

    def __init__(self, db_dir: Optional[str] = None):
        self._persons: Dict[int, np.ndarray] = {}
        self._metadata: Dict[int, Dict] = {}
        self._next_id: int = 1
        self._db_dir = db_dir or "output/registered_db"

    def register(
        self,
        template: np.ndarray,
        session_person_id: int,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Register new identity, returns registered_identity_id (R#y)."""
        rid = self._next_id
        self._next_id += 1

        norm = np.linalg.norm(template)
        if norm > 0:
            template = template / norm

        self._persons[rid] = template.astype(np.float32)
        self._metadata[rid] = {
            **(metadata or {}),
            "source_session_person_id": session_person_id,
            "registered_at": time.time(),
        }
        return rid

    def add_person(self, template: np.ndarray, metadata: Optional[Dict] = None) -> int:
        """Compatible with old interface (manual add)."""
        return self.register(template, session_person_id=-1, metadata=metadata)

    def get_template(self, person_id: int) -> Optional[np.ndarray]:
        return self._persons.get(person_id)

    def get_metadata(self, person_id: int) -> Dict:
        return self._metadata.get(person_id, {})

    def get_all_templates(self) -> Dict[int, np.ndarray]:
        return self._persons.copy()

    def is_empty(self) -> bool:
        return len(self._persons) == 0

    def count(self) -> int:
        return len(self._persons)

    def update_template(self, person_id: int, new_template: np.ndarray, weight: float = 0.1):
        """Conservatively update registered identity template."""
        old = self._persons.get(person_id)
        if old is None:
            return
        updated = old * (1.0 - weight) + new_template.astype(np.float32) * weight
        norm = np.linalg.norm(updated)
        if norm > 0:
            updated = updated / norm
        self._persons[person_id] = updated
        self._metadata.setdefault(person_id, {})["last_updated_at"] = time.time()

    def check_and_merge(self, updated_rid: int, merge_threshold: float = 0.75) -> Optional[Tuple[int, int]]:
        """Check if updated_rid similarity exceeds merge threshold with other identities.

        Returns:
            (main_rid, absorbed_rid) if merged, else None
        """
        updated_tmpl = self._persons.get(updated_rid)
        if updated_tmpl is None:
            return None

        for other_rid, other_tmpl in list(self._persons.items()):
            if other_rid == updated_rid:
                continue
            sim = float(np.dot(updated_tmpl, other_tmpl))
            if sim >= merge_threshold:
                main_rid = min(updated_rid, other_rid)
                absorbed_rid = max(updated_rid, other_rid)
                self._merge(main_rid, absorbed_rid)
                return (main_rid, absorbed_rid)

        return None

    def _merge(self, main_rid: int, absorbed_rid: int):
        """Absorb absorbed_rid into main_rid (without losing information)."""
        main_tmpl = self._persons[main_rid]
        abs_tmpl = self._persons[absorbed_rid]

        merged = main_tmpl * 0.7 + abs_tmpl * 0.3
        norm = np.linalg.norm(merged)
        if norm > 0:
            merged = merged / norm
        self._persons[main_rid] = merged

        main_meta = self._metadata.setdefault(main_rid, {})
        merged_from = main_meta.setdefault("merged_from", [])
        merged_from.append({
            "absorbed_rid": absorbed_rid,
            "absorbed_metadata": self._metadata.get(absorbed_rid, {}),
            "merged_at": time.time(),
        })

        del self._persons[absorbed_rid]
        self._metadata.pop(absorbed_rid, None)

    def save(self):
        """Persist to disk (templates + metadata), using temp file + atomic replace."""
        os.makedirs(self._db_dir, exist_ok=True)
        templates_path = os.path.join(self._db_dir, "templates.npz")
        meta_path = os.path.join(self._db_dir, "metadata.json")

        if not self._persons:
            return

        ids = sorted(self._persons.keys())
        arrays = {str(rid): self._persons[rid] for rid in ids}

        tmp_tpl = os.path.join(self._db_dir, "_tmp_templates.npz")
        tmp_meta = meta_path + ".tmp"

        np.savez(tmp_tpl, **arrays)

        meta_out = {"next_id": self._next_id, "persons": {}}
        for rid in ids:
            meta_out["persons"][str(rid)] = self._metadata.get(rid, {})

        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta_out, f, ensure_ascii=False, indent=2)

        os.replace(tmp_tpl, templates_path)
        os.replace(tmp_meta, meta_path)

        logger.info("[RegisteredDB] saved %s identities to %s", len(ids), self._db_dir)

    def load(self):
        """Load registered identities from disk."""
        templates_path = os.path.join(self._db_dir, "templates.npz")
        meta_path = os.path.join(self._db_dir, "metadata.json")

        if not os.path.exists(templates_path) or not os.path.exists(meta_path):
            return

        data = np.load(templates_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_in = json.load(f)

        self._next_id = meta_in.get("next_id", 1)
        persons_meta = meta_in.get("persons", {})

        for key, arr in data.items():
            rid = int(key)
            self._persons[rid] = arr.astype(np.float32)
            self._metadata[rid] = persons_meta.get(key, {})

        logger.info("[RegisteredDB] loaded %s identities from %s", len(self._persons), self._db_dir)

    def clear(self):
        self._persons.clear()
        self._metadata.clear()
        self._next_id = 1


class IdentityJudge:
    """Session identity judge."""

    def __init__(
        self,
        registered_db: Optional[RegisteredPersonDB] = None,
        config: Optional[IdentityConfig] = None,
        log_path: Optional[str] = "output/person_states.jsonl",
    ):
        self.registered_db = registered_db or RegisteredPersonDB()
        self.config = config or IdentityConfig()
        self.log_path = log_path
        self._person_states: Dict[int, IdentityState] = {}
        self._log_file = None

    def open(self) -> "IdentityJudge":
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            self._log_file = open(self.log_path, "a", encoding="utf-8")
        return self

    def close(self):
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def judge(
        self,
        session_person_id: int,
        template: np.ndarray,
        timestamp_ms: float,
        frame_id: int,
    ) -> IdentityDecision:
        cfg = self.config

        if self.registered_db.is_empty():
            state = IdentityState.UNKNOWN_STRONG if cfg.empty_db_as_unknown_strong else IdentityState.AMBIGUOUS
            decision = IdentityDecision(
                session_person_id=session_person_id,
                identity_state=state,
                top1_known_person_id=None,
                top1_score=-1.0,
                top2_known_person_id=None,
                top2_score=None,
                margin=None,
                decision_reason="registered_db_empty",
                timestamp_ms=timestamp_ms,
                frame_id=frame_id,
            )
            self._update_and_log(decision)
            return decision

        similarities = []
        for pid, known_template in self.registered_db.get_all_templates().items():
            sim = float(np.dot(template, known_template))
            similarities.append((pid, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)

        top1_pid, top1_score = similarities[0]
        top2_pid, top2_score = similarities[1] if len(similarities) > 1 else (None, None)
        margin = (top1_score - top2_score) if top2_score is not None else None

        identity_state: IdentityState
        decision_reason: str
        registered_identity_id: Optional[int] = None

        if top1_score >= cfg.known_threshold:
            if margin is None or margin >= cfg.margin_threshold:
                identity_state = IdentityState.KNOWN_STRONG
                registered_identity_id = top1_pid
                decision_reason = (
                    f"top1={top1_score:.3f}>=known_thresh={cfg.known_threshold},"
                    f"margin={'N/A' if margin is None else f'{margin:.3f}'}"
                    f">=margin_thresh={cfg.margin_threshold}"
                )
            else:
                identity_state = IdentityState.AMBIGUOUS
                decision_reason = (
                    f"top1={top1_score:.3f}>=known_thresh,but "
                    f"margin={margin:.3f}<margin_thresh={cfg.margin_threshold}"
                )
        elif top1_score >= cfg.band_threshold:
            identity_state = IdentityState.AMBIGUOUS
            decision_reason = (
                f"band_thresh={cfg.band_threshold}<=top1={top1_score:.3f}"
                f"<known_thresh={cfg.known_threshold}"
            )
        else:
            identity_state = IdentityState.UNKNOWN_STRONG
            decision_reason = f"top1={top1_score:.3f}<band_thresh={cfg.band_threshold}"

        decision = IdentityDecision(
            session_person_id=session_person_id,
            identity_state=identity_state,
            top1_known_person_id=top1_pid,
            top1_score=top1_score,
            top2_known_person_id=top2_pid,
            top2_score=top2_score,
            margin=margin,
            decision_reason=decision_reason,
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            registered_identity_id=registered_identity_id,
        )

        self._update_and_log(decision)
        return decision

    def _update_and_log(self, decision: IdentityDecision):
        old_state = self._person_states.get(decision.session_person_id)
        new_state = decision.identity_state
        if old_state != new_state:
            self._person_states[decision.session_person_id] = new_state
            if self._log_file:
                self._log_file.write(
                    json.dumps(decision.to_dict(), ensure_ascii=False) + "\n"
                )
                self._log_file.flush()

    def get_state(self, session_person_id: int) -> Optional[IdentityState]:
        return self._person_states.get(session_person_id)

    def get_state_counts(self) -> Dict[str, int]:
        counts = {"known_strong": 0, "ambiguous": 0, "unknown_strong": 0}
        for s in self._person_states.values():
            if s == IdentityState.KNOWN_STRONG:
                counts["known_strong"] += 1
            elif s == IdentityState.AMBIGUOUS:
                counts["ambiguous"] += 1
            elif s == IdentityState.UNKNOWN_STRONG:
                counts["unknown_strong"] += 1
        return counts

    def reset(self):
        self._person_states.clear()
