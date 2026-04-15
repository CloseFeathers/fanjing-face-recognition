"""Configuration dataclasses, constants, and security utility functions."""

from __future__ import annotations

import functools
import hmac
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

import numpy as np
from flask import jsonify, request

logger = logging.getLogger(__name__)

# ======================================================================
# Path Constraints
# ======================================================================
MODELS_DIR = Path("models").resolve()
UPLOAD_DIR = Path("uploads")

_allowed_video_dirs_str = os.environ.get("ALLOWED_VIDEO_DIRS", "uploads")
ALLOWED_VIDEO_DIRS = [Path(d.strip()).resolve() for d in _allowed_video_dirs_str.split(",")]

ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".webm"}

OUTPUT_DIR = os.environ.get("FACE_OUTPUT_DIR", "output")
FACES_DIR = os.path.join(OUTPUT_DIR, "faces")
REGISTERED_DB_DIR = os.path.join(OUTPUT_DIR, "registered_db")


def _validate_model_path(path_str: str, allowed_ext: str = ".onnx") -> Path:
    p = Path(path_str).resolve()
    if not p.is_relative_to(MODELS_DIR):
        raise ValueError("Model path is not in the allowed directory")
    if p.suffix.lower() != allowed_ext:
        raise ValueError(f"Model file must be in {allowed_ext} format")
    if not p.exists():
        raise FileNotFoundError("Specified model file does not exist")
    return p


# ======================================================================
# API Key Authentication
# ======================================================================
def _get_or_create_api_key() -> str:
    """Get API Key: prioritize env var, then persistent file, finally generate new key."""
    # 1. Prioritize environment variable
    if env_key := os.environ.get("FACE_API_KEY"):
        return env_key

    # 2. Try to read from file (project root .api_key file)
    key_file = Path(__file__).parent.parent.parent / ".api_key"
    if key_file.exists():
        stored_key = key_file.read_text().strip()
        if stored_key:
            return stored_key

    # 3. First run, generate and save
    new_key = secrets.token_urlsafe(32)
    try:
        key_file.write_text(new_key)
        logger.info(f"Generated API Key and saved to {key_file}")
    except OSError as e:
        logger.warning(f"Cannot save API Key to file: {e}, using temporary key")
    return new_key


API_KEY = _get_or_create_api_key()


def require_api_key(f: F) -> F:
    """API Key authentication decorator."""
    @functools.wraps(f)
    def decorated(*args: Any, **kwargs: Any) -> Any:
        key = request.headers.get("X-API-Key") or request.args.get("api_key")
        if not hmac.compare_digest(key or "", API_KEY):
            return jsonify({"ok": False, "error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated  # type: ignore[return-value]


# ======================================================================
# Video Stream Signing and Concurrency Limits
# ======================================================================
STREAM_SIGNATURE_EXPIRE_SEC = 300  # Signature validity period (seconds)
STREAM_SIGNATURE_LENGTH = 16      # Signature truncation length
MAX_CONCURRENT_STREAMS = 3        # Maximum concurrent video streams

_stream_lock = threading.Lock()
_active_streams = 0
_max_streams = MAX_CONCURRENT_STREAMS


def _verify_stream_signature() -> bool:
    ts = request.args.get("ts", "")
    sig = request.args.get("sig", "")
    try:
        ts_int = int(ts)
    except (ValueError, TypeError):
        return False
    if abs(time.time() - ts_int) > STREAM_SIGNATURE_EXPIRE_SEC:
        return False
    expected = hmac.new(
        API_KEY.encode(), ts.encode(), "sha256"
    ).hexdigest()[:STREAM_SIGNATURE_LENGTH]
    return hmac.compare_digest(sig, expected)


# ======================================================================
# Business Configuration Dataclasses
# ======================================================================
@dataclass
class CreditGateConfig:
    """Credit Gate: Control track embedding eligibility via credit score."""
    enabled: bool = True
    credit_increment: float = 1.0
    credit_decrement: float = 0.5
    credit_threshold: float = 3.0
    credit_max: float = 10.0


@dataclass
class Module5Config:
    """Module 5: Session identity state machine + auto-registration."""
    enabled: bool = False
    known_threshold: float = 0.55
    band_threshold: float = 0.35
    margin_threshold: float = 0.10


class EmbedReason:
    NEW_SAMPLE = "new_sample"
    SKIPPED_NO_DETECT = "skipped_no_detect"
    SKIPPED_NO_NEW_SAMPLE = "skipped_no_new_sample"
    SKIPPED_COOLDOWN = "skipped_cooldown"
    SKIPPED_CACHED = "skipped_cached"
    SKIPPED_QUALITY_FAIL = "skipped_quality_fail"
    SKIPPED_EMBED_DISABLED = "skipped_embed_disabled"
    SKIPPED_LOW_CREDIT = "skipped_low_credit"
    ERROR = "error"


@dataclass
class TrackEmbedState:
    """Per-track embedding state."""
    last_embed_time: float = 0.0
    last_embed_frame: int = 0
    cached_person_id: Optional[int] = None
    cached_similarity: Optional[float] = None
    cached_embedding: Optional[np.ndarray] = None
    embed_count: int = 0
