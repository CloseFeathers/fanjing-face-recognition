"""配置 dataclass、常量与安全工具函数。"""

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
from typing import Optional

import numpy as np
from flask import jsonify, request

logger = logging.getLogger(__name__)

# ======================================================================
# 路径约束
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
        raise ValueError("模型路径不在允许的目录中")
    if p.suffix.lower() != allowed_ext:
        raise ValueError(f"模型文件必须是 {allowed_ext} 格式")
    if not p.exists():
        raise FileNotFoundError("指定的模型文件不存在")
    return p


# ======================================================================
# API Key 认证
# ======================================================================
def _get_or_create_api_key() -> str:
    """获取 API Key：优先环境变量，其次持久化文件，最后生成新 Key"""
    # 1. 优先使用环境变量
    if env_key := os.environ.get("FACE_API_KEY"):
        return env_key

    # 2. 尝试从文件读取（项目根目录的 .api_key 文件）
    key_file = Path(__file__).parent.parent.parent / ".api_key"
    if key_file.exists():
        stored_key = key_file.read_text().strip()
        if stored_key:
            return stored_key

    # 3. 首次运行，生成并保存
    new_key = secrets.token_urlsafe(32)
    try:
        key_file.write_text(new_key)
        logger.info(f"已生成 API Key 并保存到 {key_file}")
    except OSError as e:
        logger.warning(f"无法保存 API Key 到文件: {e}，使用临时 Key")
    return new_key


API_KEY = _get_or_create_api_key()


def require_api_key(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key") or request.args.get("api_key")
        if not hmac.compare_digest(key or "", API_KEY):
            return jsonify({"ok": False, "error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# ======================================================================
# 视频流签名与并发限制
# ======================================================================
_stream_lock = threading.Lock()
_active_streams = 0
_max_streams = 3


def _verify_stream_signature() -> bool:
    ts = request.args.get("ts", "")
    sig = request.args.get("sig", "")
    try:
        ts_int = int(ts)
    except (ValueError, TypeError):
        return False
    if abs(time.time() - ts_int) > 300:
        return False
    expected = hmac.new(API_KEY.encode(), ts.encode(), "sha256").hexdigest()[:16]
    return hmac.compare_digest(sig, expected)


# ======================================================================
# 业务配置 dataclass
# ======================================================================
@dataclass
class CreditGateConfig:
    """Credit Gate: 用信用积分控制 track 是否可进入 embedding。"""
    enabled: bool = True
    credit_increment: float = 1.0
    credit_decrement: float = 0.5
    credit_threshold: float = 3.0
    credit_max: float = 10.0


@dataclass
class Module5Config:
    """Module 5: Session 内身份状态机 + 自动注册。"""
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
    """Per-track embedding 状态"""
    last_embed_time: float = 0.0
    last_embed_frame: int = 0
    cached_person_id: Optional[int] = None
    cached_similarity: Optional[float] = None
    cached_embedding: Optional[np.ndarray] = None
    embed_count: int = 0
