"""
Web 测试前端后端 —— Flask MJPEG 流 + REST API + 实时统计。

Module 4b: 性能优化版
- 细粒度计时: align_ms, sample_ms, emb_ms, template_ms, person_ms
- Embedding 缓存: 同一 face_image_path 不重复推理
- Per-track cooldown: 每 500ms 或 N 帧限频
- 只在 detect=Y + 新样本时触发 embedding
- Fail-open: embedding 慢或出错不拖垮主视频流
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8")
            except Exception:
                pass

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

from .config import (
    ALLOWED_VIDEO_DIRS,
    ALLOWED_VIDEO_EXT,
    API_KEY,
    UPLOAD_DIR,
    CreditGateConfig,
    EmbedReason,
    Module5Config,
    TrackEmbedState,
    _validate_model_path,
    _verify_stream_signature,
    require_api_key,
)

logger = logging.getLogger(__name__)

from ..detectors.scrfd_detector import SCRFDDetector
from ..ingestion.camera_source import CameraSource
from ..ingestion.video_source import VideoSource
from ..tracking.bot_sort import BoTSORTConfig, BoTSORTTracker
from ..tracking.draw import draw_tracks
from ..tracking.track import TrackState

SPEAKING_AVAILABLE = False
MeshDetector = None
MouthAnalyzer = None
MouthWorker = None
SpeakingAnalyzer = None

try:
    from ..speaking.mesh_detector import MeshDetector as _MeshDetector
    from ..speaking.mouth_analyzer import MouthAnalyzer as _MouthAnalyzer
    from ..speaking.mouth_worker import MouthWorker as _MouthWorker
    from ..speaking.speaking_analyzer import SpeakingAnalyzer as _SpeakingAnalyzer
    MeshDetector = _MeshDetector
    MouthAnalyzer = _MouthAnalyzer
    MouthWorker = _MouthWorker
    SpeakingAnalyzer = _SpeakingAnalyzer
    SPEAKING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[Warning] Speaking 模块不可用 (mediapipe 未安装): {e}")
from ..alignment.aligner import FaceAligner
from ..alignment.quality import QualityConfig, QualityGate
from ..alignment.track_sampler import TrackSampler

# Embedding 模块 (可选导入)
EMBEDDING_AVAILABLE = False
MODULE5_AVAILABLE = False
ArcFaceEmbedder = None
TrackTemplateManager = None
PersonRegistry = None
EmbeddingLogger = None
# Module 5
IdentityState = None
IdentityJudge = None
IdentityConfig = None
RegisteredPersonDB = None
FaceCandidatePool = None
CandidateConfig = None
EmbeddingSample = None

try:
    from ..embedding import (
        ArcFaceEmbedder as _ArcFaceEmbedder,
    )
    from ..embedding import (
        EmbeddingLogger as _EmbeddingLogger,
    )
    from ..embedding import (
        PersonRegistry as _PersonRegistry,
    )
    from ..embedding import (
        TrackTemplateManager as _TrackTemplateManager,
    )
    ArcFaceEmbedder = _ArcFaceEmbedder
    TrackTemplateManager = _TrackTemplateManager
    PersonRegistry = _PersonRegistry
    EmbeddingLogger = _EmbeddingLogger
    EMBEDDING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[Warning] Embedding 模块不可用: {e}")

# Module 5: 三态身份判定 + 候选池
try:
    from ..embedding import (
        CandidateConfig as _CandidateConfig,
    )
    from ..embedding import (
        EmbeddingSample as _EmbeddingSample,
    )
    from ..embedding import (
        FaceCandidatePool as _FaceCandidatePool,
    )
    from ..embedding import (
        IdentityConfig as _IdentityConfig,
    )
    from ..embedding import (
        IdentityJudge as _IdentityJudge,
    )
    from ..embedding import (
        IdentityState as _IdentityState,
    )
    from ..embedding import (
        RegisteredPersonDB as _RegisteredPersonDB,
    )
    IdentityState = _IdentityState
    IdentityJudge = _IdentityJudge
    IdentityConfig = _IdentityConfig
    RegisteredPersonDB = _RegisteredPersonDB
    FaceCandidatePool = _FaceCandidatePool
    CandidateConfig = _CandidateConfig
    EmbeddingSample = _EmbeddingSample
    MODULE5_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[Warning] Module 5 不可用: {e}")

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB


@app.after_request
def _add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if "text/html" in response.content_type:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: blob:; "
            "script-src 'self' 'unsafe-inline'; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
    return response


# ======================================================================
# 全局状态
# ======================================================================

class PipelineState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.source = None
        self.detector: Optional[SCRFDDetector] = None
        self.tracker: Optional[BoTSORTTracker] = None
        self.aligner: Optional[FaceAligner] = None
        self.qgate: Optional[QualityGate] = None
        self.sampler: Optional[TrackSampler] = None
        self.running = False
        self.mode = ""
        self.det_every_n = 1
        self.align_enabled = False
        self.sample_min_det = 0.60

        # 说话状态检测
        self.speaking_enabled: bool = False
        self.mouth_worker: Optional[MouthWorker] = None
        self.mouth_states: Dict[int, str] = {}
        self._last_frame = None
        self.show_hud: bool = True

        # Embedding 模块 (Module 4)
        self.embed_enabled = False
        self.embedder = None
        self.template_mgr = None
        self.person_registry = None
        self.emb_logger = None

        # Embedding 缓存: face_image_path -> embedding (np.ndarray)
        self._embed_cache: Dict[str, np.ndarray] = {}
        # Per-track 状态
        self._track_embed_state: Dict[int, TrackEmbedState] = {}
        # Cooldown 配置
        self.embed_cooldown_ms: float = 1000.0  # 每 track 1000ms 限频 (更激进)
        # embed_cooldown_frames 已移除，仅用 cooldown_ms 控制
        self.max_embed_per_frame: int = 1       # 每帧最多处理 1 个 embedding
        self.max_embed_per_track: int = 5       # 每 track 最多提取 5 次 embedding（足够生成 template）

        # track_id -> person_id 映射 (用于显示)
        self.track_to_person: Dict[int, int] = {}
        # track_id -> similarity 映射 (用于显示)
        self.track_similarities: Dict[int, float] = {}

        # 基础统计
        self.fps = 0.0
        self.detect_ms = 0.0
        self.track_ms = 0.0
        self.num_faces = 0
        self.num_tracks = 0
        self.num_persons = 0
        self.frame_id = 0
        self.timestamp_ms = 0.0
        self.dropped_frames = 0
        self.width = 0
        self.height = 0
        self.source_id = ""
        self.did_detect = True

        # 细粒度计时统计 (Module 4b)
        self.align_ms = 0.0      # 对齐耗时
        self.sample_ms = 0.0     # 采样保存耗时
        self.emb_ms = 0.0        # embedding 推理耗时
        self.template_ms = 0.0   # template 更新耗时
        self.person_ms = 0.0     # person matching 耗时
        self.encode_ms = 0.0     # JPEG 编码耗时
        self.total_frame_ms = 0.0  # 总帧处理耗时

        # Alignment 统计
        self.align_evaluated = 0
        self.align_passed = 0
        self.align_saved = 0

        # Embedding 统计 (增强)
        self.embed_extracted = 0
        self.templates_created = 0
        self.emb_error_count = 0
        self.last_error = ""

        # 缓存统计 (新增)
        self.emb_cache_hit = 0
        self.emb_cache_miss = 0
        self.emb_cooldown_skip = 0  # 因 cooldown 跳过的次数

        # Embedding 触发日志 (新增)
        self.embed_trigger_log: list[dict] = []
        self.embed_trigger_log_max = 100

        # ======== Credit Gate (信用积分制) ========
        self.credit_gate_cfg = CreditGateConfig()

        # ======== 异步身份 Worker ========
        self.identity_worker: Optional["IdentityWorker"] = None

        # ======== MODULE 5: 三态身份判定 + 候选池 ========
        self.m5_cfg = Module5Config()  # 默认关闭
        self.registered_db = None      # RegisteredPersonDB (正式身份库)
        self.identity_judge = None     # IdentityJudge (三态判定器)
        self.candidate_pool = None     # FaceCandidatePool (候选池)
        # session_person_id -> IdentityState
        self.person_identity_states: Dict[int, str] = {}
        # session_person_id -> candidate_id
        self.person_to_candidate: Dict[int, int] = {}
        # session_person_id -> registered_identity_id (P#x → R#y)
        self.person_to_registered: Dict[int, int] = {}
        # Module 5 统计
        self.m5_known_strong_count = 0
        self.m5_ambiguous_count = 0
        self.m5_unknown_strong_count = 0
        self.m5_registered_count = 0
        self.m5_candidate_count = 0
        self.m5_ready_candidate_count = 0
        # AMBIGUOUS 超时注册
        self._ambiguous_since: Dict[int, float] = {}
        self.amb_timeout_sec: float = 15.0

        self._fps_start = 0.0
        self._fps_count = 0

        self.log_entries: list[dict] = []
        self.log_max = 200

        # 线程安全: identity worker 写入后原子替换引用，主线程只读快照
        self._identity_snapshot: dict = {
            "track_to_person": {},
            "track_similarities": {},
            "person_identity_states": {},
            "person_to_registered": {},
        }

    def get_track_state(self, track_id: int) -> TrackEmbedState:
        """获取或创建 track 的 embedding 状态"""
        if track_id not in self._track_embed_state:
            self._track_embed_state[track_id] = TrackEmbedState()
        return self._track_embed_state[track_id]

    def check_cooldown(self, track_id: int, current_time: float, current_frame: int) -> bool:
        """检查是否在 cooldown 中。返回 True 表示可以执行 embedding"""
        ts = self.get_track_state(track_id)

        time_elapsed = (current_time - ts.last_embed_time) * 1000
        if time_elapsed < self.embed_cooldown_ms:
            return False
        return True

    def update_embed_time(self, track_id: int, current_time: float, current_frame: int):
        """更新 track 的 embedding 时间"""
        ts = self.get_track_state(track_id)
        ts.last_embed_time = current_time
        ts.last_embed_frame = current_frame

    def get_cached_embedding(self, path: str) -> Optional[np.ndarray]:
        """获取缓存的 embedding"""
        return self._embed_cache.get(path)

    def set_cached_embedding(self, path: str, embedding: np.ndarray):
        """设置缓存的 embedding"""
        self._embed_cache[path] = embedding

    def log_embed_trigger(self, frame_id: int, track_id: int, reason: str,
                          details: Optional[str] = None):
        """记录 embedding 触发日志"""
        entry = {
            "frame_id": frame_id,
            "track_id": track_id,
            "reason": reason,
            "timestamp": time.time(),
        }
        if details:
            entry["details"] = details

        self.embed_trigger_log.append(entry)
        if len(self.embed_trigger_log) > self.embed_trigger_log_max:
            self.embed_trigger_log = self.embed_trigger_log[-self.embed_trigger_log_max:]

    def update_stats_from_ft(self, ft, dropped):
        self.detect_ms = ft.detect_time_ms
        self.track_ms = ft.track_time_ms
        self.num_tracks = ft.num_tracks
        self.num_faces = len([t for t in ft.tracks if t.get("det_score") is not None]) if ft.did_detect else self.num_faces
        self.frame_id = ft.frame_id
        self.timestamp_ms = ft.timestamp_ms
        self.dropped_frames = dropped
        self.width = ft.width
        self.height = ft.height
        self.source_id = ft.source_id
        self.did_detect = ft.did_detect

        self._fps_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_start
        if elapsed >= 1.0:
            self.fps = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_start = now

    def update_align_stats(self):
        if self.sampler:
            self.align_evaluated = self.sampler.total_evaluated
            self.align_passed = self.sampler.total_passed
            self.align_saved = self.sampler.total_saved

    def update_embed_stats(self):
        if self.person_registry:
            try:
                self.num_persons = self.person_registry.get_person_count()
            except Exception as e:
                logger.error(f"[Stats] person count error: {e}")
        if self.template_mgr:
            try:
                self.templates_created = len(self.template_mgr.get_all_templates())
            except Exception as e:
                logger.error(f"[Stats] template count error: {e}")

    def record_emb_error(self, error_msg: str):
        """记录 embedding 错误"""
        self.emb_error_count += 1
        self.last_error = error_msg[:200]

    def append_log(self, entry: dict):
        self.log_entries.append(entry)
        if len(self.log_entries) > self.log_max:
            self.log_entries = self.log_entries[-self.log_max:]

    def reset(self):
        self.fps = 0.0
        self.detect_ms = 0.0
        self.track_ms = 0.0
        self.num_faces = 0
        self.num_tracks = 0
        self.num_persons = 0
        self.frame_id = 0
        self.timestamp_ms = 0.0
        self.dropped_frames = 0
        self.did_detect = True

        # 细粒度计时重置
        self.align_ms = 0.0
        self.sample_ms = 0.0
        self.emb_ms = 0.0
        self.template_ms = 0.0
        self.person_ms = 0.0
        self.encode_ms = 0.0
        self.total_frame_ms = 0.0

        self.align_evaluated = 0
        self.align_passed = 0
        self.align_saved = 0
        self.embed_extracted = 0
        self.templates_created = 0
        self.emb_error_count = 0
        self.last_error = ""

        # 缓存统计重置
        self.emb_cache_hit = 0
        self.emb_cache_miss = 0
        self.emb_cooldown_skip = 0

        # Module 5 统计重置
        self.person_identity_states.clear()
        self.person_to_candidate.clear()
        self.person_to_registered.clear()
        self.m5_known_strong_count = 0
        self.m5_ambiguous_count = 0
        self.m5_unknown_strong_count = 0
        self.m5_registered_count = 0
        self.m5_candidate_count = 0
        self.m5_ready_candidate_count = 0
        self._ambiguous_since.clear()
        if self.identity_judge:
            self.identity_judge.reset()
        if self.candidate_pool:
            self.candidate_pool.reset()

        self.track_to_person.clear()
        self.track_similarities.clear()
        self._embed_cache.clear()
        self._track_embed_state.clear()
        self.embed_trigger_log.clear()
        self.log_entries.clear()
        self._fps_start = time.monotonic()
        self._fps_count = 0

        self._identity_snapshot = {
            "track_to_person": {},
            "track_similarities": {},
            "person_identity_states": {},
            "person_to_registered": {},
        }

        self.mouth_states.clear()
        self._person_avatars = {}
        self._person_names = {}
        self._persons_cache = []
        self._last_frame = None


state = PipelineState()
_detector_loaded = False


def ensure_detector(model_path="models/det_10g.onnx",
                    det_thresh=0.5, det_size=640,
                    emit_thresh=0.15):
    global _detector_loaded
    if state.detector is None or not _detector_loaded:
        state.detector = SCRFDDetector(
            model_path=model_path,
            det_size=(det_size, det_size),
            det_thresh=det_thresh,
            emit_thresh=emit_thresh,
        )
        state.detector.load()
        _detector_loaded = True
    return state.detector


# ======================================================================
# IdentityWorker: 异步身份链路 (后台单线程)
# ======================================================================

class IdentityWorker:
    """将对齐→质量→信用→采样→embedding→模板→person→M5 放入后台线程。

    主线程 submit() 最新的 (frame_image, ft, stracks) 到任务槽,
    worker 线程取走并执行 _alignment_step。latest-only: 新帧覆盖旧帧。
    """

    def __init__(self):
        self._pending = None   # (frame_image, ft, stracks) | None
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._pending = None
        self._event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="identity-worker")
        self._thread.start()

    def stop(self):
        self._running = False
        self._event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def submit(self, frame_image, ft, active_stracks):
        """覆写式提交: 新帧直接覆盖旧的待处理帧。"""
        with self._lock:
            self._pending = (frame_image, ft, active_stracks)
        self._event.set()

    def _loop(self):
        while self._running:
            self._event.wait(timeout=1.0)
            self._event.clear()

            with self._lock:
                job = self._pending
                self._pending = None

            if job is None:
                continue

            frame_image, ft, active_stracks = job
            try:
                _alignment_step(frame_image, ft, active_stracks=active_stracks)
            except Exception as e:
                import traceback
                state.record_emb_error(f"async_identity: {e}")
                traceback.print_exc()


# ======================================================================
# 对齐 + Embedding 步骤 (性能优化版)
# ======================================================================

def _alignment_step(frame_image, ft, active_stracks=None):
    """对检测帧中 confirmed + 有 kps5 的 track 执行对齐采样和 embedding 提取。

    性能优化 (Module 4b):
    1. 只在 detect=Y 时执行
    2. 只在有新的 quality_passed 样本时触发 embedding
    3. Per-track cooldown 限频 (1000ms)
    4. 每帧最多处理 1 个 embedding (避免多人时雪崩)
    5. 每 track 最多提取 N 次 embedding (足够生成 template)
    6. 基于 track_id 缓存 embedding (同一 track 复用)

    Credit Gate (信用积分制, 替代 Face Gating):
    7. quality 通过 → credit += increment; 失败 → credit -= decrement
    8. credit >= threshold 才允许 embedding
    9. credit 不足时仍采样 (积累样本), 但不做 embedding
    10. 不 REMOVE 任何轨迹
    """
    # 如果不是检测帧，跳过整个步骤
    if not ft.did_detect:
        return

    # 如果 alignment 未启用，跳过
    if not state.align_enabled:
        return

    tracker = state.tracker
    aligner = state.aligner
    qgate = state.qgate
    sampler = state.sampler

    if not all([tracker, aligner, qgate, sampler]):
        return

    # Embedding 组件 (可能为 None)
    embedder = state.embedder if state.embed_enabled else None
    template_mgr = state.template_mgr if state.embed_enabled else None
    person_registry = state.person_registry if state.embed_enabled else None
    emb_logger = state.emb_logger if state.embed_enabled else None

    current_time = time.monotonic()
    current_frame = ft.frame_id
    credit_cfg = state.credit_gate_cfg

    # 累计计时
    total_align_ms = 0.0
    total_sample_ms = 0.0
    total_emb_ms = 0.0
    total_template_ms = 0.0
    total_person_ms = 0.0

    # 每帧 embedding 计数限制
    embed_count_this_frame = 0

    stracks = active_stracks if active_stracks is not None else tracker.last_active_stracks
    for strack in stracks:
        if strack.state != TrackState.CONFIRMED:
            continue

        # kps5/det_score 缺失 → 扣信用, 跳过
        if strack.kps5 is None or strack.det_score is None:
            if credit_cfg.enabled:
                strack.face_valid_credit = max(0.0, strack.face_valid_credit - credit_cfg.credit_decrement)
            continue

        # 阈值 D: 弱框跳过 alignment (但在 tracker 中仍有用)
        if strack.det_score < state.sample_min_det:
            if credit_cfg.enabled:
                strack.face_valid_credit = max(0.0, strack.face_valid_credit - credit_cfg.credit_decrement)
            continue

        track_id = strack.track_id
        ts = state.get_track_state(track_id)

        # Step 1: 对齐
        t0 = time.monotonic()
        try:
            aligned = aligner.align(frame_image, strack.kps5)
            if aligned is None:
                if credit_cfg.enabled:
                    strack.face_valid_credit = max(0.0, strack.face_valid_credit - credit_cfg.credit_decrement)
                continue
        except Exception as e:
            state.record_emb_error(f"align: {e}")
            continue
        total_align_ms += (time.monotonic() - t0) * 1000

        # Step 2: 质量评估
        try:
            bbox = strack.bbox_xyxy_clipped(ft.width, ft.height)
            quality = qgate.evaluate(aligned, bbox, strack.det_score, strack.kps5)
        except Exception as e:
            state.record_emb_error(f"quality: {e}")
            continue

        # Step 3: 更新信用分 (Credit Gate)
        if credit_cfg.enabled:
            if quality.passed:
                strack.face_valid_credit = min(
                    credit_cfg.credit_max,
                    strack.face_valid_credit + credit_cfg.credit_increment,
                )
            else:
                strack.face_valid_credit = max(
                    0.0,
                    strack.face_valid_credit - credit_cfg.credit_decrement,
                )

        # Step 4: 采样 (quality 通过就采样, 不管信用分)
        t0 = time.monotonic()
        info = None
        if quality.passed:
            try:
                info = sampler.try_add(
                    track_id=track_id,
                    frame_id=ft.frame_id,
                    timestamp_ms=ft.timestamp_ms,
                    aligned_face=aligned,
                    quality=quality,
                )
                if info is not None:
                    strack.ever_sampled = True
            except Exception as e:
                state.record_emb_error(f"sampler: {e}")
        total_sample_ms += (time.monotonic() - t0) * 1000

        # quality 不通过 → 跳过 embedding, 使用缓存
        if not quality.passed:
            state.log_embed_trigger(current_frame, track_id, EmbedReason.SKIPPED_QUALITY_FAIL)
            if ts.cached_person_id is not None:
                state.track_to_person[track_id] = ts.cached_person_id
                if ts.cached_similarity is not None:
                    state.track_similarities[track_id] = ts.cached_similarity
            continue

        # 没有新样本 → 使用缓存
        if info is None:
            state.log_embed_trigger(current_frame, track_id, EmbedReason.SKIPPED_NO_NEW_SAMPLE)
            if ts.cached_person_id is not None:
                state.track_to_person[track_id] = ts.cached_person_id
                if ts.cached_similarity is not None:
                    state.track_similarities[track_id] = ts.cached_similarity
            continue

        # embedding 未启用
        if embedder is None:
            state.log_embed_trigger(current_frame, track_id, EmbedReason.SKIPPED_EMBED_DISABLED)
            continue

        # Step 5: Credit Gate — 信用分不足, 跳过 embedding
        if credit_cfg.enabled and strack.face_valid_credit < credit_cfg.credit_threshold:
            state.log_embed_trigger(current_frame, track_id, EmbedReason.SKIPPED_LOW_CREDIT,
                                   f"credit={strack.face_valid_credit:.1f}<{credit_cfg.credit_threshold}")
            if ts.cached_person_id is not None:
                state.track_to_person[track_id] = ts.cached_person_id
                if ts.cached_similarity is not None:
                    state.track_similarities[track_id] = ts.cached_similarity
            continue

        # Step 6: 每帧 embedding 数量限制
        if embed_count_this_frame >= state.max_embed_per_frame:
            state.log_embed_trigger(current_frame, track_id, EmbedReason.SKIPPED_COOLDOWN,
                                   "max_per_frame reached")
            if ts.cached_person_id is not None:
                state.track_to_person[track_id] = ts.cached_person_id
                if ts.cached_similarity is not None:
                    state.track_similarities[track_id] = ts.cached_similarity
            continue

        # Step 7: 每 track embedding 次数限制 (含条件解锁)
        if ts.embed_count >= state.max_embed_per_track:
            should_unlock = False
            if state.m5_cfg.enabled:
                pid = state.track_to_person.get(track_id)
                if pid is not None:
                    id_state = state.person_identity_states.get(pid)
                    if id_state in ("AMBIGUOUS", "UNKNOWN_STRONG"):
                        time_since_last = (current_time - ts.last_embed_time) * 1000
                        if time_since_last >= 5000:
                            should_unlock = True

            if not should_unlock:
                state.log_embed_trigger(current_frame, track_id, EmbedReason.SKIPPED_CACHED,
                                       f"track reached max {state.max_embed_per_track}")
                if ts.cached_person_id is not None:
                    state.track_to_person[track_id] = ts.cached_person_id
                    if ts.cached_similarity is not None:
                        state.track_similarities[track_id] = ts.cached_similarity
                state.emb_cache_hit += 1
                continue

        # Step 8: Cooldown 检查
        if not state.check_cooldown(track_id, current_time, current_frame):
            state.emb_cooldown_skip += 1
            state.log_embed_trigger(current_frame, track_id, EmbedReason.SKIPPED_COOLDOWN)
            if ts.cached_person_id is not None:
                state.track_to_person[track_id] = ts.cached_person_id
                if ts.cached_similarity is not None:
                    state.track_similarities[track_id] = ts.cached_similarity
            continue

        # Step 9: 执行 embedding 推理
        state.emb_cache_miss += 1
        t0 = time.monotonic()
        try:
            embedding = embedder.extract(aligned)
        except Exception as e:
            state.record_emb_error(f"embedder: {e}")
            state.log_embed_trigger(current_frame, track_id, EmbedReason.ERROR, str(e))
            embedding = None
        total_emb_ms += (time.monotonic() - t0) * 1000

        if embedding is None:
            continue

        # 更新统计和状态
        state.embed_extracted += 1
        embed_count_this_frame += 1
        ts.embed_count += 1
        ts.cached_embedding = embedding
        state.update_embed_time(track_id, current_time, current_frame)
        state.log_embed_trigger(current_frame, track_id, EmbedReason.NEW_SAMPLE,
                               f"embed_count={ts.embed_count}")

        face_path = info.save_path

        # Step 10: 记录 embedding (独立 try-except)
        if emb_logger is not None:
            try:
                emb_logger.log(
                    track_id=track_id,
                    frame_id=ft.frame_id,
                    timestamp_ms=ft.timestamp_ms,
                    embedding=embedding,
                    quality_score=quality.score,
                    face_image_path=face_path,
                )
            except Exception as e:
                state.record_emb_error(f"emb_logger: {e}")

        # Step 11: 添加到 track template manager (计时)
        template = None
        if template_mgr is not None:
            t0 = time.monotonic()
            try:
                template = template_mgr.add_sample(
                    track_id=track_id,
                    frame_id=ft.frame_id,
                    timestamp_ms=ft.timestamp_ms,
                    embedding=embedding,
                    quality_score=quality.score,
                    image_path=face_path,
                )
            except Exception as e:
                state.record_emb_error(f"template_mgr: {e}")
            total_template_ms += (time.monotonic() - t0) * 1000

        # Step 12: Person matching (只在 template 更新时触发, 计时)
        assignment = None
        if template is not None and person_registry is not None:
            t0 = time.monotonic()
            try:
                assignment = person_registry.assign(template)
                # 更新全局映射和缓存
                state.track_to_person[track_id] = assignment.person_id
                state.track_similarities[track_id] = assignment.top1_similarity
                ts.cached_person_id = assignment.person_id
                ts.cached_similarity = assignment.top1_similarity
                # 标记 linked_person (Face Gating)
                strack.linked_person = True
            except Exception as e:
                state.record_emb_error(f"person_registry: {e}")
            total_person_ms += (time.monotonic() - t0) * 1000

        # Step 13: M5 身份判定 + 自动注册
        if (state.m5_cfg.enabled and MODULE5_AVAILABLE and
            assignment is not None and state.identity_judge is not None):
            try:
                session_person_id = assignment.person_id
                person_obj = person_registry.get_person(session_person_id)
                if person_obj is not None:
                    identity_decision = state.identity_judge.judge(
                        session_person_id=session_person_id,
                        template=person_obj.template,
                        timestamp_ms=ft.timestamp_ms,
                        frame_id=ft.frame_id,
                    )
                    state.person_identity_states[session_person_id] = identity_decision.identity_state.value

                    id_state = identity_decision.identity_state

                    if id_state == IdentityState.KNOWN_STRONG:
                        state._ambiguous_since.pop(session_person_id, None)
                        rid = identity_decision.registered_identity_id
                        if rid is not None:
                            state.person_to_registered[session_person_id] = rid
                            state.registered_db.update_template(rid, person_obj.template, weight=0.1)
                            merge_result = state.registered_db.check_and_merge(rid, merge_threshold=0.75)
                            if merge_result is not None:
                                main_rid, absorbed_rid = merge_result
                                for pid, r in list(state.person_to_registered.items()):
                                    if r == absorbed_rid:
                                        state.person_to_registered[pid] = main_rid
                                logger.info(f"[M6] Merged R#{absorbed_rid} -> R#{main_rid}")

                    elif id_state == IdentityState.UNKNOWN_STRONG:
                        state._ambiguous_since.pop(session_person_id, None)
                        if state.candidate_pool is not None:
                            sample = EmbeddingSample(
                                embedding=embedding,
                                quality_score=quality.score,
                                track_id=track_id,
                                frame_id=ft.frame_id,
                                timestamp_ms=ft.timestamp_ms,
                                image_path=face_path,
                            )
                            state.candidate_pool.try_add_or_update(
                                session_person_id=session_person_id,
                                identity_state=identity_decision.identity_state,
                                identity_decision=identity_decision,
                                sample=sample,
                                current_samples_count=person_obj.sample_count,
                                current_tracks_count=len(person_obj.track_ids),
                                avg_quality=quality.score,
                            )
                            candidate = state.candidate_pool.get_candidate(session_person_id)
                            if candidate is not None:
                                ready, _score, _reasons = candidate.check_register_ready()
                                if ready:
                                    rid = state.registered_db.register(
                                        template=person_obj.template,
                                        session_person_id=session_person_id,
                                        metadata={"quality": quality.score, "frame_id": ft.frame_id},
                                    )
                                    state.person_to_registered[session_person_id] = rid
                                    logger.info(f"[M5] US -> registered P#{session_person_id} -> R#{rid}")

                    elif id_state == IdentityState.AMBIGUOUS:
                        now = time.monotonic()
                        if session_person_id not in state._ambiguous_since:
                            state._ambiguous_since[session_person_id] = now

                        elapsed = now - state._ambiguous_since[session_person_id]

                        if elapsed >= state.amb_timeout_sec and state.candidate_pool is not None:
                            sample = EmbeddingSample(
                                embedding=embedding,
                                quality_score=quality.score,
                                track_id=track_id,
                                frame_id=ft.frame_id,
                                timestamp_ms=ft.timestamp_ms,
                                image_path=face_path,
                            )
                            state.candidate_pool.try_add_or_update(
                                session_person_id=session_person_id,
                                identity_state=identity_decision.identity_state,
                                identity_decision=identity_decision,
                                sample=sample,
                                current_samples_count=person_obj.sample_count,
                                current_tracks_count=len(person_obj.track_ids),
                                avg_quality=quality.score,
                            )
                            candidate = state.candidate_pool.get_candidate(session_person_id)
                            if candidate is not None:
                                ready, _score, _reasons = candidate.check_register_ready()
                                if ready:
                                    rid = state.registered_db.register(
                                        template=person_obj.template,
                                        session_person_id=session_person_id,
                                        metadata={"quality": quality.score, "frame_id": ft.frame_id,
                                                  "reason": "ambiguous_timeout"},
                                    )
                                    state.person_to_registered[session_person_id] = rid
                                    state._ambiguous_since.pop(session_person_id, None)
                                    logger.info(f"[M5] AMB timeout -> registered P#{session_person_id} -> R#{rid}")

            except Exception as e:
                state.record_emb_error(f"module5: {e}")

    # 更新 Module 5 统计
    if state.m5_cfg.enabled and state.identity_judge is not None:
        try:
            counts = state.identity_judge.get_state_counts()
            state.m5_known_strong_count = counts.get("known_strong", 0)
            state.m5_ambiguous_count = counts.get("ambiguous", 0)
            state.m5_unknown_strong_count = counts.get("unknown_strong", 0)
        except Exception as e:
            logger.error(f"[M5] identity count error: {e}")

        if state.registered_db is not None:
            state.m5_registered_count = state.registered_db.count()

        if state.candidate_pool is not None:
            try:
                pool_counts = state.candidate_pool.get_counts()
                state.m5_candidate_count = pool_counts.get("candidate_count", 0)
                state.m5_ready_candidate_count = pool_counts.get("ready_candidate_count", 0)
            except Exception as e:
                logger.error(f"[M5] candidate count error: {e}")

    alpha = 0.3
    state.align_ms = state.align_ms * (1 - alpha) + total_align_ms * alpha
    state.sample_ms = state.sample_ms * (1 - alpha) + total_sample_ms * alpha
    state.emb_ms = state.emb_ms * (1 - alpha) + total_emb_ms * alpha
    state.template_ms = state.template_ms * (1 - alpha) + total_template_ms * alpha
    state.person_ms = state.person_ms * (1 - alpha) + total_person_ms * alpha

    try:
        state.update_align_stats()
    except Exception as e:
        logger.error(f"[Stats] align stats error: {e}")

    try:
        state.update_embed_stats()
    except Exception as e:
        logger.error(f"[Stats] embed stats error: {e}")

    state._identity_snapshot = {
        "track_to_person": dict(state.track_to_person),
        "track_similarities": dict(state.track_similarities),
        "person_identity_states": dict(state.person_identity_states),
        "person_to_registered": dict(state.person_to_registered),
    }


# ======================================================================
# MJPEG 帧生成器
# ======================================================================

def _get_person_name(pid):
    """从内存缓存或 RegisteredPersonDB metadata 获取名字。"""
    name = state._person_names.get(pid, "")
    if name:
        return name
    rid = state.person_to_registered.get(pid)
    if rid is not None and state.registered_db is not None:
        meta = state.registered_db.get_metadata(rid)
        dn = meta.get("display_name", "")
        if dn:
            state._person_names[pid] = dn
            return dn
    return ""


def _generate_frames():
    while state.running:
        frame_start = time.monotonic()

        src = state.source
        if src is None:
            time.sleep(0.01)
            continue

        try:
            if state.mode == "camera":
                frame = src.read(timeout=2.0)
            else:
                frame = src.read()
        except Exception as e:
            logger.error(f"[Error] source.read: {e}")
            time.sleep(0.01)
            continue

        if frame is None:
            if state.mode == "video":
                state.running = False
                break
            continue

        state._last_frame = frame.image

        det = state.detector
        tracker = state.tracker

        # 主渲染逻辑
        try:
            if det is None or not det.is_loaded or tracker is None:
                display = frame.image.copy()
            else:
                do_det = (frame.frame_id % state.det_every_n == 0)
                ft = tracker.step(frame, det, do_detect=do_det)
                dropped = src.dropped_frames
                state.update_stats_from_ft(ft, dropped)
                state.append_log(ft.to_dict())

                # Alignment + Embedding (异步后台线程)
                # 只在有 CONFIRMED track 且对齐已启用时才提交
                if do_det and state.identity_worker is not None and state.align_enabled:
                    confirmed = [t for t in tracker.last_active_stracks
                                 if t.state == TrackState.CONFIRMED
                                 and t.det_score is not None and t.kps5 is not None]
                    if confirmed:
                        state.identity_worker.submit(
                            frame.image, ft, confirmed
                        )

                # 说话状态检测 (异步 MouthWorker)
                if state.mouth_worker is not None:
                    for strack in tracker.last_active_stracks:
                        if strack.state == TrackState.CONFIRMED and strack.det_score is not None:
                            bbox = strack.bbox_xyxy_clipped(ft.width, ft.height)
                            x1, y1, x2, y2 = [int(v) for v in bbox]
                            crop = frame.image[max(0, y1):y2, max(0, x1):x2]
                            if crop.size > 0:
                                state.mouth_worker.submit(strack.track_id, crop.copy(), ft.timestamp_ms)
                    state.mouth_states = state.mouth_worker.get_results()

                det_faces = state.num_faces

                # 读取 identity worker 的快照（原子引用读取，无锁）
                _snap = state._identity_snapshot
                _snap_tp = _snap["track_to_person"]
                _snap_ts = _snap["track_similarities"]
                _snap_is = _snap["person_identity_states"]
                _snap_pr = _snap["person_to_registered"]

                # 缓存 persons 数据供 /api/persons 使用（避免跨线程访问 tracker）
                try:
                    import base64 as _b64p

                    import cv2 as _cv2p
                    _persons_cache = []
                    _fh, _fw = frame.image.shape[:2]
                    for _st in tracker.last_active_stracks:
                        if _st.state != TrackState.CONFIRMED:
                            continue
                        _tid = _st.track_id
                        _pid = _snap_tp.get(_tid)
                        _mouth = state.mouth_states.get(_tid, "")
                        _ident = _snap_is.get(_pid, "") if _pid else ""
                        _rid = _snap_pr.get(_pid) if _pid else None
                        _bbox = _st.bbox_xyxy_clipped(_fw, _fh)
                        _bnorm = [round(_bbox[0]/_fw,4), round(_bbox[1]/_fh,4),
                                  round(_bbox[2]/_fw,4), round(_bbox[3]/_fh,4)]
                        if _pid is not None and _pid not in state._person_avatars:
                            try:
                                _x1,_y1,_x2,_y2 = [int(v) for v in _bbox]
                                _crop = frame.image[max(0,_y1):_y2, max(0,_x1):_x2]
                                if _crop.size > 0:
                                    _sm = _cv2p.resize(_crop, (80,80))
                                    _, _buf = _cv2p.imencode('.jpg', _sm, [_cv2p.IMWRITE_JPEG_QUALITY,75])
                                    state._person_avatars[_pid] = _b64p.b64encode(_buf).decode('ascii')
                            except Exception:
                                pass
                        _persons_cache.append({
                            "track_id": _tid, "person_id": _pid,
                            "identity": _ident, "registered_id": _rid,
                            "mouth": _mouth, "bbox": _bnorm,
                            "name": _get_person_name(_pid) if _pid else "",
                            "avatar": state._person_avatars.get(_pid) if _pid else None,
                        })
                    state._persons_cache = _persons_cache
                except Exception:
                    pass

                # 绘制显示
                display = draw_tracks(
                    frame.image,
                    ft,
                    state.fps,
                    dropped,
                    det_faces,
                    track_to_person=_snap_tp if state.embed_enabled else None,
                    show_person=state.embed_enabled,
                    show_similarity=state.embed_enabled,
                    track_similarities=_snap_ts if state.embed_enabled else None,
                    person_count=state.num_persons,
                    # Module 5 参数
                    person_identity_states=_snap_is if state.m5_cfg.enabled else None,
                    person_to_candidate=state.person_to_candidate if state.m5_cfg.enabled else None,
                    person_to_registered=_snap_pr if state.m5_cfg.enabled else None,
                    mouth_states=state.mouth_states,
                    show_hud=getattr(state, 'show_hud', True),
                )

        except Exception as e:
            logger.error(f"[Error] frame processing: {e}")
            display = frame.image.copy()

        # 编码 (计时)
        encode_start = time.monotonic()
        try:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
            _, buf = cv2.imencode(".jpg", display, encode_params)
            state.encode_ms = state.encode_ms * 0.7 + (time.monotonic() - encode_start) * 1000 * 0.3
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )
        except Exception as e:
            logger.error(f"[Error] encoding: {e}")
            continue

        # 总帧时间
        state.total_frame_ms = state.total_frame_ms * 0.7 + (time.monotonic() - frame_start) * 1000 * 0.3

    # 流结束
    try:
        black = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(black, "Stream ended", (180, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
        _, buf = cv2.imencode(".jpg", black)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buf.tobytes()
            + b"\r\n"
        )
    except Exception as e:
        logger.error(f"[Stream] end frame error: {e}")


# ======================================================================
# 路由
# ======================================================================

@app.route("/")
def index():
    ts = str(int(time.time()))
    sig = hmac.new(API_KEY.encode(), ts.encode(), "sha256").hexdigest()[:16]
    resp = app.make_response(render_template(
        "index_v2.html", api_key=API_KEY, stream_ts=ts, stream_sig=sig))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/legacy")
def index_legacy():
    ts = str(int(time.time()))
    sig = hmac.new(API_KEY.encode(), ts.encode(), "sha256").hexdigest()[:16]
    resp = app.make_response(render_template(
        "index.html", api_key=API_KEY, stream_ts=ts, stream_sig=sig))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/video_feed")
def video_feed():
    if not _verify_stream_signature():
        return "Forbidden", 403
    from . import config as _cfg
    with _cfg._stream_lock:
        if _cfg._active_streams >= _cfg._max_streams:
            return "Too many streams", 429
        _cfg._active_streams += 1

    def gen():
        try:
            yield from _generate_frames()
        finally:
            from . import config as _cfg2
            with _cfg2._stream_lock:
                _cfg2._active_streams -= 1

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/start", methods=["POST"])
@require_api_key
def api_start():
    data = request.get_json(force=True)
    logger.debug("[/api/start] 请求参数: %s", json.dumps(data, ensure_ascii=False))

    mode = data.get("mode", "camera")
    if mode not in ("camera", "video"):
        return jsonify({"ok": False, "error": "mode must be 'camera' or 'video'"}), 400

    det_thresh = max(0.01, min(1.0, float(data.get("det_thresh", 0.5))))
    det_size = max(320, min(1280, int(data.get("det_size", 640))))

    try:
        model_path = _validate_model_path(data.get("model", "models/det_10g.onnx"))
    except (ValueError, FileNotFoundError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    model = str(model_path)

    # 四阈值分离 (带边界约束)
    detector_emit_min_det = max(0.01, min(1.0, float(data.get("detector_emit_min_det", 0.15))))
    track_new_high_thres = max(0.01, min(1.0, float(data.get("track_new_high_thres", 0.50))))
    track_update_low_thres = max(0.01, min(1.0, float(data.get("track_update_low_thres", 0.15))))
    sample_min_det = max(0.01, min(1.0, float(data.get("sample_min_det", 0.60))))

    det_every_n = max(1, min(30, int(data.get("det_every_n", 1))))
    max_age = max(1, min(300, int(data.get("max_age", 30))))
    min_hits = max(1, min(30, int(data.get("min_hits", 3))))
    match_iou = max(0.01, min(1.0, float(data.get("match_iou_threshold", 0.3))))

    align_enabled = bool(data.get("align_enabled", True))
    min_quality_det = max(0.0, min(1.0, float(data.get("min_quality_det", 0.60))))
    min_quality_area = max(0, min(100000, float(data.get("min_quality_area", 900))))
    min_quality_blur = max(0.0, min(200.0, float(data.get("min_quality_blur", 40.0))))
    max_samples = max(1, min(100, int(data.get("max_samples", 10))))

    embed_enabled = bool(data.get("embed_enabled", True))
    similarity_threshold = max(0.0, min(1.0, float(data.get("similarity_threshold", 0.4))))
    margin_threshold = max(0.0, min(1.0, float(data.get("margin_threshold", 0.1))))
    min_template_samples = max(1, min(50, int(data.get("min_template_samples", 3))))
    try:
        arcface_model_path = _validate_model_path(data.get("arcface_model", "models/w600k_r50.onnx"))
    except (ValueError, FileNotFoundError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    arcface_model = str(arcface_model_path)

    embed_cooldown_ms = max(0.0, min(10000.0, float(data.get("embed_cooldown_ms", 200.0))))
    max_embed_per_frame = max(1, min(20, int(data.get("max_embed_per_frame", 5))))
    max_embed_per_track = max(1, min(50, int(data.get("max_embed_per_track", 5))))

    credit_enabled = bool(data.get("credit_enabled", True))
    credit_increment = max(0.0, min(10.0, float(data.get("credit_increment", 1.0))))
    credit_decrement = max(0.0, min(10.0, float(data.get("credit_decrement", 0.5))))
    credit_threshold = max(0.0, min(100.0, float(data.get("credit_threshold", 3.0))))
    credit_max = max(0.0, min(100.0, float(data.get("credit_max", 10.0))))

    speaking_enabled = bool(data.get("speaking_enabled", True))
    show_hud = bool(data.get("show_hud", True))

    m5_enabled = bool(data.get("m5_enabled", True))
    m5_known_threshold = max(0.0, min(1.0, float(data.get("m5_known_threshold", 0.55))))
    m5_band_threshold = max(0.0, min(1.0, float(data.get("m5_band_threshold", 0.35))))
    m5_margin_threshold = max(0.0, min(1.0, float(data.get("m5_margin_threshold", 0.10))))

    _stop_pipeline()

    # 检测器 (emit_thresh 放行弱框给 tracker)
    global _detector_loaded
    if (state.detector is not None and _detector_loaded
            and state.detector._det_thresh == det_thresh
            and state.detector._emit_thresh == detector_emit_min_det
            and state.detector._det_size == (det_size, det_size)):
        pass
    else:
        det = SCRFDDetector(
            model_path=model,
            det_size=(det_size, det_size),
            det_thresh=det_thresh,
            emit_thresh=detector_emit_min_det,
        )
        det.load()
        state.detector = det
        _detector_loaded = True

    # Tracker (四阈值分离)
    cfg = BoTSORTConfig(
        detector_emit_min_det=detector_emit_min_det,
        track_new_high_thres=track_new_high_thres,
        track_update_low_thres=track_update_low_thres,
        match_iou_threshold=match_iou,
        min_hits=min_hits,
        max_age=max_age,
    )
    state.tracker = BoTSORTTracker(cfg)
    state.sample_min_det = sample_min_det
    state.det_every_n = max(1, det_every_n)
    state.align_enabled = align_enabled

    # Cooldown 配置
    state.embed_cooldown_ms = embed_cooldown_ms
    state.max_embed_per_frame = max_embed_per_frame
    state.max_embed_per_track = max_embed_per_track

    # Credit Gate 配置
    state.credit_gate_cfg = CreditGateConfig(
        enabled=credit_enabled,
        credit_increment=credit_increment,
        credit_decrement=credit_decrement,
        credit_threshold=credit_threshold,
        credit_max=credit_max,
    )

    # Module 5 配置 (三态身份判定 + 候选池)
    state.m5_cfg = Module5Config(
        enabled=m5_enabled,
        known_threshold=m5_known_threshold,
        band_threshold=m5_band_threshold,
        margin_threshold=m5_margin_threshold,
    )

    # Alignment 组件
    state.aligner = FaceAligner(output_size=(112, 112))
    state.qgate = QualityGate(QualityConfig(
        min_det_score=min_quality_det,
        min_bbox_area=min_quality_area,
        min_blur_score=min_quality_blur,
    ))
    if state.sampler:
        try:
            state.sampler.close()
        except Exception:
            pass
    state.sampler = TrackSampler(
        max_samples=max_samples,
        output_dir="output/faces",
        log_path="output/track_faces.jsonl",
    ).open()

    # Embedding
    warnings = []
    if embed_enabled and not align_enabled:
        warnings.append("embed_enabled=True 但 align_enabled=False，人员匹配需要对齐数据")

    if embed_enabled and not EMBEDDING_AVAILABLE:
        warnings.append("Embedding 模块不可用")
        embed_enabled = False

    state.embed_enabled = embed_enabled

    if embed_enabled:
        if not os.path.exists(arcface_model):
            return jsonify({
                "ok": False,
                "error": f"ArcFace model not found: {arcface_model}"
            }), 400

        try:
            state.embedder = ArcFaceEmbedder(model_path=arcface_model)
        except Exception:
            logger.exception("ArcFace model load failed")
            return jsonify({"ok": False, "error": "ArcFace 模型加载失败"}), 500

        try:
            state.template_mgr = TrackTemplateManager(
                min_samples=min_template_samples,
                max_samples=max_samples,
                aggregation="quality_weighted",
                log_path="output/track_templates.jsonl",
            ).open()
        except Exception:
            logger.exception("Template manager init failed")
            return jsonify({"ok": False, "error": "模板管理器初始化失败"}), 500

        try:
            state.person_registry = PersonRegistry(
                similarity_threshold=similarity_threshold,
                margin_threshold=margin_threshold,
                log_path="output/person_assignments.jsonl",
            ).open()
        except Exception:
            logger.exception("Person registry init failed")
            return jsonify({"ok": False, "error": "人员注册表初始化失败"}), 500

        try:
            state.emb_logger = EmbeddingLogger(
                output_dir="output/embeddings",
                log_path="output/embeddings.jsonl",
            ).open()
        except Exception:
            logger.exception("Embedding logger init failed")
            return jsonify({"ok": False, "error": "Embedding 日志初始化失败"}), 500
    else:
        state.embedder = None
        state.template_mgr = None
        state.person_registry = None
        state.emb_logger = None

    # Module 5 初始化 (三态身份判定 + 候选池)
    if m5_enabled and MODULE5_AVAILABLE and embed_enabled:
        try:
            state.registered_db = RegisteredPersonDB(db_dir="output/registered_db")
            state.registered_db.load()
            logger.info(f"[M6] RegisteredDB initialized, loaded {state.registered_db.count()} identities")

            # 三态判定器
            state.identity_judge = IdentityJudge(
                registered_db=state.registered_db,
                config=IdentityConfig(
                    known_threshold=m5_known_threshold,
                    band_threshold=m5_band_threshold,
                    margin_threshold=m5_margin_threshold,
                ),
                log_path="output/person_states.jsonl",
            ).open()

            # 候选池
            state.candidate_pool = FaceCandidatePool(
                session_id=None,  # 自动生成
                source_id=data.get("path", f"camera:{data.get('device', 0)}"),
                config=CandidateConfig(
                    min_samples_to_enter=3,
                    min_tracks_to_enter=1,
                ),
                candidates_log_path="output/candidates.jsonl",
                summaries_log_path="output/candidate_summaries.jsonl",
            ).open()

            logger.info("[Module 5] 三态判定 + 候选池已启用")
        except Exception as e:
            logger.info(f"[Module 5] 初始化失败: {e}")
            state.identity_judge = None
            state.candidate_pool = None
    else:
        state.registered_db = None
        state.identity_judge = None
        state.candidate_pool = None

    state.reset()
    state.mode = mode

    try:
        if mode == "camera":
            device = int(data.get("device", 0))
            src = CameraSource(device=device)
            src.open()
            state.source = src
            state.source_id = src.source_id
        else:
            path = data.get("path", "")
            realtime = bool(data.get("realtime", False))
            if not path:
                return jsonify({"ok": False, "error": "path is required"}), 400
            video_path = Path(path).resolve()
            if not any(video_path.is_relative_to(d) for d in ALLOWED_VIDEO_DIRS):
                return jsonify({"ok": False, "error": "视频路径不在允许的目录中"}), 403
            if not video_path.exists():
                return jsonify({"ok": False, "error": "文件不存在"}), 404
            src = VideoSource(path=str(video_path), realtime=realtime)
            src.open()
            state.source = src
            state.source_id = src.source_id

        state.running = True

        worker = IdentityWorker()
        worker.start()
        state.identity_worker = worker

        state.speaking_enabled = speaking_enabled
        state.show_hud = show_hud
        if speaking_enabled and SPEAKING_AVAILABLE:
            try:
                analyzer = SpeakingAnalyzer()
                mw = MouthWorker(analyzer)
                mw.start()
                state.mouth_worker = mw
                logger.info("[Speaking] SpeakingAnalyzer + MouthWorker started")
            except Exception as e:
                logger.error(f"[Speaking] init failed: {e}")
                state.mouth_worker = None
        else:
            if speaking_enabled and not SPEAKING_AVAILABLE:
                warnings.append("Speaking 模块不可用 (mediapipe 未安装)")
            state.mouth_worker = None

        result = {
            "ok": True,
            "mode": mode,
            "source_id": state.source_id,
            "align_enabled": state.align_enabled,
            "embed_enabled": state.embed_enabled,
            "embed_cooldown_ms": state.embed_cooldown_ms,
        }
        if warnings:
            result["warnings"] = warnings

        return jsonify(result)
    except Exception:
        logger.exception("api_start failed")
        return jsonify({"ok": False, "error": "启动失败，请检查服务端日志"}), 500


@app.route("/api/stop", methods=["POST"])
@require_api_key
def api_stop():
    _stop_pipeline()
    return jsonify({"ok": True})


def _stop_pipeline():
    state.running = False

    if state.identity_worker is not None:
        state.identity_worker.stop()
        state.identity_worker = None

    if state.mouth_worker is not None:
        state.mouth_worker.stop()
        state.mouth_worker = None

    time.sleep(0.1)

    src = state.source
    if src is not None:
        try:
            src.close()
        except Exception as e:
            logger.error(f"[Stop] source close error: {e}")
        state.source = None

    if state.sampler:
        try:
            state.sampler.close()
        except Exception as e:
            logger.error(f"[Stop] sampler close error: {e}")
        state.sampler = None

    if state.template_mgr:
        try:
            state.template_mgr.close()
        except Exception as e:
            logger.error(f"[Stop] template_mgr close error: {e}")
        state.template_mgr = None

    if state.person_registry:
        try:
            state.person_registry.close()
        except Exception as e:
            logger.error(f"[Stop] person_registry close error: {e}")
        state.person_registry = None

    if state.emb_logger:
        try:
            state.emb_logger.close()
        except Exception as e:
            logger.error(f"[Stop] emb_logger close error: {e}")
        state.emb_logger = None

    if state.candidate_pool:
        try:
            summaries = state.candidate_pool.flush_summaries()
            logger.info(f"[Module 5] 生成 {len(summaries)} 个候选摘要")
        except Exception as e:
            logger.info(f"[Module 5] flush_summaries 失败: {e}")
        try:
            state.candidate_pool.close()
        except Exception as e:
            logger.error(f"[Stop] candidate_pool close error: {e}")
        state.candidate_pool = None

    if state.identity_judge:
        try:
            state.identity_judge.close()
        except Exception as e:
            logger.error(f"[Stop] identity_judge close error: {e}")
        state.identity_judge = None

    if state.registered_db is not None:
        try:
            n = state.registered_db.count()
            logger.info(f"[M6] Saving RegisteredDB ({n} identities)...")
            state.registered_db.save()
        except Exception as e:
            import traceback
            logger.error(f"[M6] RegisteredDB save FAILED: {e}")
            traceback.print_exc()
    else:
        logger.info("[M6] registered_db is None, nothing to save")
    state.registered_db = None

    state.embedder = None
    state.embed_enabled = False


@app.route("/api/flush_summaries", methods=["POST"])
@require_api_key
def api_flush_summaries():
    """手动触发生成候选摘要"""
    if state.candidate_pool is None:
        return jsonify({"ok": False, "error": "candidate_pool not enabled"})

    try:
        summaries = state.candidate_pool.flush_summaries()
        return jsonify({
            "ok": True,
            "count": len(summaries),
            "summaries": [s.to_dict() for s in summaries],
        })
    except Exception:
        logger.exception("flush_summaries failed")
        return jsonify({"ok": False, "error": "操作失败"})


def _count_credit_above_threshold() -> int:
    """统计当前信用分达标的 track 数量。"""
    if state.tracker is None:
        return 0
    thresh = state.credit_gate_cfg.credit_threshold
    return sum(1 for t in state.tracker.last_active_stracks
               if t.state == TrackState.CONFIRMED and t.face_valid_credit >= thresh)


@app.route("/api/persons")
def api_persons():
    """返回主循环缓存的 persons 数据（不访问 tracker，零开销）。"""
    return jsonify(getattr(state, '_persons_cache', []))


@app.route("/api/person/rename", methods=["POST"])
@require_api_key
def api_person_rename():
    """重命名一个 person，同时持久化到 RegisteredPersonDB。"""
    data = request.get_json(force=True)
    pid = data.get("person_id")
    name = data.get("name", "").strip()[:50]
    if pid is None:
        return jsonify({"ok": False, "error": "missing person_id"}), 400

    try:
        pid = int(pid)
    except (ValueError, TypeError):
        return jsonify({"ok": False, "error": "person_id must be integer"}), 400
    state._person_names[pid] = name

    rid = state.person_to_registered.get(pid)
    if rid is not None and state.registered_db is not None:
        meta = state.registered_db.get_metadata(rid)
        meta["display_name"] = name
        state.registered_db._metadata[rid] = meta
        try:
            state.registered_db.save()
        except Exception as e:
            logger.error(f"[Rename] save error: {e}")

    return jsonify({"ok": True, "person_id": pid, "name": name})


@app.route("/api/stats")
def api_stats():
    return jsonify({
        "running": state.running,
        "mode": state.mode,
        "source_id": state.source_id,
        "frame_id": state.frame_id,
        "timestamp_ms": round(state.timestamp_ms, 1),
        "fps": round(state.fps, 1),

        # 细粒度计时 (Module 4b)
        "det_ms": round(state.detect_ms, 1),
        "trk_ms": round(state.track_ms, 1),
        "align_ms": round(state.align_ms, 1),
        "sample_ms": round(state.sample_ms, 1),
        "emb_ms": round(state.emb_ms, 1),
        "template_ms": round(state.template_ms, 1),
        "person_ms": round(state.person_ms, 1),
        "encode_ms": round(state.encode_ms, 1),
        "total_ms": round(state.total_frame_ms, 1),

        "num_faces": state.num_faces,
        "num_tracks": state.num_tracks,
        "num_persons": state.num_persons,
        "dropped_frames": state.dropped_frames,
        "width": state.width,
        "height": state.height,
        "did_detect": state.did_detect,

        "align_enabled": state.align_enabled,
        "align_evaluated": state.align_evaluated,
        "align_passed": state.align_passed,
        "align_saved": state.align_saved,

        "embed_enabled": state.embed_enabled,
        "embed_extracted": state.embed_extracted,
        "templates_created": state.templates_created,
        "emb_error_count": state.emb_error_count,
        "has_error": bool(state.last_error),

        # 缓存统计 (Module 4b)
        "emb_cache_hit": state.emb_cache_hit,
        "emb_cache_miss": state.emb_cache_miss,
        "emb_cooldown_skip": state.emb_cooldown_skip,

        # Credit Gate 统计
        "credit_gate_enabled": state.credit_gate_cfg.enabled,
        "credit_above_threshold": _count_credit_above_threshold(),

        # Module 5 统计
        "m5_enabled": state.m5_cfg.enabled,
        "m5_known_strong_count": state.m5_known_strong_count,
        "m5_ambiguous_count": state.m5_ambiguous_count,
        "m5_unknown_strong_count": state.m5_unknown_strong_count,
        "m5_registered_count": state.m5_registered_count,
        "m5_candidate_count": state.m5_candidate_count,
        "m5_ready_candidate_count": state.m5_ready_candidate_count,

        # 弱检测路径统计
        **(state.tracker.get_det_stats() if state.tracker else {}),
    })


@app.route("/api/embed_log")
def api_embed_log():
    """获取 embedding 触发日志"""
    try:
        n = max(1, min(500, int(request.args.get("n", 50))))
    except (ValueError, TypeError):
        n = 50
    entries = state.embed_trigger_log[-n:]
    return jsonify(entries)


@app.route("/api/gating_log")
def api_gating_log():
    """(保留兼容) Face Gating 已移除, 返回空列表。"""
    return jsonify([])


@app.route("/api/track_status/<int:track_id>")
def api_track_status(track_id: int):
    """查询指定 track 的详细状态 (用于调试)。

    返回该 track 在系统中停留的层级:
    - tracked: 仅 track 层
    - sampled: 已进入采样
    - embedded: 已进入 embedding
    - assigned: 已分配 person
    """
    # 查找 track
    tracker = state.tracker
    if tracker is None:
        return jsonify({"ok": False, "error": "Tracker not running"})

    strack = None
    for t in tracker.last_active_stracks:
        if t.track_id == track_id:
            strack = t
            break

    if strack is None:
        # 检查是否在 lost 或已移除
        for t in tracker.tracked_stracks + tracker.lost_stracks:
            if t.track_id == track_id:
                strack = t
                break

    if strack is None:
        return jsonify({
            "ok": True,
            "track_id": track_id,
            "status": "not_found",
            "message": "Track 不存在或已被移除",
        })

    # 获取 embedding 状态
    ts = state._track_embed_state.get(track_id)
    person_id = state.track_to_person.get(track_id)
    similarity = state.track_similarities.get(track_id)

    # 判断层级
    level = "tracked"
    if strack.ever_sampled:
        level = "sampled"
    if ts is not None and ts.embed_count > 0:
        level = "embedded"
    if person_id is not None:
        level = "assigned"

    return jsonify({
        "ok": True,
        "track_id": track_id,
        "status": "found",
        "level": level,
        "track_state": strack.state,
        "face_valid_credit": round(strack.face_valid_credit, 2),
        "ever_sampled": strack.ever_sampled,
        "linked_person": strack.linked_person,
        "age": strack.age,
        "embed_count": ts.embed_count if ts else 0,
        "person_id": person_id,
        "similarity": round(similarity, 4) if similarity else None,
    })


@app.route("/api/log")
def api_log():
    try:
        n = max(1, min(500, int(request.args.get("n", 50))))
    except (ValueError, TypeError):
        n = 50
    entries = state.log_entries[-n:]
    return jsonify(entries)


@app.route("/api/upload_video", methods=["POST"])
@require_api_key
def api_upload_video():
    f = request.files.get("file")
    if f is None or f.filename == "":
        return jsonify({"ok": False, "error": "No file"}), 400

    raw_ext = Path(f.filename).suffix.lower() if f.filename else ""
    safe_name = secure_filename(f.filename)
    if not safe_name or not Path(safe_name).suffix:
        if raw_ext in ALLOWED_VIDEO_EXT:
            safe_name = f"upload{raw_ext}"
        else:
            return jsonify({"ok": False, "error": "Invalid filename"}), 400

    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXT:
        return jsonify({"ok": False, "error": f"Unsupported format: {ext}"}), 400

    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    UPLOAD_DIR.mkdir(exist_ok=True)
    save_path = (UPLOAD_DIR / unique_name).resolve()

    if not save_path.is_relative_to(UPLOAD_DIR.resolve()):
        return jsonify({"ok": False, "error": "Invalid path"}), 400

    f.save(str(save_path))
    rel_path = str(UPLOAD_DIR / unique_name)
    return jsonify({"ok": True, "path": rel_path})


@app.route("/api/videos")
def api_videos():
    upload_dir = Path("uploads")
    if not upload_dir.exists():
        return jsonify([])
    exts = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}
    files = [f.name for f in upload_dir.iterdir() if f.suffix.lower() in exts]
    return jsonify(sorted(files))
