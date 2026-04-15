"""
SpeakingAnalyzer — XGBoost 说话检测 + BiSeNet 遮挡检测。

替代旧的手调规则 MouthAnalyzer, 完全兼容 MouthWorker 接口。

架构:
  L1: BiSeNet face parsing (每 track 每 N 次调用跑一次)
      + |yaw| > 60 自遮挡
  L2: XGBoost (blendshape 滑窗特征, 无 VSDLM)
  L3: Hysteresis 状态机 (yaw-adaptive)
"""

from __future__ import annotations

import json
import os
from collections import deque
from typing import Dict, Tuple

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import xgboost as xgb
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .mouth_analyzer import MouthState

ALL_MOUTH_INDICES_MP = sorted(set([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    415, 310, 311, 312, 13, 82, 81, 80, 191,
]))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
LIP_CLASSES = {11, 12, 13}


class _TrackState:
    __slots__ = ("ring", "bis_cache_ratio", "bis_counter",
                 "hyst_status", "hyst_on", "hyst_off",
                 "unobservable_streak",
                 "occ_streak", "vis_streak", "occ_status",
                 "jaw_history")

    def __init__(self, window: int):
        self.ring: deque = deque(maxlen=window)
        self.bis_cache_ratio: float = -1.0
        self.bis_counter: int = 0
        self.hyst_status: str = "not_speaking"
        self.hyst_on: int = 0
        self.hyst_off: int = 0
        self.unobservable_streak: int = 0
        self.occ_streak: int = 0
        self.vis_streak: int = 0
        self.occ_status: str = "visible"
        self.jaw_history: deque = deque(maxlen=5)


class SpeakingAnalyzer:
    """XGBoost 说话检测 + BiSeNet 遮挡, 兼容 MouthWorker 接口。"""

    def __init__(
        self,
        model_path: str = "models/speaking/speaking_model.pkl",
        bisenet_path: str = "models/speaking/resnet18.onnx",
        landmarker_path: str = "models/face_landmarker.task",
        bisenet_every_n: int = 5,
        hyst_on_thresh: float = 0.70,
        hyst_off_thresh: float = 0.30,
        hyst_on_frames: int = 3,
        hyst_off_frames: int = 2,
    ):
        # XGBoost (JSON format, no pickle)
        meta_path = os.path.splitext(model_path)[0] + "_meta.json" \
            if model_path.endswith(".json") \
            else model_path.replace(".pkl", "_meta.json").replace(".json", "_meta.json")
        if model_path.endswith(".pkl"):
            model_path = model_path.replace(".pkl", ".json")
        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            model_dir = os.path.dirname(model_path) or "models/speaking"
            model_path = os.path.join(model_dir, "speaking_model.json")
            meta_path = os.path.join(model_dir, "speaking_meta.json")

        self._xgb_booster = xgb.Booster()
        self._xgb_booster.load_model(model_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self._raw_features = meta["raw_features"]
        self._feature_cols = meta["feature_cols"]
        self._window = meta["window_size"]
        self._n_raw = len(self._raw_features)

        # MediaPipe with blendshapes
        base_opts = mp_python.BaseOptions(model_asset_path=landmarker_path)
        lm_opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(lm_opts)

        # BiSeNet
        bis_opts = ort.SessionOptions()
        bis_opts.intra_op_num_threads = 2
        self._bisenet = ort.InferenceSession(bisenet_path, sess_options=bis_opts)
        self._bis_input = self._bisenet.get_inputs()[0].name
        self._bis_every_n = bisenet_every_n

        # Hysteresis params
        self._on_thresh = hyst_on_thresh
        self._off_thresh = hyst_off_thresh
        self._base_on_frames = hyst_on_frames
        self._base_off_frames = hyst_off_frames

        # Per-track state
        self._tracks: Dict[int, _TrackState] = {}

    # ------------------------------------------------------------------
    # Public interface (MouthWorker compatible)
    # ------------------------------------------------------------------

    def analyze(
        self, track_id: int, face_crop_bgr: np.ndarray,
        timestamp_ms: float = 0.0,
    ) -> MouthState:
        ts = self._get_track(track_id)
        h, w = face_crop_bgr.shape[:2]

        # MediaPipe
        rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_img)

        if not result.face_landmarks:
            return self._emit("unobservable", 0.0, "no_face", timestamp_ms)

        lm = result.face_landmarks[0]
        pts = np.array([(pt.x * w, pt.y * h) for pt in lm], dtype=np.float32)

        # Yaw / pitch / roll
        yaw = pitch = roll = 0.0
        if result.facial_transformation_matrixes:
            yaw, pitch, roll = self._extract_ypr(
                result.facial_transformation_matrixes[0])

        # L1: Self-occlusion
        yaw_abs = abs(yaw)
        if yaw_abs > 60:
            return self._emit("self_occluded", 0.0,
                              f"yaw={yaw:.0f}", timestamp_ms)

        # L1: BiSeNet occlusion (every N calls per track)
        ts.bis_counter += 1
        bis_refreshed = False
        if ts.bis_counter >= self._bis_every_n:
            ts.bis_counter = 0
            ts.bis_cache_ratio = self._run_bisenet(face_crop_bgr, pts, w, h)
            bis_refreshed = True

        lip_ratio = ts.bis_cache_ratio

        # BiSeNet failed or unanalyzable
        if lip_ratio < 0:
            ts.unobservable_streak += 1
            if ts.unobservable_streak >= 10:
                ts.hyst_status = "unobservable"
                ts.hyst_on = 0
                ts.hyst_off = 0
                return self._emit("unobservable", 0.0,
                                  "bisenet_fail", timestamp_ms)
            return self._emit(ts.hyst_status, 0.0,
                              f"bisenet_fail({ts.unobservable_streak})",
                              timestamp_ms)

        ts.unobservable_streak = 0

        # Occlusion hysteresis: only update on BiSeNet refresh
        if bis_refreshed:
            if lip_ratio < 0.20:
                ts.occ_streak += 1
                ts.vis_streak = 0
            elif lip_ratio >= 0.30:
                ts.vis_streak += 1
                ts.occ_streak = 0

            if ts.occ_streak >= 2:
                ts.occ_status = "occluded"
            elif ts.vis_streak >= 2:
                ts.occ_status = "visible"

        if ts.occ_status == "occluded":
            return self._emit("occluded", 0.0,
                              f"lip={lip_ratio:.2f}", timestamp_ms)

        # Blendshapes
        bs_dict = {}
        if result.face_blendshapes:
            for bs in result.face_blendshapes[0]:
                bs_dict[bs.category_name] = float(bs.score)

        # L2: XGBoost speaking prob
        row = self._build_raw_row(bs_dict, yaw, pitch, roll)
        ts.ring.append(row)

        if len(ts.ring) < 3:
            return self._emit("not_speaking", 0.0,
                              f"warmup={len(ts.ring)}", timestamp_ms)

        feat = self._compute_features(ts)
        prob = float(self._xgb_booster.predict(xgb.DMatrix(feat))[0])

        # Partial occlusion suppression
        if lip_ratio < 0.25:
            prob *= 0.4

        # L3: Hysteresis (yaw-adaptive)
        status = self._hysteresis(ts, prob, yaw_abs)
        return self._emit(status, prob,
                          f"p={prob:.2f},lip={lip_ratio:.2f},yaw={yaw:.0f}",
                          timestamp_ms)

    def remove_track(self, track_id: int):
        self._tracks.pop(track_id, None)

    def reset(self):
        self._tracks.clear()

    def close(self):
        """释放 MediaPipe 资源。"""
        if hasattr(self, "_landmarker") and self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __del__(self):
        """析构时确保资源释放。"""
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # BiSeNet
    # ------------------------------------------------------------------

    def _run_bisenet(self, face_crop_bgr: np.ndarray, pts: np.ndarray,
                     w: int, h: int) -> float:
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        bw, bh = x_max - x_min, y_max - y_min
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        half = max(bw, bh) / 2 * 1.3
        fx1 = max(0, int(cx - half))
        fy1 = max(0, int(cy - half))
        fx2 = min(w, int(cx + half))
        fy2 = min(h, int(cy + half))

        crop = face_crop_bgr[fy1:fy2, fx1:fx2]
        if crop.size == 0:
            return -1.0

        img = cv2.resize(crop, (512, 512))
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0
        blob = (blob - IMAGENET_MEAN) / IMAGENET_STD
        blob = np.expand_dims(blob, axis=0)

        out = self._bisenet.run(None, {self._bis_input: blob})
        seg = np.argmax(out[0][0], axis=0) if out[0].ndim == 4 else out[0][0]

        mouth_pts = pts[ALL_MOUTH_INDICES_MP]
        mic_x = mouth_pts[:, 0] - fx1
        mic_y = mouth_pts[:, 1] - fy1
        crop_h, crop_w = crop.shape[:2]
        sx = 512.0 / crop_w
        sy = 512.0 / crop_h
        sx1 = int(max(0, (mic_x.min() - 3) * sx))
        sx2 = int(min(512, (mic_x.max() + 3) * sx))
        sy1 = int(max(0, (mic_y.min() - 3) * sy))
        sy2 = int(min(512, (mic_y.max() + 3) * sy))

        roi = seg[sy1:sy2, sx1:sx2]
        if roi.size == 0:
            return -1.0
        return float(np.isin(roi, list(LIP_CLASSES)).sum()) / roi.size

    # ------------------------------------------------------------------
    # XGBoost features
    # ------------------------------------------------------------------

    def _build_raw_row(self, bs_dict: dict, yaw: float,
                       pitch: float, roll: float) -> np.ndarray:
        row = []
        for name in self._raw_features:
            if name.startswith("bs_"):
                row.append(bs_dict.get(name[3:], 0.0))
            elif name == "yaw":
                row.append(yaw)
            elif name == "pitch":
                row.append(pitch)
            elif name == "roll":
                row.append(roll)
            else:
                row.append(0.0)
        return np.array(row, dtype=np.float32)

    def _compute_features(self, ts: _TrackState) -> np.ndarray:
        win = np.array(list(ts.ring))
        current = win[-1]
        w_mean = np.mean(win, axis=0)
        w_std = np.std(win, axis=0)
        w_min = np.min(win, axis=0)
        w_max = np.max(win, axis=0)
        w_range = w_max - w_min
        diffs = np.diff(win, axis=0)
        w_dmean = np.mean(np.abs(diffs), axis=0) if len(diffs) > 0 else np.zeros(self._n_raw)
        return np.concatenate([current, w_mean, w_std, w_range,
                               w_dmean, w_min, w_max]).reshape(1, -1)

    # ------------------------------------------------------------------
    # Hysteresis
    # ------------------------------------------------------------------

    def _hysteresis(self, ts: _TrackState, prob: float,
                    yaw_abs: float) -> str:
        if yaw_abs > 45:
            extra = int((yaw_abs - 45) / 5)
            on_needed = self._base_on_frames + extra
            off_needed = self._base_off_frames + extra
        else:
            on_needed = self._base_on_frames
            off_needed = self._base_off_frames

        if prob >= self._on_thresh:
            ts.hyst_on += 1
            ts.hyst_off = 0
        elif prob < self._off_thresh:
            ts.hyst_off += 1
            ts.hyst_on = 0
        else:
            ts.hyst_on = max(0, ts.hyst_on - 1)
            ts.hyst_off = max(0, ts.hyst_off - 1)

        if ts.hyst_on >= on_needed:
            ts.hyst_status = "speaking"
        elif ts.hyst_off >= off_needed:
            ts.hyst_status = "not_speaking"

        return ts.hyst_status

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_track(self, track_id: int) -> _TrackState:
        if track_id not in self._tracks:
            self._tracks[track_id] = _TrackState(self._window)
        return self._tracks[track_id]

    @staticmethod
    def _extract_ypr(matrix) -> Tuple[float, float, float]:
        m = np.array(matrix).reshape(4, 4)
        yaw = float(np.degrees(np.arctan2(m[0, 2], m[0, 0])))
        sy = np.sqrt(m[0, 0] ** 2 + m[1, 0] ** 2)
        pitch = float(np.degrees(np.arctan2(-m[2, 0], sy)))
        roll = float(np.degrees(np.arctan2(m[2, 1], m[2, 2])))
        return yaw, pitch, roll

    @staticmethod
    def _emit(status: str, prob: float, reason: str,
              ts: float) -> MouthState:
        return MouthState(
            status=status,
            speaking_prob=prob,
            occlusion_prob=1.0 if "occlu" in status else 0.0,
            confidence=0.8,
            reason_code=reason,
            timestamp_ms=ts,
        )
