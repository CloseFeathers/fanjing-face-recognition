"""
SCRFDDetector — SCRFD face detector based on ONNX Runtime.

No need to install insightface package, directly load SCRFD ONNX model for inference.

Model source:
    InsightFace official model repository (GitHub Releases)
    Default uses det_10g.onnx from buffalo_l package (scrfd_10g_bnkps, 10GF, with 5-point keypoints)
    Lightweight alternative: det_500m.onnx from buffalo_sc package (scrfd_500m_bnkps, 500MF)

Loading method:
    Use onnxruntime.InferenceSession to directly load .onnx file
    Auto-detect model output count to determine if keypoints included (9 outputs = kps, 6 = no kps)

Model download:
    python scripts/download_model.py
    Model saved to models/det_10g.onnx
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# Windows terminal UTF-8 protection
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

from ..ingestion.frame import Frame
from .detection import FaceDetection, FrameDetections


class SCRFDDetector:
    """SCRFD face detector (pure ONNX Runtime implementation).

    Parameters:
        model_path: ONNX model file path
        det_size:   Detection input size (H, W), default (640, 640)
        det_thresh: Confidence threshold, default 0.5
        nms_thresh: NMS IoU threshold, default 0.4
        gpu_id:     GPU number, -1 = CPU
    """

    # SCRFD preprocessing constants (consistent with insightface official)
    INPUT_MEAN = 127.5
    INPUT_STD = 128.0

    def __init__(
        self,
        model_path: str = "models/det_10g.onnx",
        det_size: Tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        gpu_id: int = -1,
        emit_thresh: Optional[float] = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._det_size = det_size      # (H, W)
        self._det_thresh = det_thresh
        self._emit_thresh = emit_thresh if emit_thresh is not None else det_thresh
        self._nms_thresh = nms_thresh
        self._gpu_id = gpu_id

        # Runtime state (filled after load)
        self._session: Optional[ort.InferenceSession] = None
        self._input_name: str = ""
        self._fmc: int = 3             # feature map count
        self._strides: List[int] = []
        self._num_anchors: int = 2
        self._use_kps: bool = False
        self._anchor_cache: Dict[str, np.ndarray] = {}

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def load(self) -> "SCRFDDetector":
        """Load ONNX model."""
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self._model_path}\n"
                f"Please run: python scripts/download_model.py"
            )

        providers: List[str] = []
        if self._gpu_id >= 0:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        logger.info(f"Loading model {self._model_path} ...")
        det_threads = max((os.cpu_count() or 8) - 2, 4)
        logger.info(f"providers={providers}, det_size={self._det_size}, "
                    f"det_thresh={self._det_thresh}, emit_thresh={self._emit_thresh}, "
                    f"threads={det_threads}")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = det_threads
        sess_opts.inter_op_num_threads = 1

        self._session = ort.InferenceSession(
            str(self._model_path), sess_options=sess_opts, providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name

        # Auto-detect model configuration from output count
        outputs = self._session.get_outputs()
        num_outputs = len(outputs)
        if num_outputs == 9:
            self._fmc = 3
            self._strides = [8, 16, 32]
            self._num_anchors = 2
            self._use_kps = True
        elif num_outputs == 6:
            self._fmc = 3
            self._strides = [8, 16, 32]
            self._num_anchors = 2
            self._use_kps = False
        elif num_outputs == 15:
            self._fmc = 5
            self._strides = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self._use_kps = True
        elif num_outputs == 10:
            self._fmc = 5
            self._strides = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self._use_kps = False
        else:
            raise ValueError(f"Unsupported SCRFD model output count: {num_outputs}")

        # Check if output has batch dimension (3D → batched, 2D → non-batched)
        self._batched = len(outputs[0].shape) == 3

        kps_str = "yes" if self._use_kps else "no"
        batch_str = "batched" if self._batched else "non-batched"
        logger.info(f"Model loaded. strides={self._strides}, "
                    f"anchors={self._num_anchors}, keypoints={kps_str}, {batch_str}")
        return self

    # ==================================================================
    # Detection main entry
    # ==================================================================

    def detect(self, frame: Frame) -> FrameDetections:
        """Execute face detection on single frame.

        Args:
            frame: Frame from Ingestion layer (BGR image)

        Returns:
            FrameDetections
        """
        if self._session is None:
            raise RuntimeError("Detector not loaded, please call load() first")

        img = frame.image  # BGR, uint8
        W, H = frame.width, frame.height

        t0 = time.perf_counter()

        # 1. Preprocess: letterbox resize + normalize
        det_img, det_scale = self._preprocess(img)

        # 2. Inference
        blob = cv2.dnn.blobFromImage(
            det_img,
            1.0 / self.INPUT_STD,
            (det_img.shape[1], det_img.shape[0]),  # (W, H)
            (self.INPUT_MEAN, self.INPUT_MEAN, self.INPUT_MEAN),
            swapRB=True,
        )
        net_outs = self._session.run(None, {self._input_name: blob})

        # 3. Postprocess: decode + NMS
        bboxes, scores, kpss = self._postprocess(
            net_outs, det_img.shape[0], det_img.shape[1]
        )

        t1 = time.perf_counter()
        detect_ms = (t1 - t0) * 1000.0

        # 4. Map back to original image coordinates + clip boundaries
        faces: list[FaceDetection] = []
        for i in range(len(bboxes)):
            bx = bboxes[i] / det_scale
            x1 = float(max(0.0, min(bx[0], W)))
            y1 = float(max(0.0, min(bx[1], H)))
            x2 = float(max(0.0, min(bx[2], W)))
            y2 = float(max(0.0, min(bx[3], H)))
            if x1 >= x2 or y1 >= y2:
                continue

            score = float(scores[i])

            kps5 = None
            if kpss is not None:
                kp = kpss[i] / det_scale  # (5, 2)
                kps5 = [
                    [float(max(0.0, min(kp[j, 0], W))),
                     float(max(0.0, min(kp[j, 1], H)))]
                    for j in range(5)
                ]

            faces.append(FaceDetection(
                bbox_xyxy=[x1, y1, x2, y2],
                score=score,
                class_id=0,
                kps5=kps5,
            ))

        return FrameDetections(
            timestamp_ms=frame.timestamp_ms,
            frame_id=frame.frame_id,
            source_id=frame.source_id,
            width=frame.width,
            height=frame.height,
            num_faces=len(faces),
            detect_time_ms=detect_ms,
            faces=faces,
        )

    # ==================================================================
    # Preprocessing
    # ==================================================================

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """Letterbox resize: proportional scaling + bottom-right black padding.

        Returns:
            (det_img, det_scale)  det_scale = new_size / original_size
        """
        input_h, input_w = self._det_size
        im_h, im_w = img.shape[:2]

        im_ratio = im_h / im_w
        model_ratio = input_h / input_w

        if im_ratio > model_ratio:
            new_h = input_h
            new_w = int(new_h / im_ratio)
        else:
            new_w = input_w
            new_h = int(new_w * im_ratio)

        det_scale = new_h / im_h
        resized = cv2.resize(img, (new_w, new_h))

        det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized
        return det_img, det_scale

    # ==================================================================
    # Postprocessing
    # ==================================================================

    def _postprocess(
        self,
        net_outs: list,
        input_h: int,
        input_w: int,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Decode network outputs → (bboxes, scores, kpss).

        Returns:
            bboxes: (N, 4)  x1y1x2y2 in det_img coordinates
            scores: (N,)
            kpss:   (N, 5, 2) or None
        """
        all_scores = []
        all_bboxes = []
        all_kpss = []

        fmc = self._fmc

        for idx, stride in enumerate(self._strides):
            # Model output arrangement: [score_s1,...,score_sN, bbox_s1,...,bbox_sN, kps_s1,...,kps_sN]
            # Batched model output shape=(1, A, C), non-batched output shape=(A, C)
            if self._batched:
                scores = net_outs[idx][0]                       # (A, 1)
                bbox_preds = net_outs[idx + fmc][0] * stride    # (A, 4)
            else:
                scores = net_outs[idx]                           # (A, 1)
                bbox_preds = net_outs[idx + fmc] * stride        # (A, 4)

            feat_h = input_h // stride
            feat_w = input_w // stride
            anchors = self._get_anchor_centers(feat_h, feat_w, stride)

            # Threshold filter (use emit_thresh to pass weak boxes to tracker)
            pos = np.where(scores[:, 0] >= self._emit_thresh)[0]
            if len(pos) == 0:
                continue

            pos_scores = scores[pos, 0]
            pos_bbox = bbox_preds[pos]
            pos_anchors = anchors[pos]

            # Decode bbox: anchor_center ± distance
            bboxes = self._distance2bbox(pos_anchors, pos_bbox)
            all_scores.append(pos_scores)
            all_bboxes.append(bboxes)

            # Keypoints
            if self._use_kps:
                if self._batched:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride  # (A, 10)
                else:
                    kps_preds = net_outs[idx + fmc * 2] * stride      # (A, 10)
                pos_kps = kps_preds[pos]
                kpss = self._distance2kps(pos_anchors, pos_kps)
                all_kpss.append(kpss)

        if len(all_scores) == 0:
            empty_b = np.zeros((0, 4), dtype=np.float32)
            empty_s = np.zeros((0,), dtype=np.float32)
            return empty_b, empty_s, None

        scores = np.concatenate(all_scores, axis=0)
        bboxes = np.concatenate(all_bboxes, axis=0)
        kpss = np.concatenate(all_kpss, axis=0) if all_kpss else None

        # NMS
        keep = self._nms(bboxes, scores)
        bboxes = bboxes[keep]
        scores = scores[keep]
        if kpss is not None:
            kpss = kpss[keep].reshape(-1, 5, 2)

        return bboxes, scores, kpss

    # ==================================================================
    # Anchor generation (with cache)
    # ==================================================================

    def _get_anchor_centers(
        self, feat_h: int, feat_w: int, stride: int
    ) -> np.ndarray:
        """Generate feature map grid anchor centers, cached to avoid recomputation."""
        key = f"{feat_h}_{feat_w}_{stride}"
        if key in self._anchor_cache:
            return self._anchor_cache[key]

        # mgrid[:h, :w] → [row_grid, col_grid]
        # We need [x, y] i.e. [col, row], so use [::-1]
        # Consistent with insightface original: center = grid_index * stride (no offset)
        grid = np.stack(
            np.mgrid[:feat_h, :feat_w][::-1], axis=-1
        ).astype(np.float32)
        centers = (grid * stride).reshape(-1, 2)

        if self._num_anchors > 1:
            # Duplicate each grid position num_anchors times
            centers = np.stack(
                [centers] * self._num_anchors, axis=1
            ).reshape(-1, 2)

        self._anchor_cache[key] = centers
        return centers

    # ==================================================================
    # Decode functions
    # ==================================================================

    @staticmethod
    def _distance2bbox(
        points: np.ndarray, distance: np.ndarray
    ) -> np.ndarray:
        """anchor_center + distance → xyxy bbox.

        points:   (N, 2)  [cx, cy]
        distance: (N, 4)  [left, top, right, bottom]
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def _distance2kps(
        points: np.ndarray, distance: np.ndarray
    ) -> np.ndarray:
        """anchor_center + offset → keypoint coordinates.

        points:   (N, 2)  [cx, cy]
        distance: (N, 10) [dx0,dy0, dx1,dy1, ..., dx4,dy4]
        returns:  (N, 10) [x0,y0, x1,y1, ..., x4,y4]
        """
        num_points = distance.shape[1] // 2
        result = np.zeros_like(distance)
        for i in range(num_points):
            result[:, i * 2] = points[:, 0] + distance[:, i * 2]
            result[:, i * 2 + 1] = points[:, 1] + distance[:, i * 2 + 1]
        return result

    # ==================================================================
    # NMS
    # ==================================================================

    def _nms(self, bboxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Non-Maximum Suppression, using OpenCV implementation."""
        # cv2.dnn.NMSBoxes requires [x, y, w, h] format
        boxes_xywh = []
        for b in bboxes:
            boxes_xywh.append([float(b[0]), float(b[1]),
                               float(b[2] - b[0]), float(b[3] - b[1])])

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scores.tolist(),
            self._emit_thresh,
            self._nms_thresh,
        )
        if len(indices) == 0:
            return np.array([], dtype=np.int64)
        return np.array(indices).flatten()

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def use_kps(self) -> bool:
        return self._use_kps

    @property
    def model_path(self) -> str:
        return str(self._model_path)
