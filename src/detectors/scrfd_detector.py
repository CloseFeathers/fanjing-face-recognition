"""
SCRFDDetector —— 基于 ONNX Runtime 的 SCRFD 人脸检测器。

无需安装 insightface 包，直接加载 SCRFD ONNX 模型进行推理。

模型来源：
    InsightFace 官方模型仓库 (GitHub Releases)
    默认使用 buffalo_l 包中的 det_10g.onnx (scrfd_10g_bnkps, 10GF, 带 5 点关键点)
    轻量替代: buffalo_sc 包中的 det_500m.onnx (scrfd_500m_bnkps, 500MF)

加载方式：
    使用 onnxruntime.InferenceSession 直接加载 .onnx 文件
    自动检测模型输出数量来判断是否包含关键点 (9 outputs = kps, 6 = no kps)

模型下载：
    python scripts/download_model.py
    模型保存至 models/det_10g.onnx
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

# Windows 终端 UTF-8 保护
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
    """SCRFD 人脸检测器 (纯 ONNX Runtime 实现)。

    Parameters:
        model_path: ONNX 模型文件路径
        det_size:   检测输入尺寸 (H, W), 默认 (640, 640)
        det_thresh: 置信度阈值, 默认 0.5
        nms_thresh: NMS IoU 阈值, 默认 0.4
        gpu_id:     GPU 编号, -1 = CPU
    """

    # SCRFD 预处理常量 (与 insightface 官方一致)
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

        # 运行时状态 (load 后填充)
        self._session: Optional[ort.InferenceSession] = None
        self._input_name: str = ""
        self._fmc: int = 3             # feature map count
        self._strides: List[int] = []
        self._num_anchors: int = 2
        self._use_kps: bool = False
        self._anchor_cache: Dict[str, np.ndarray] = {}

    # ==================================================================
    # 生命周期
    # ==================================================================

    def load(self) -> "SCRFDDetector":
        """加载 ONNX 模型。"""
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"找不到模型文件: {self._model_path}\n"
                f"请先运行: python scripts/download_model.py"
            )

        providers: List[str] = []
        if self._gpu_id >= 0:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        logger.info(f"加载模型 {self._model_path} ...")
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

        # 通过输出数量自动判断模型配置
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
            raise ValueError(f"不支持的 SCRFD 模型输出数: {num_outputs}")

        # 判断输出是否带 batch 维度 (3D → batched, 2D → non-batched)
        self._batched = len(outputs[0].shape) == 3

        kps_str = "有" if self._use_kps else "无"
        batch_str = "batched" if self._batched else "non-batched"
        logger.info(f"模型加载完毕. strides={self._strides}, "
                    f"anchors={self._num_anchors}, 关键点={kps_str}, {batch_str}")
        return self

    # ==================================================================
    # 检测主入口
    # ==================================================================

    def detect(self, frame: Frame) -> FrameDetections:
        """对单帧执行人脸检测。

        Args:
            frame: Ingestion 层输出的 Frame (BGR 图像)

        Returns:
            FrameDetections
        """
        if self._session is None:
            raise RuntimeError("检测器未加载，请先调用 load()")

        img = frame.image  # BGR, uint8
        W, H = frame.width, frame.height

        t0 = time.perf_counter()

        # 1. 预处理: letterbox resize + normalize
        det_img, det_scale = self._preprocess(img)

        # 2. 推理
        blob = cv2.dnn.blobFromImage(
            det_img,
            1.0 / self.INPUT_STD,
            (det_img.shape[1], det_img.shape[0]),  # (W, H)
            (self.INPUT_MEAN, self.INPUT_MEAN, self.INPUT_MEAN),
            swapRB=True,
        )
        net_outs = self._session.run(None, {self._input_name: blob})

        # 3. 后处理: 解码 + NMS
        bboxes, scores, kpss = self._postprocess(
            net_outs, det_img.shape[0], det_img.shape[1]
        )

        t1 = time.perf_counter()
        detect_ms = (t1 - t0) * 1000.0

        # 4. 还原到原图坐标 + 裁剪边界
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
    # 预处理
    # ==================================================================

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """Letterbox resize: 等比缩放 + 右下角填充黑色。

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
    # 后处理
    # ==================================================================

    def _postprocess(
        self,
        net_outs: list,
        input_h: int,
        input_w: int,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """解码网络输出 → (bboxes, scores, kpss)。

        Returns:
            bboxes: (N, 4)  x1y1x2y2 in det_img 坐标
            scores: (N,)
            kpss:   (N, 5, 2) or None
        """
        all_scores = []
        all_bboxes = []
        all_kpss = []

        fmc = self._fmc

        for idx, stride in enumerate(self._strides):
            # 模型输出排列: [score_s1,...,score_sN, bbox_s1,...,bbox_sN, kps_s1,...,kps_sN]
            # batched 模型输出 shape=(1, A, C), non-batched 输出 shape=(A, C)
            if self._batched:
                scores = net_outs[idx][0]                       # (A, 1)
                bbox_preds = net_outs[idx + fmc][0] * stride    # (A, 4)
            else:
                scores = net_outs[idx]                           # (A, 1)
                bbox_preds = net_outs[idx + fmc] * stride        # (A, 4)

            feat_h = input_h // stride
            feat_w = input_w // stride
            anchors = self._get_anchor_centers(feat_h, feat_w, stride)

            # 阈值过滤 (用 emit_thresh 放行弱框给 tracker)
            pos = np.where(scores[:, 0] >= self._emit_thresh)[0]
            if len(pos) == 0:
                continue

            pos_scores = scores[pos, 0]
            pos_bbox = bbox_preds[pos]
            pos_anchors = anchors[pos]

            # 解码 bbox: anchor_center ± distance
            bboxes = self._distance2bbox(pos_anchors, pos_bbox)
            all_scores.append(pos_scores)
            all_bboxes.append(bboxes)

            # 关键点
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
    # 锚点生成 (带缓存)
    # ==================================================================

    def _get_anchor_centers(
        self, feat_h: int, feat_w: int, stride: int
    ) -> np.ndarray:
        """生成特征图网格锚点中心坐标，缓存以避免重复计算。"""
        key = f"{feat_h}_{feat_w}_{stride}"
        if key in self._anchor_cache:
            return self._anchor_cache[key]

        # mgrid[:h, :w] → [row_grid, col_grid]
        # 我们需要 [x, y] 即 [col, row]，所以用 [::-1]
        # 与 insightface 原版一致: center = grid_index * stride (不加偏移)
        grid = np.stack(
            np.mgrid[:feat_h, :feat_w][::-1], axis=-1
        ).astype(np.float32)
        centers = (grid * stride).reshape(-1, 2)

        if self._num_anchors > 1:
            # 每个网格位置复制 num_anchors 份
            centers = np.stack(
                [centers] * self._num_anchors, axis=1
            ).reshape(-1, 2)

        self._anchor_cache[key] = centers
        return centers

    # ==================================================================
    # 解码函数
    # ==================================================================

    @staticmethod
    def _distance2bbox(
        points: np.ndarray, distance: np.ndarray
    ) -> np.ndarray:
        """anchor_center + distance → xyxy bbox。

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
        """anchor_center + offset → 关键点坐标。

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
        """Non-Maximum Suppression，使用 OpenCV 实现。"""
        # cv2.dnn.NMSBoxes 需要 [x, y, w, h] 格式
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
    # 属性
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
