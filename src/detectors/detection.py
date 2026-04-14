"""
Detection 数据结构 —— 人脸检测的统一输出契约。

FrameDetections 是一帧的完整检测结果，可直接序列化为 JSONL 的一行。
FaceDetection 是单张人脸的检测信息。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class FaceDetection:
    """单个人脸检测结果。

    Attributes:
        bbox_xyxy: 像素坐标 [x1, y1, x2, y2]，保证 x1<x2, y1<y2
        score:     检测置信度 (0~1)
        class_id:  固定为 0 表示 face
        kps5:      5 个关键点 [[x,y], ...] 顺序为
                   left_eye, right_eye, nose, left_mouth_corner, right_mouth_corner
                   如果模型不输出关键点则为 None
    """

    bbox_xyxy: List[float]
    score: float
    class_id: int = 0
    kps5: Optional[List[List[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "bbox_xyxy": [round(v, 2) for v in self.bbox_xyxy],
            "score": round(self.score, 4),
            "class_id": self.class_id,
        }
        if self.kps5 is not None:
            d["kps5"] = [[round(x, 2), round(y, 2)] for x, y in self.kps5]
        return d


@dataclass(slots=True)
class FrameDetections:
    """一帧的全部人脸检测结果。

    Attributes:
        timestamp_ms:   来自 Ingestion Frame 的时间戳
        frame_id:       帧序号，与 Ingestion 对齐
        source_id:      数据来源标识
        width:          帧宽度（像素）
        height:         帧高度（像素）
        num_faces:      检测到的人脸数量
        detect_time_ms: 本帧检测耗时（毫秒）
        faces:          人脸列表
    """

    timestamp_ms: float
    frame_id: int
    source_id: str
    width: int
    height: int
    num_faces: int = 0
    detect_time_ms: float = 0.0
    faces: List[FaceDetection] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ms": round(self.timestamp_ms, 3),
            "frame_id": self.frame_id,
            "source_id": self.source_id,
            "width": self.width,
            "height": self.height,
            "num_faces": self.num_faces,
            "detect_time_ms": round(self.detect_time_ms, 2),
            "faces": [f.to_dict() for f in self.faces],
        }
