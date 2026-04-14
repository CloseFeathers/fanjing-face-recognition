"""
Frame 数据结构 —— 整个系统的帧流通单元。

图像格式固定为 BGR（OpenCV 原生格式），避免无谓转换开销。
后续模块如需 RGB 请自行转换。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass(slots=True)
class Frame:
    """一帧画面及其元数据。

    Attributes:
        image:        BGR uint8 numpy 数组, shape = (H, W, 3)
        timestamp_ms: 毫秒级时间戳
                      - 摄像头模式: 来自单调时钟 (time.monotonic)
                      - 视频文件模式: 来自视频自身播放位置
        frame_id:     从 0 开始的递增序号
        source_id:    标识来源, 如 "camera:0" 或 "file:/path/to/video.mp4"
        width:        帧宽度 (像素)
        height:       帧高度 (像素)
        dropped_frames: 截至当前帧被丢弃的累计帧数
        extra:        预留扩展字段
    """

    image: np.ndarray
    timestamp_ms: float
    frame_id: int
    source_id: str
    width: int
    height: int
    dropped_frames: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------
    def meta_dict(self) -> Dict[str, Any]:
        """返回不含图像像素的元数据字典（可直接序列化为 JSON）。"""
        return {
            "timestamp_ms": round(self.timestamp_ms, 3),
            "frame_id": self.frame_id,
            "source_id": self.source_id,
            "width": self.width,
            "height": self.height,
            "dropped_frames": self.dropped_frames,
        }

    def __repr__(self) -> str:
        return (
            f"Frame(id={self.frame_id}, ts={self.timestamp_ms:.1f}ms, "
            f"src={self.source_id!r}, {self.width}x{self.height}, "
            f"dropped={self.dropped_frames})"
        )
