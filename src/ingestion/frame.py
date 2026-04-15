"""
Frame data structure — The frame circulation unit for the entire system.

Image format is fixed to BGR (OpenCV native format) to avoid unnecessary conversion overhead.
Downstream modules should convert to RGB themselves if needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass(slots=True)
class Frame:
    """A single frame and its metadata.

    Attributes:
        image:        BGR uint8 numpy array, shape = (H, W, 3)
        timestamp_ms: Millisecond timestamp
                      - Camera mode: from monotonic clock (time.monotonic)
                      - Video file mode: from video container playback position
        frame_id:     Incrementing sequence number starting from 0
        source_id:    Source identifier, e.g., "camera:0" or "file:/path/to/video.mp4"
        width:        Frame width (pixels)
        height:       Frame height (pixels)
        dropped_frames: Cumulative dropped frame count up to current frame
        extra:        Reserved extension field
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
    # Convenience methods
    # ------------------------------------------------------------------
    def meta_dict(self) -> Dict[str, Any]:
        """Return metadata dict without image pixels (directly JSON serializable)."""
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
