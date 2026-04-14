"""
JSONL 帧日志记录器 —— 将每帧元数据追加写入 .jsonl 文件。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, TextIO

from .frame import Frame


class FrameLogger:
    """将 Frame 元数据以 JSONL 格式逐行写入文件。

    Parameters:
        path: 输出文件路径, 默认 output/ingestion_frames.jsonl
    """

    def __init__(self, path: str = "output/ingestion_frames.jsonl") -> None:
        self._path = Path(path)
        self._fp: Optional[TextIO] = None

    def open(self) -> "FrameLogger":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self._path, "w", encoding="utf-8")
        return self

    def log(self, frame: Frame) -> None:
        if self._fp is None:
            return
        line = json.dumps(frame.meta_dict(), ensure_ascii=False)
        self._fp.write(line + "\n")
        self._fp.flush()

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    # context-manager
    def __enter__(self) -> "FrameLogger":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()
