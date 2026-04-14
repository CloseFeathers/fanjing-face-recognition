#!/usr/bin/env python3
"""
Module 0 验收入口 —— 摄像头 / 视频文件预览 + JSONL 日志。

用法:
  # 摄像头模式（按 q 退出）
  python run_ingestion.py --camera 0

  # 视频文件模式 - 尽可能快跑完
  python run_ingestion.py --video path/to/video.mp4

  # 视频文件模式 - 按原始帧率实时播放
  python run_ingestion.py --video path/to/video.mp4 --realtime

  # 追加 --log 将帧元数据写入 output/ingestion_frames.jsonl
  python run_ingestion.py --camera 0 --log
  python run_ingestion.py --video path/to/video.mp4 --log

  # 自定义日志路径
  python run_ingestion.py --camera 0 --log --log-path my_log.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Windows 终端 UTF-8 输出保护
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import cv2

from src.ingestion.camera_source import CameraSource
from src.ingestion.video_source import VideoSource
from src.ingestion.overlay import draw_overlay
from src.ingestion.logger import FrameLogger


WINDOW_NAME = "Ingestion Preview"


# ======================================================================
# 摄像头预览
# ======================================================================

def run_camera(device: int, logger: FrameLogger | None) -> None:
    print(f"[Ingestion] 摄像头模式  device={device}")
    print("[Ingestion] 按 q 退出")

    with CameraSource(device=device) as src:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            frame = src.read(timeout=5.0)
            if frame is None:
                print("[Ingestion] 读取超时，重试...")
                continue

            display = draw_overlay(frame.image, frame, src.fps)
            cv2.imshow(WINDOW_NAME, display)

            if logger:
                logger.log(frame)

            # waitKey(1) 保证 GUI 刷新；按 q 退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    print(f"[Ingestion] 结束. 总帧数={frame.frame_id + 1}, 丢帧={src.dropped_frames}")


# ======================================================================
# 视频文件预览
# ======================================================================

def run_video(path: str, realtime: bool, logger: FrameLogger | None) -> None:
    print(f"[Ingestion] 视频文件模式  path={path}  realtime={realtime}")
    print("[Ingestion] 按 q 退出")

    with VideoSource(path=path, realtime=realtime) as src:
        print(f"[Ingestion] 视频帧率={src.video_fps:.2f}, 总帧数={src.total_frames}")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        last_frame = None
        while True:
            frame = src.read()
            if frame is None:
                print("[Ingestion] 视频播放完毕.")
                break

            last_frame = frame
            display = draw_overlay(frame.image, frame, src.fps)
            cv2.imshow(WINDOW_NAME, display)

            if logger:
                logger.log(frame)

            # realtime 模式用 waitKey(1)；快跑模式也用 1 以保持 GUI 响应
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[Ingestion] 用户中止.")
                break

    cv2.destroyAllWindows()
    if last_frame:
        print(
            f"[Ingestion] 结束. 最后 frame_id={last_frame.frame_id}, "
            f"最后 ts={last_frame.timestamp_ms:.1f}ms"
        )


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Module 0: Ingestion 预览与日志",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--camera", type=int, metavar="DEVICE",
                       help="摄像头设备号, 例如 0")
    group.add_argument("--video", type=str, metavar="PATH",
                       help="视频文件路径")
    parser.add_argument("--realtime", action="store_true",
                        help="视频文件按原始帧率播放（默认尽可能快）")
    parser.add_argument("--log", action="store_true",
                        help="将帧元数据写入 JSONL 日志")
    parser.add_argument("--log-path", type=str,
                        default="output/ingestion_frames.jsonl",
                        help="JSONL 日志路径 (默认 output/ingestion_frames.jsonl)")

    args = parser.parse_args()

    logger = None
    if args.log:
        logger = FrameLogger(path=args.log_path)
        logger.open()
        print(f"[Ingestion] 日志输出 -> {args.log_path}")

    try:
        if args.camera is not None:
            run_camera(args.camera, logger)
        else:
            run_video(args.video, args.realtime, logger)
    except KeyboardInterrupt:
        print("\n[Ingestion] Ctrl+C 中止.")
    finally:
        if logger:
            logger.close()
            print(f"[Ingestion] 日志已保存.")


if __name__ == "__main__":
    main()
