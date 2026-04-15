#!/usr/bin/env python3
"""
Module 0 Acceptance Entry — Camera / Video file preview + JSONL logging.

Usage:
  # Camera mode (press q to quit)
  python run_ingestion.py --camera 0

  # Video file mode - run as fast as possible
  python run_ingestion.py --video path/to/video.mp4

  # Video file mode - play at original frame rate
  python run_ingestion.py --video path/to/video.mp4 --realtime

  # Add --log to write frame metadata to output/ingestion_frames.jsonl
  python run_ingestion.py --camera 0 --log
  python run_ingestion.py --video path/to/video.mp4 --log

  # Custom log path
  python run_ingestion.py --camera 0 --log --log-path my_log.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Windows terminal UTF-8 output protection
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
# Camera Preview
# ======================================================================

def run_camera(device: int, logger: FrameLogger | None) -> None:
    print(f"[Ingestion] Camera mode  device={device}")
    print("[Ingestion] Press q to quit")

    with CameraSource(device=device) as src:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            frame = src.read(timeout=5.0)
            if frame is None:
                print("[Ingestion] Read timeout, retrying...")
                continue

            display = draw_overlay(frame.image, frame, src.fps)
            cv2.imshow(WINDOW_NAME, display)

            if logger:
                logger.log(frame)

            # waitKey(1) ensures GUI refresh; press q to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    print(f"[Ingestion] Done. total_frames={frame.frame_id + 1}, dropped={src.dropped_frames}")


# ======================================================================
# Video File Preview
# ======================================================================

def run_video(path: str, realtime: bool, logger: FrameLogger | None) -> None:
    print(f"[Ingestion] Video file mode  path={path}  realtime={realtime}")
    print("[Ingestion] Press q to quit")

    with VideoSource(path=path, realtime=realtime) as src:
        print(f"[Ingestion] Video fps={src.video_fps:.2f}, total_frames={src.total_frames}")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        last_frame = None
        while True:
            frame = src.read()
            if frame is None:
                print("[Ingestion] Video playback finished.")
                break

            last_frame = frame
            display = draw_overlay(frame.image, frame, src.fps)
            cv2.imshow(WINDOW_NAME, display)

            if logger:
                logger.log(frame)

            # realtime mode uses waitKey(1); fast mode also uses 1 to keep GUI responsive
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[Ingestion] User aborted.")
                break

    cv2.destroyAllWindows()
    if last_frame:
        print(
            f"[Ingestion] Done. last frame_id={last_frame.frame_id}, "
            f"last ts={last_frame.timestamp_ms:.1f}ms"
        )


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Module 0: Ingestion preview and logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--camera", type=int, metavar="DEVICE",
                       help="Camera device index, e.g. 0")
    group.add_argument("--video", type=str, metavar="PATH",
                       help="Video file path")
    parser.add_argument("--realtime", action="store_true",
                        help="Play video at original frame rate (default: fast-forward)")
    parser.add_argument("--log", action="store_true",
                        help="Write frame metadata to JSONL log")
    parser.add_argument("--log-path", type=str,
                        default="output/ingestion_frames.jsonl",
                        help="JSONL log path (default: output/ingestion_frames.jsonl)")

    args = parser.parse_args()

    logger = None
    if args.log:
        logger = FrameLogger(path=args.log_path)
        logger.open()
        print(f"[Ingestion] Log output -> {args.log_path}")

    try:
        if args.camera is not None:
            run_camera(args.camera, logger)
        else:
            run_video(args.video, args.realtime, logger)
    except KeyboardInterrupt:
        print("\n[Ingestion] Ctrl+C aborted.")
    finally:
        if logger:
            logger.close()
            print(f"[Ingestion] Log saved.")


if __name__ == "__main__":
    main()
