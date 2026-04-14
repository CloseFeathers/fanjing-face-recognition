#!/usr/bin/env python3
"""
Module 2+3+4 验收入口 —— Tracking + Alignment + Embedding 管线。

用法:
  python run_tracking.py --camera 0
  python run_tracking.py --video path/to/video.mp4 --realtime --log
  python run_tracking.py --camera 0 --det-every-n 3 --log
  python run_tracking.py --camera 0 --align --log
  python run_tracking.py --camera 0 --align --embed --log  # 启用 embedding + person matching
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

from src.ingestion.camera_source import CameraSource
from src.ingestion.video_source import VideoSource
from src.detectors.scrfd_detector import SCRFDDetector
from src.tracking.bot_sort import BoTSORTTracker, BoTSORTConfig
from src.tracking.track import TrackState
from src.tracking.draw import draw_tracks
from src.alignment.aligner import FaceAligner
from src.alignment.quality import QualityGate, QualityConfig
from src.alignment.track_sampler import TrackSampler

# Embedding 相关 (可选导入，模型可能未下载)
try:
    from src.embedding import (
        ArcFaceEmbedder,
        TrackTemplateManager,
        PersonRegistry,
        EmbeddingLogger,
    )
    EMBEDDING_AVAILABLE = True
except ImportError as e:
    EMBEDDING_AVAILABLE = False
    print(f"[Warning] Embedding 模块不可用: {e}")

WINDOW_NAME = "Tracking Preview"

# ======================================================================
# 全局状态 (用于 embedding pipeline)
# ======================================================================

# track_id -> person_id 映射 (用于显示)
_track_to_person: Dict[int, int] = {}

# track_id -> 最新相似度 (用于显示)
_track_similarities: Dict[int, float] = {}


# ======================================================================
# JSONL Logger (tracks)
# ======================================================================

class TrackLogger:
    def __init__(self, path: str = "output/tracks.jsonl") -> None:
        self._path = Path(path)
        self._fp = None

    def open(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self._path, "w", encoding="utf-8")
        return self

    def log(self, d: dict):
        if self._fp:
            self._fp.write(json.dumps(d, ensure_ascii=False) + "\n")
            self._fp.flush()

    def close(self):
        if self._fp:
            self._fp.close()
            self._fp = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()


class FPSCounter:
    def __init__(self):
        self._start = time.monotonic()
        self._count = 0
        self._fps = 0.0

    def tick(self) -> float:
        self._count += 1
        now = time.monotonic()
        elapsed = now - self._start
        if elapsed >= 1.0:
            self._fps = self._count / elapsed
            self._count = 0
            self._start = now
        return self._fps

    @property
    def fps(self):
        return self._fps


# ======================================================================
# 对齐 + 采样 + Embedding 步骤
# ======================================================================

# 最近一次对齐成功的人脸 (用于 debug 预览窗口)
_last_aligned_face = None


def alignment_step(
    frame_image,
    tracker,
    ft,
    aligner,
    qgate,
    sampler,
    embedder=None,
    template_mgr=None,
    person_registry=None,
    emb_logger=None,
):
    """在 tracker.step() 之后执行对齐、质量采样、embedding 提取。

    流程:
    1. warpAffine 对齐
    2. 在对齐图上评估质量
    3. 通过则保存人脸图像
    4. 如果启用 embedding: 提取 embedding → 聚合 track template → person matching

    只处理检测帧中 confirmed + 有 kps5 的 track。
    """
    global _last_aligned_face, _track_to_person, _track_similarities

    if not ft.did_detect:
        return 0, 0

    evaluated = 0
    sampled = 0

    for strack in tracker.last_active_stracks:
        if strack.state != TrackState.CONFIRMED:
            continue
        if strack.kps5 is None or strack.det_score is None:
            continue

        # 1. 先做仿射对齐 (warpAffine)
        aligned = aligner.align(frame_image, strack.kps5)
        if aligned is None:
            continue

        evaluated += 1

        # 更新 debug 预览
        _last_aligned_face = aligned.copy()

        # 2. 在对齐后的 112x112 图上评估质量
        bbox = strack.bbox_xyxy_clipped(ft.width, ft.height)
        quality = qgate.evaluate(
            aligned, bbox, strack.det_score, strack.kps5,
        )

        # 3. 通过则保存人脸图像
        info = sampler.try_add(
            track_id=strack.track_id,
            frame_id=ft.frame_id,
            timestamp_ms=ft.timestamp_ms,
            aligned_face=aligned,
            quality=quality,
        )
        if info is None:
            continue

        sampled += 1

        # 4. Embedding 流程 (如果启用)
        if embedder is not None and quality.passed:
            embedding = embedder.extract(aligned)
            if embedding is not None:
                # 4a. 记录 embedding
                if emb_logger is not None:
                    emb_logger.log(
                        track_id=strack.track_id,
                        frame_id=ft.frame_id,
                        timestamp_ms=ft.timestamp_ms,
                        embedding=embedding,
                        quality_score=quality.score,
                        face_image_path=info.save_path,
                    )

                # 4b. 添加到 track template manager
                if template_mgr is not None:
                    template = template_mgr.add_sample(
                        track_id=strack.track_id,
                        frame_id=ft.frame_id,
                        timestamp_ms=ft.timestamp_ms,
                        embedding=embedding,
                        quality_score=quality.score,
                        image_path=info.save_path,
                    )

                    # 4c. 如果生成了 template，执行 person matching
                    if template is not None and person_registry is not None:
                        assignment = person_registry.assign(template)

                        # 更新全局映射 (供显示用)
                        _track_to_person[strack.track_id] = assignment.person_id
                        _track_similarities[strack.track_id] = assignment.top1_similarity

    return evaluated, sampled


# ======================================================================
# 摄像头
# ======================================================================

def run_camera(
    device,
    detector,
    tracker,
    det_every_n,
    track_logger,
    aligner,
    qgate,
    sampler,
    debug_align=False,
    embedder=None,
    template_mgr=None,
    person_registry=None,
    emb_logger=None,
    show_person=True,
    show_similarity=False,
):
    embed_mode = embedder is not None
    print(f"[Pipeline] camera mode  device={device}  det_every_n={det_every_n}")
    print(f"[Pipeline] align={'ON' if aligner else 'OFF'}  embed={'ON' if embed_mode else 'OFF'}  press q to quit")

    fps_c = FPSCounter()
    with CameraSource(device=device) as src:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if debug_align and aligner:
            cv2.namedWindow("Aligned Face (debug)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Aligned Face (debug)", 224, 224)
        ft = None
        while True:
            frame = src.read(timeout=5.0)
            if frame is None:
                continue

            do_det = (frame.frame_id % det_every_n == 0)
            ft = tracker.step(frame, detector, do_detect=do_det)
            pfps = fps_c.tick()

            if aligner and sampler and do_det:
                alignment_step(
                    frame.image, tracker, ft, aligner, qgate, sampler,
                    embedder, template_mgr, person_registry, emb_logger,
                )

            det_faces = len([t for t in ft.tracks if t.get("det_score") is not None]) if do_det else 0
            person_count = person_registry.get_person_count() if person_registry else 0

            display = draw_tracks(
                frame.image, ft, pfps, src.dropped_frames, det_faces,
                track_to_person=_track_to_person if show_person else None,
                show_person=show_person,
                show_similarity=show_similarity,
                track_similarities=_track_similarities if show_similarity else None,
                person_count=person_count,
            )
            cv2.imshow(WINDOW_NAME, display)

            if debug_align and _last_aligned_face is not None:
                big = cv2.resize(_last_aligned_face, (224, 224), interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Aligned Face (debug)", big)

            if track_logger:
                track_logger.log(ft.to_dict())

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    if ft:
        print(f"[Pipeline] done. last frame={ft.frame_id}, dropped={src.dropped_frames}")
    _print_sampler_stats(sampler)
    _print_person_stats(person_registry)


# ======================================================================
# 视频
# ======================================================================

def run_video(
    path,
    realtime,
    detector,
    tracker,
    det_every_n,
    track_logger,
    aligner,
    qgate,
    sampler,
    debug_align=False,
    embedder=None,
    template_mgr=None,
    person_registry=None,
    emb_logger=None,
    show_person=True,
    show_similarity=False,
):
    embed_mode = embedder is not None
    print(f"[Pipeline] video mode  path={path}  realtime={realtime}  det_every_n={det_every_n}")
    print(f"[Pipeline] align={'ON' if aligner else 'OFF'}  embed={'ON' if embed_mode else 'OFF'}  press q to quit")

    fps_c = FPSCounter()
    with VideoSource(path=path, realtime=realtime) as src:
        print(f"[Pipeline] video fps={src.video_fps:.2f}, total_frames={src.total_frames}")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if debug_align and aligner:
            cv2.namedWindow("Aligned Face (debug)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Aligned Face (debug)", 224, 224)

        last_ft = None
        while True:
            frame = src.read()
            if frame is None:
                print("[Pipeline] video finished.")
                break

            do_det = (frame.frame_id % det_every_n == 0)
            ft = tracker.step(frame, detector, do_detect=do_det)
            last_ft = ft
            pfps = fps_c.tick()

            if aligner and sampler and do_det:
                alignment_step(
                    frame.image, tracker, ft, aligner, qgate, sampler,
                    embedder, template_mgr, person_registry, emb_logger,
                )

            det_faces = len([t for t in ft.tracks if t.get("det_score") is not None]) if do_det else 0
            person_count = person_registry.get_person_count() if person_registry else 0

            display = draw_tracks(
                frame.image, ft, pfps, src.dropped_frames, det_faces,
                track_to_person=_track_to_person if show_person else None,
                show_person=show_person,
                show_similarity=show_similarity,
                track_similarities=_track_similarities if show_similarity else None,
                person_count=person_count,
            )
            cv2.imshow(WINDOW_NAME, display)

            if debug_align and _last_aligned_face is not None:
                big = cv2.resize(_last_aligned_face, (224, 224), interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Aligned Face (debug)", big)

            if track_logger:
                track_logger.log(ft.to_dict())

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[Pipeline] user abort.")
                break

    cv2.destroyAllWindows()
    if last_ft:
        print(f"[Pipeline] done. last frame={last_ft.frame_id}, ts={last_ft.timestamp_ms:.1f}ms")
    _print_sampler_stats(sampler)
    _print_person_stats(person_registry)


def _print_sampler_stats(sampler):
    if sampler is None:
        return
    print(f"[Alignment] evaluated={sampler.total_evaluated}  "
          f"passed={sampler.total_passed}  saved={sampler.total_saved}")
    for tid, samples in sampler.get_all_tracks().items():
        best = max(samples, key=lambda s: s.quality_score) if samples else None
        print(f"  track #{tid}: {len(samples)} samples"
              f"  best_q={best.quality_score:.3f}" if best else "")


def _print_person_stats(person_registry):
    if person_registry is None:
        return
    persons = person_registry.get_all_persons()
    print(f"[Embedding] total_persons={len(persons)}")
    for pid, person in persons.items():
        print(f"  person #{pid}: tracks={person.track_ids}  "
              f"samples={person.sample_count}  "
              f"frames={person.first_seen_frame}-{person.last_seen_frame}")


# ======================================================================
# CLI
# ======================================================================

def main():
    p = argparse.ArgumentParser(
        description="Module 2+3+4: Tracking + Alignment + Embedding pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--camera", type=int, metavar="DEVICE")
    g.add_argument("--video", type=str, metavar="PATH")
    p.add_argument("--realtime", action="store_true")

    # 检测器
    p.add_argument("--model", default="models/det_10g.onnx")
    p.add_argument("--det-size", type=int, default=640)
    p.add_argument("--gpu", type=int, default=-1)

    # Tracker
    p.add_argument("--det-threshold", type=float, default=0.4)
    p.add_argument("--high-thres", type=float, default=0.5)
    p.add_argument("--low-thres", type=float, default=0.1)
    p.add_argument("--match-iou-threshold", type=float, default=0.3)
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--max-age", type=int, default=30)
    p.add_argument("--det-every-n", type=int, default=1)

    # Alignment
    p.add_argument("--align", action="store_true",
                   help="Enable face alignment + quality sampling")
    p.add_argument("--align-size", type=int, default=112)
    p.add_argument("--max-samples", type=int, default=10,
                   help="Max samples per track (default: 10)")
    p.add_argument("--min-quality-det", type=float, default=0.60,
                   help="Min det_score for quality gate (default: 0.60)")
    p.add_argument("--min-quality-area", type=float, default=3600,
                   help="Min bbox area for quality gate (default: 3600)")
    p.add_argument("--min-quality-blur", type=float, default=15.0,
                   help="Min blur score on aligned 112x112 (default: 15.0)")
    p.add_argument("--debug-align", action="store_true",
                   help="Open extra window showing aligned face in real-time")

    # Embedding (Module 4)
    p.add_argument("--embed", action="store_true",
                   help="Enable embedding extraction + person matching (requires --align)")
    p.add_argument("--arcface-model", default="models/w600k_r50.onnx",
                   help="ArcFace ONNX model path")
    p.add_argument("--similarity-threshold", type=float, default=0.4,
                   help="Similarity threshold for person matching (default: 0.4)")
    p.add_argument("--margin-threshold", type=float, default=0.1,
                   help="Margin threshold: top1 - top2 > margin (default: 0.1)")
    p.add_argument("--min-template-samples", type=int, default=3,
                   help="Min samples to generate track template (default: 3)")
    p.add_argument("--show-similarity", action="store_true",
                   help="Show similarity score on each face box")

    # 日志
    p.add_argument("--log", action="store_true")
    p.add_argument("--log-path", default="output/tracks.jsonl")

    args = p.parse_args()

    # 加载检测器
    detector = SCRFDDetector(
        model_path=args.model,
        det_size=(args.det_size, args.det_size),
        det_thresh=args.det_threshold,
        gpu_id=args.gpu,
    ).load()

    # Tracker
    cfg = BoTSORTConfig(
        det_threshold=args.det_threshold,
        high_thres=args.high_thres,
        low_thres=args.low_thres,
        match_iou_threshold=args.match_iou_threshold,
        min_hits=args.min_hits,
        max_age=args.max_age,
    )
    tracker = BoTSORTTracker(cfg)
    print(f"[Tracker] config: {cfg}")

    # Alignment
    aligner, qgate, sampler = None, None, None
    if args.align:
        aligner = FaceAligner(output_size=(args.align_size, args.align_size))
        qgate = QualityGate(QualityConfig(
            min_det_score=args.min_quality_det,
            min_bbox_area=args.min_quality_area,
            min_blur_score=args.min_quality_blur,
        ))
        sampler = TrackSampler(
            max_samples=args.max_samples,
            output_dir="output/faces",
            log_path="output/track_faces.jsonl",
        ).open()
        print(f"[Alignment] ON: size={args.align_size}x{args.align_size}, "
              f"max_samples={args.max_samples}")

    # Embedding (Module 4)
    embedder, template_mgr, person_registry, emb_logger = None, None, None, None
    if args.embed:
        if not args.align:
            print("[Error] --embed requires --align")
            sys.exit(1)
        if not EMBEDDING_AVAILABLE:
            print("[Error] Embedding module not available. Check installation.")
            sys.exit(1)
        if not os.path.exists(args.arcface_model):
            print(f"[Error] ArcFace model not found: {args.arcface_model}")
            print("[Info] Run: python scripts/download_arcface.py")
            sys.exit(1)

        embedder = ArcFaceEmbedder(model_path=args.arcface_model)
        template_mgr = TrackTemplateManager(
            min_samples=args.min_template_samples,
            max_samples=args.max_samples,
            aggregation="quality_weighted",
            log_path="output/track_templates.jsonl",
        ).open()
        person_registry = PersonRegistry(
            similarity_threshold=args.similarity_threshold,
            margin_threshold=args.margin_threshold,
            log_path="output/person_assignments.jsonl",
        ).open()
        emb_logger = EmbeddingLogger(
            output_dir="output/embeddings",
            log_path="output/embeddings.jsonl",
        ).open()
        print(f"[Embedding] ON: sim_thresh={args.similarity_threshold}, "
              f"margin_thresh={args.margin_threshold}, "
              f"min_samples={args.min_template_samples}")

    track_logger = None
    if args.log:
        track_logger = TrackLogger(args.log_path).open()
        print(f"[Log] tracks -> {args.log_path}")

    debug_align = args.debug_align and args.align
    show_person = args.embed
    show_similarity = args.show_similarity and args.embed

    try:
        if args.camera is not None:
            run_camera(
                args.camera, detector, tracker, args.det_every_n,
                track_logger, aligner, qgate, sampler, debug_align,
                embedder, template_mgr, person_registry, emb_logger,
                show_person, show_similarity,
            )
        else:
            run_video(
                args.video, args.realtime, detector, tracker, args.det_every_n,
                track_logger, aligner, qgate, sampler, debug_align,
                embedder, template_mgr, person_registry, emb_logger,
                show_person, show_similarity,
            )
    except KeyboardInterrupt:
        print("\n[Pipeline] Ctrl+C")
    finally:
        if track_logger:
            track_logger.close()
            print("[Log] tracks saved.")
        if sampler:
            sampler.close()
            print("[Alignment] track_faces.jsonl saved.")
        if template_mgr:
            template_mgr.close()
            print("[Embedding] track_templates.jsonl saved.")
        if person_registry:
            person_registry.close()
            print("[Embedding] person_assignments.jsonl saved.")
        if emb_logger:
            emb_logger.close()
            print("[Embedding] embeddings.jsonl saved.")


if __name__ == "__main__":
    main()
