"""
BiSeNet Face Parsing occlusion detection — Frame skip effect test.

Model input fixed at 512×512, cannot reduce resolution.
Test frame skip strategy: Run BiSeNet every N frames, reuse last result in between.

Display: Actual inference time, effective per-frame time, lip_ratio, detection status.

Press 1/3/5/8 to switch frame skip interval, press q to quit
"""
import time
import cv2
import numpy as np
import onnxruntime as ort

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src.speaking.mesh_detector import ALL_MOUTH_INDICES

LIP_CLASSES = {11, 12, 13}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _face_bbox_from_landmarks(pts, img_h, img_w, pad=0.3):
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    bw = x_max - x_min
    bh = y_max - y_min
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = max(bw, bh) / 2 * (1 + pad)
    return (max(0, int(cx - half)), max(0, int(cy - half)),
            min(img_w, int(cx + half)), min(img_h, int(cy + half)))


def _bisenet_preprocess(face_crop_bgr):
    img = cv2.resize(face_crop_bgr, (512, 512))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blob = img_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    blob = (blob - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(blob, axis=0)


def _classify(lip_ratio, yaw):
    if abs(yaw) > 60:
        return "SELF_OCC"
    if lip_ratio > 0.25:
        return "VISIBLE"
    if lip_ratio > 0.10:
        return "PARTIAL"
    return "OCCLUDED"


STATUS_COLORS = {
    "VISIBLE": (0, 255, 0),
    "PARTIAL": (0, 200, 255),
    "OCCLUDED": (0, 0, 255),
    "SELF_OCC": (0, 128, 255),
    "NO_FACE": (128, 128, 128),
}


def _make_color_map(seg_map, roi_box):
    color_map = np.zeros((512, 512, 3), dtype=np.uint8)
    color_map[seg_map == 1] = (180, 180, 180)
    color_map[seg_map == 10] = (180, 220, 180)
    color_map[seg_map == 11] = (0, 0, 255)
    color_map[seg_map == 12] = (0, 128, 255)
    color_map[seg_map == 13] = (0, 200, 255)
    color_map[seg_map == 2] = (255, 200, 100)
    color_map[seg_map == 3] = (255, 200, 100)
    color_map[seg_map == 4] = (255, 100, 100)
    color_map[seg_map == 5] = (255, 100, 100)
    color_map[seg_map == 17] = (80, 50, 30)
    sx1, sy1, sx2, sy2 = roi_box
    cv2.rectangle(color_map, (sx1, sy1), (sx2, sy2), (255, 0, 255), 2)
    return cv2.resize(color_map, (180, 180))


def main():
    print("=== BiSeNet Occlusion — Skip-frame Test ===")
    print("Keys: 1=every frame, 3=every 3, 5=every 5, 8=every 8, q=quit")

    base_options = mp_python.BaseOptions(model_asset_path="models/face_landmarker.task")
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    bisenet = ort.InferenceSession("models/speaking/resnet18.onnx", sess_options=sess_opts)
    bis_input_name = bisenet.get_inputs()[0].name

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    skip_n = 1
    frame_idx = 0
    cached_seg = None
    cached_lip_ratio = 0.0
    cached_roi_box = (0, 0, 1, 1)
    last_bis_ms = 0.0
    bis_run_count = 0
    total_bis_ms = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        display = frame.copy()
        yaw = 0.0
        status = "NO_FACE"
        lip_ratio = cached_lip_ratio
        ran_bisenet = False

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            pts = np.array([(l.x * w, l.y * h) for l in lm], dtype=np.float32)

            if result.facial_transformation_matrixes:
                m = np.array(result.facial_transformation_matrixes[0]).reshape(4, 4)
                yaw = float(np.degrees(np.arctan2(m[0, 2], m[0, 0])))

            fx1, fy1, fx2, fy2 = _face_bbox_from_landmarks(pts, h, w, pad=0.3)
            face_crop = frame[fy1:fy2, fx1:fx2]

            if face_crop.size > 0:
                mouth_pts = pts[ALL_MOUTH_INDICES]
                mic_x = mouth_pts[:, 0] - fx1
                mic_y = mouth_pts[:, 1] - fy1
                crop_h, crop_w = face_crop.shape[:2]

                run_this_frame = (frame_idx % skip_n == 0) or cached_seg is None

                if run_this_frame:
                    t0 = time.perf_counter()
                    blob = _bisenet_preprocess(face_crop)
                    outputs = bisenet.run(None, {bis_input_name: blob})
                    parsing = outputs[0]
                    if parsing.ndim == 4:
                        cached_seg = np.argmax(parsing[0], axis=0)
                    else:
                        cached_seg = parsing[0]
                    last_bis_ms = (time.perf_counter() - t0) * 1000
                    bis_run_count += 1
                    total_bis_ms += last_bis_ms
                    ran_bisenet = True

                    scale_x = 512.0 / crop_w
                    scale_y = 512.0 / crop_h
                    sx1 = int(max(0, (mic_x.min() - 3) * scale_x))
                    sx2 = int(min(512, (mic_x.max() + 3) * scale_x))
                    sy1 = int(max(0, (mic_y.min() - 3) * scale_y))
                    sy2 = int(min(512, (mic_y.max() + 3) * scale_y))
                    cached_roi_box = (sx1, sy1, sx2, sy2)

                    roi_seg = cached_seg[sy1:sy2, sx1:sx2]
                    if roi_seg.size > 0:
                        lip_pixels = np.isin(roi_seg, list(LIP_CLASSES)).sum()
                        cached_lip_ratio = lip_pixels / roi_seg.size
                    else:
                        cached_lip_ratio = 0.0
                    lip_ratio = cached_lip_ratio

                status = _classify(lip_ratio, yaw)

                mx1 = int(mouth_pts[:, 0].min() - 5)
                my1 = int(mouth_pts[:, 1].min() - 5)
                mx2 = int(mouth_pts[:, 0].max() + 5)
                my2 = int(mouth_pts[:, 1].max() + 5)
                cv2.rectangle(display, (mx1, my1), (mx2, my2), (255, 0, 255), 2)
                cv2.rectangle(display, (fx1, fy1), (fx2, fy2), (255, 255, 0), 1)

                if cached_seg is not None:
                    mini = _make_color_map(cached_seg, cached_roi_box)
                    display[10:190, w - 190:w - 10] = mini

        sc = STATUS_COLORS.get(status, (255, 255, 255))
        avg_bis = total_bis_ms / bis_run_count if bis_run_count else 0
        effective_ms = avg_bis / skip_n

        lines = [
            f"skip: every {skip_n} frame{'s' if skip_n > 1 else ''}  [press 1/3/5/8]",
            f"bis: {last_bis_ms:.0f}ms  effective: {effective_ms:.1f}ms/f",
            f"yaw: {yaw:.0f}  lip: {lip_ratio:.3f}  {'*RUN*' if ran_bisenet else '(cached)'}",
        ]
        y_pos = 30
        for line in lines:
            cv2.putText(display, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y_pos += 24

        cv2.putText(display, status, (10, y_pos + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, sc, 3, cv2.LINE_AA)

        cv2.imshow("BiSeNet Occlusion (skip-frame)", display)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            skip_n = 1
            print(f">> skip = every 1 frame")
        elif key == ord('3'):
            skip_n = 3
            print(f">> skip = every 3 frames")
        elif key == ord('5'):
            skip_n = 5
            print(f">> skip = every 5 frames")
        elif key == ord('8'):
            skip_n = 8
            print(f">> skip = every 8 frames")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    if bis_run_count > 0:
        print(f"\n=== Summary ===")
        print(f"BiSeNet runs: {bis_run_count}/{frame_idx} frames")
        print(f"Avg inference: {total_bis_ms / bis_run_count:.1f}ms")
        print(f"Last skip_n: {skip_n} → effective {total_bis_ms / bis_run_count / skip_n:.1f}ms/frame")


if __name__ == "__main__":
    main()
