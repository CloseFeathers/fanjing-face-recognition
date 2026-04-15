"""
Full mouth analysis test — BiSeNet occlusion + XGBoost speaking + Hysteresis state machine.

Three-layer judgment:
  L1 Occlusion: BiSeNet face parsing (every 5 frames) + yaw > 60 self-occlusion
  L2 Speaking: XGBoost (blendshape sliding window features, no VSDLM)
  L3 Debounce: Hysteresis state machine

Press q to quit
"""

import os
import pickle
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src.speaking.mesh_detector import (
    LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER,
    UPPER_LIP_OUTER, LOWER_LIP_OUTER,
    ALL_MOUTH_INDICES,
)

# ======================================================================
# Constants
# ======================================================================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
LIP_CLASSES = {11, 12, 13}

MOUTH_BS_NAMES = [
    "jawOpen", "jawForward", "jawLeft", "jawRight",
    "mouthClose", "mouthFunnel", "mouthPucker",
    "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
]

STATUS_COLORS = {
    "SPEAKING": (0, 255, 0),
    "NOT_SPEAKING": (180, 180, 180),
    "OCCLUDED": (0, 0, 255),
    "SELF_OCCLUDED": (0, 128, 255),
    "UNOBSERVABLE": (100, 100, 100),
    "WARMING_UP": (200, 200, 0),
}


# ======================================================================
# BiSeNet helpers
# ======================================================================

def face_bbox_from_landmarks(pts, img_h, img_w, pad=0.3):
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    bw, bh = x_max - x_min, y_max - y_min
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = max(bw, bh) / 2 * (1 + pad)
    return (max(0, int(cx - half)), max(0, int(cy - half)),
            min(img_w, int(cx + half)), min(img_h, int(cy + half)))


def bisenet_infer(bisenet, input_name, face_crop_bgr):
    img = cv2.resize(face_crop_bgr, (512, 512))
    blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0
    blob = (blob - IMAGENET_MEAN) / IMAGENET_STD
    blob = np.expand_dims(blob, axis=0)
    out = bisenet.run(None, {input_name: blob})
    parsing = out[0]
    if parsing.ndim == 4:
        return np.argmax(parsing[0], axis=0)
    return parsing[0]


def compute_lip_ratio(seg_map, mouth_pts, crop_origin, crop_size):
    fx1, fy1 = crop_origin
    crop_h, crop_w = crop_size
    mic_x = mouth_pts[:, 0] - fx1
    mic_y = mouth_pts[:, 1] - fy1
    sx = 512.0 / crop_w
    sy = 512.0 / crop_h
    sx1 = int(max(0, (mic_x.min() - 3) * sx))
    sx2 = int(min(512, (mic_x.max() + 3) * sx))
    sy1 = int(max(0, (mic_y.min() - 3) * sy))
    sy2 = int(min(512, (mic_y.max() + 3) * sy))
    roi = seg_map[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return 0.0, (sx1, sy1, sx2, sy2)
    lip_px = np.isin(roi, list(LIP_CLASSES)).sum()
    return lip_px / roi.size, (sx1, sy1, sx2, sy2)


def make_seg_colormap(seg_map, roi_box):
    c = np.zeros((512, 512, 3), dtype=np.uint8)
    c[seg_map == 1] = (180, 180, 180)
    c[seg_map == 10] = (180, 220, 180)
    c[seg_map == 11] = (0, 0, 255)
    c[seg_map == 12] = (0, 128, 255)
    c[seg_map == 13] = (0, 200, 255)
    c[seg_map == 4] = (255, 100, 100)
    c[seg_map == 5] = (255, 100, 100)
    c[seg_map == 17] = (80, 50, 30)
    sx1, sy1, sx2, sy2 = roi_box
    cv2.rectangle(c, (sx1, sy1), (sx2, sy2), (255, 0, 255), 2)
    return cv2.resize(c, (160, 160))


# ======================================================================
# XGBoost speaking model wrapper
# ======================================================================

class SpeakingPredictor:
    def __init__(self, model_path="models/speaking/speaking_model.pkl"):
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_cols = data["feature_cols"]
        self.raw_features = data["raw_features"]
        self.window = data["window_size"]
        self.n_raw = len(self.raw_features)

        self.buffers: dict = {}

    def _get_buffer(self, track_id):
        if track_id not in self.buffers:
            self.buffers[track_id] = deque(maxlen=self.window)
        return self.buffers[track_id]

    def update(self, track_id, blendshapes: dict, yaw: float,
               pitch: float, roll: float):
        buf = self._get_buffer(track_id)
        row = []
        for name in self.raw_features:
            if name.startswith("bs_"):
                bs_name = name[3:]
                row.append(blendshapes.get(bs_name, 0.0))
            elif name == "yaw":
                row.append(yaw)
            elif name == "pitch":
                row.append(pitch)
            elif name == "roll":
                row.append(roll)
            else:
                row.append(0.0)
        buf.append(np.array(row, dtype=np.float32))

        if len(buf) < 3:
            return None

        win = np.array(list(buf))
        current = win[-1]
        w_mean = np.mean(win, axis=0)
        w_std = np.std(win, axis=0)
        w_min = np.min(win, axis=0)
        w_max = np.max(win, axis=0)
        w_range = w_max - w_min
        diffs = np.diff(win, axis=0)
        w_dmean = np.mean(np.abs(diffs), axis=0) if len(diffs) > 0 else np.zeros(self.n_raw)

        feat = np.concatenate([current, w_mean, w_std, w_range, w_dmean,
                               w_min, w_max])
        feat = feat.reshape(1, -1)
        prob = self.model.predict_proba(feat)[0, 1]
        return float(prob)

    def remove(self, track_id):
        self.buffers.pop(track_id, None)


# ======================================================================
# Hysteresis state machine
# ======================================================================

class HysteresisStateMachine:
    def __init__(self, on_thresh=0.55, off_thresh=0.35,
                 on_frames=4, off_frames=6):
        self.on_thresh = on_thresh
        self.off_thresh = off_thresh
        self.base_on_frames = on_frames
        self.base_off_frames = off_frames
        self.states: dict = {}

    def update(self, track_id, prob, yaw_abs=0.0):
        if track_id not in self.states:
            self.states[track_id] = {"status": "NOT_SPEAKING",
                                      "on_count": 0, "off_count": 0}
        st = self.states[track_id]

        # Adaptive: large yaw → need more consecutive frames to switch
        if yaw_abs > 45:
            extra = int((yaw_abs - 45) / 5)  # +1 per 5° beyond 45
            on_needed = self.base_on_frames + extra
            off_needed = self.base_off_frames + extra
        else:
            on_needed = self.base_on_frames
            off_needed = self.base_off_frames

        if prob >= self.on_thresh:
            st["on_count"] += 1
            st["off_count"] = 0
        elif prob < self.off_thresh:
            st["off_count"] += 1
            st["on_count"] = 0
        else:
            st["on_count"] = max(0, st["on_count"] - 1)
            st["off_count"] = max(0, st["off_count"] - 1)

        if st["on_count"] >= on_needed:
            st["status"] = "SPEAKING"
        elif st["off_count"] >= off_needed:
            st["status"] = "NOT_SPEAKING"

        return st["status"]

    def remove(self, track_id):
        self.states.pop(track_id, None)


# ======================================================================
# Yaw helpers
# ======================================================================

def extract_yaw_pitch_roll(matrix):
    m = np.array(matrix).reshape(4, 4)
    yaw = float(np.degrees(np.arctan2(m[0, 2], m[0, 0])))
    sy = np.sqrt(m[0, 0]**2 + m[1, 0]**2)
    pitch = float(np.degrees(np.arctan2(-m[2, 0], sy)))
    roll = float(np.degrees(np.arctan2(m[2, 1], m[2, 2])))
    return yaw, pitch, roll


# ======================================================================
# Main
# ======================================================================

def main():
    print("=== Full Mouth Analysis Test ===")
    print("  L1: BiSeNet occlusion (every 5 frames)")
    print("  L2: XGBoost speaking (blendshape features)")
    print("  L3: Hysteresis state machine")
    print("  Press q to quit\n")

    # MediaPipe with blendshapes
    base_opts = mp_python.BaseOptions(model_asset_path="models/face_landmarker.task")
    lm_opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(lm_opts)

    # BiSeNet
    bis_opts = ort.SessionOptions()
    bis_opts.intra_op_num_threads = 4
    bisenet = ort.InferenceSession("models/speaking/resnet18.onnx",
                                    sess_options=bis_opts)
    bis_input = bisenet.get_inputs()[0].name

    # XGBoost
    predictor = SpeakingPredictor()
    hysteresis = HysteresisStateMachine()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    frame_idx = 0
    cached_seg = None
    cached_lip_ratio = 0.0
    cached_roi_box = (0, 0, 1, 1)
    cached_seg_mini = None
    bis_skip = 5

    TRACK_ID = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        display = frame.copy()
        yaw = pitch = roll = 0.0
        speaking_prob = 0.0
        lip_ratio = cached_lip_ratio
        final_status = "UNOBSERVABLE"
        occ_status = ""
        bs_dict = {}

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            pts = np.array([(l.x * w, l.y * h) for l in lm], dtype=np.float32)

            if result.facial_transformation_matrixes:
                yaw, pitch, roll = extract_yaw_pitch_roll(
                    result.facial_transformation_matrixes[0])

            if result.face_blendshapes:
                for bs in result.face_blendshapes[0]:
                    bs_dict[bs.category_name] = float(bs.score)

            # ── L1: BiSeNet occlusion ──
            run_bisenet = (frame_idx % bis_skip == 0) or cached_seg is None

            if abs(yaw) > 60:
                occ_status = "SELF_OCCLUDED"
                final_status = "SELF_OCCLUDED"
            else:
                if run_bisenet:
                    fx1, fy1, fx2, fy2 = face_bbox_from_landmarks(pts, h, w)
                    face_crop = frame[fy1:fy2, fx1:fx2]
                    if face_crop.size > 0:
                        cached_seg = bisenet_infer(bisenet, bis_input, face_crop)
                        mouth_pts = pts[ALL_MOUTH_INDICES]
                        cached_lip_ratio, cached_roi_box = compute_lip_ratio(
                            cached_seg, mouth_pts, (fx1, fy1),
                            (fy2 - fy1, fx2 - fx1))
                        cached_seg_mini = make_seg_colormap(cached_seg, cached_roi_box)

                lip_ratio = cached_lip_ratio
                if lip_ratio < 0.10:
                    occ_status = "OCCLUDED"
                    final_status = "OCCLUDED"
                else:
                    occ_status = "PARTIAL" if lip_ratio < 0.25 else "VISIBLE"

                    # ── L2: XGBoost speaking ──
                    prob = predictor.update(TRACK_ID, bs_dict, yaw, pitch, roll)
                    if prob is not None:
                        if occ_status == "PARTIAL":
                            prob *= 0.4
                        speaking_prob = prob
                        # ── L3: Hysteresis (yaw-adaptive) ──
                        final_status = hysteresis.update(
                            TRACK_ID, speaking_prob, yaw_abs=abs(yaw))
                    else:
                        final_status = "WARMING_UP"

            # Draw mouth ROI
            mouth_pts = pts[ALL_MOUTH_INDICES]
            mx1 = int(mouth_pts[:, 0].min() - 5)
            my1 = int(mouth_pts[:, 1].min() - 5)
            mx2 = int(mouth_pts[:, 0].max() + 5)
            my2 = int(mouth_pts[:, 1].max() + 5)
            cv2.rectangle(display, (mx1, my1), (mx2, my2), (255, 0, 255), 2)

        total_ms = (time.perf_counter() - t0) * 1000
        frame_idx += 1

        # ── Display ──
        sc = STATUS_COLORS.get(final_status, (255, 255, 255))

        # Final status (top-left, large)
        cv2.putText(display, final_status, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(display, final_status, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, sc, 3, cv2.LINE_AA)

        # Seg map (top-right)
        if cached_seg_mini is not None:
            display[10:170, w - 170:w - 10] = cached_seg_mini

        # YAW (right-center)
        yaw_abs = abs(yaw)
        if yaw_abs < 15:
            yc = (0, 255, 0)
        elif yaw_abs < 30:
            yc = (0, 255, 255)
        elif yaw_abs < 45:
            yc = (0, 165, 255)
        else:
            yc = (0, 0, 255)
        yaw_text = f"YAW {yaw:+.0f}"
        cv2.putText(display, yaw_text, (w - 200, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(display, yaw_text, (w - 200, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, yc, 2, cv2.LINE_AA)

        # Speaking prob bar (bottom-left)
        bar_x, bar_y = 10, h - 70
        bar_w, bar_h = 200, 20
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (60, 60, 60), -1)
        fill = int(bar_w * min(speaking_prob, 1.0))
        bar_color = (0, 255, 0) if speaking_prob > 0.55 else (0, 200, 200) if speaking_prob > 0.35 else (150, 150, 150)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
                      bar_color, -1)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (100, 100, 100), 1)
        cv2.putText(display, f"prob: {speaking_prob:.2f}", (bar_x + bar_w + 10, bar_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # Bottom info
        info = f"occ: {occ_status}  lip: {lip_ratio:.2f}  ms: {total_ms:.0f}"
        cv2.putText(display, info, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow("Full Mouth Analysis", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
