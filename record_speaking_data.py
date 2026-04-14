"""
说话检测数据录制工具 v2 — 按场景独立录制

每个场景单独一个视频文件, 可以随时重录任何场景。

操作:
  启动后显示场景菜单, 输入编号录制单个场景, 输入 'all' 录制全部。
  录制中:
    空格键 按住 = 正在说话
    空格键 松开 = 不在说话
    Enter       = 提前结束当前场景
    ESC         = 中止并询问是否保留

输出目录: data/recordings/session_YYYYMMDD_HHMMSS/
  01_pos_frontal_normal/
    video.mp4, mouth_roi.mp4, features.csv, events.jsonl
  02_pos_frontal_fast/
    ...
  metadata.json

依赖: pip install keyboard Pillow
"""

import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

try:
    import keyboard
except ImportError:
    print("ERROR: pip install keyboard")
    sys.exit(1)
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("ERROR: pip install Pillow")
    sys.exit(1)

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src.speaking.mesh_detector import (
    LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER,
    UPPER_LIP_OUTER, LOWER_LIP_OUTER,
)

# ======================================================================
# 场景定义
# ======================================================================

SCENARIOS = [
    # 1-6: 正面说话 (已录)
    {"id": "pos_frontal_normal", "name": "正脸 · 正常说话", "dur": 60,
     "cat": "SPEAKING",
     "inst": "请面对镜头自然地连续说话。\n全程按住空格键。"},
    {"id": "pos_frontal_fast", "name": "正脸 · 快速说话", "dur": 30,
     "cat": "SPEAKING",
     "inst": "请尽量快速地说话。\n全程按住空格。"},
    {"id": "pos_frontal_slow", "name": "正脸 · 慢速说话", "dur": 30,
     "cat": "SPEAKING",
     "inst": "请放慢语速说话，一字一句清晰。\n全程按住空格。"},
    {"id": "pos_frontal_pause", "name": "正脸 · 短句+停顿", "dur": 40,
     "cat": "SPEAKING",
     "inst": "请说短句，每句之间停顿1-2秒。\n★ 说话按空格，停顿松开。标注精度很重要！"},
    {"id": "pos_frontal_subtle", "name": "正脸 · 低幅度说话", "dur": 30,
     "cat": "SPEAKING",
     "inst": "请不怎么张嘴地说话（含糊、小声）。\n全程按住空格。"},
    {"id": "pos_frontal_smile", "name": "正脸 · 微笑说话", "dur": 30,
     "cat": "SPEAKING",
     "inst": "请保持微笑的状态说话。\n全程按住空格。"},

    # 7-12: 侧面说话 (已录)
    {"id": "pos_left15", "name": "左转15° · 说话", "dur": 30,
     "cat": "SPEAKING",
     "inst": "请向左转约15度，连续说话。\n全程按住空格。参考 yaw（-10~-20）。"},
    {"id": "pos_right15", "name": "右转15° · 说话", "dur": 30,
     "cat": "SPEAKING",
     "inst": "请向右转约15度，连续说话。\n全程按住空格。参考 yaw（+10~+20）。"},
    {"id": "pos_left30", "name": "左转30° · 说话", "dur": 30,
     "cat": "SPEAKING",
     "inst": "请向左转约30度，连续说话。\n全程按住空格。参考 yaw（-25~-35）。"},
    {"id": "pos_right30", "name": "右转30° · 说话", "dur": 30,
     "cat": "SPEAKING",
     "inst": "请向右转约30度，连续说话。\n全程按住空格。参考 yaw（+25~+35）。"},
    {"id": "pos_headmove", "name": "说话+自然摆头", "dur": 30,
     "cat": "SPEAKING",
     "inst": "请一边说话一边自然地左右转头。\n全程按住空格。模拟正常交流。"},
    {"id": "pos_headnod", "name": "说话+点头", "dur": 30,
     "cat": "SPEAKING",
     "inst": "请一边说话一边点头（上下）。\n全程按住空格。"},

    # 13-18: 正面不说话 (已录)
    {"id": "neg_still", "name": "正脸 · 闭嘴静止", "dur": 25,
     "cat": "NOT_SPEAKING",
     "inst": "请闭嘴保持静止，不要说话。\n【不要按空格键】"},
    {"id": "neg_open_still", "name": "正脸 · 微张嘴静止", "dur": 20,
     "cat": "NOT_SPEAKING",
     "inst": "请微微张开嘴，但不说话不动。\n【不要按空格键】★ Hard Negative"},
    {"id": "neg_breathe", "name": "正脸 · 张嘴呼吸", "dur": 20,
     "cat": "NOT_SPEAKING",
     "inst": "请张嘴呼吸，但不说话。\n【不要按空格键】"},
    {"id": "neg_left20", "name": "左转20° · 静止", "dur": 15,
     "cat": "NOT_SPEAKING",
     "inst": "请向左转约20度，保持静止不说话。\n【不要按空格键】"},
    {"id": "neg_right20", "name": "右转20° · 静止", "dur": 15,
     "cat": "NOT_SPEAKING",
     "inst": "请向右转约20度，保持静止不说话。\n【不要按空格键】"},
    {"id": "neg_smile", "name": "微笑 · 不说话", "dur": 15,
     "cat": "NOT_SPEAKING",
     "inst": "请微笑，但不说话不出声。\n【不要按空格键】"},

    # 19: 占位 (保持编号对齐)
    {"id": "_skip_19", "name": "(已跳过)", "dur": 0,
     "cat": "SKIP", "inst": ""},

    # 20-22: 不说话 (已录)
    {"id": "neg_yawn", "name": "打哈欠", "dur": 15,
     "cat": "NOT_SPEAKING",
     "inst": "请假装打哈欠几次。\n【不要按空格键】★ Hard Negative"},
    {"id": "neg_chew", "name": "咀嚼 · 不说话", "dur": 20,
     "cat": "NOT_SPEAKING",
     "inst": "请做咀嚼动作，但不说话。\n【不要按空格键】★ Hard Negative"},
    {"id": "neg_headmove", "name": "转头点头 · 不说话", "dur": 20,
     "cat": "NOT_SPEAKING",
     "inst": "请自由转头、点头，但不说话。\n【不要按空格键】"},

    # === 23-30: 新增场景（侧面补充） ===
    {"id": "pos_left45", "name": "左转45° · 说话", "dur": 30,
     "cat": "SPEAKING",
     "inst": "请向左转约45度（大幅侧转），连续说话。\n全程按住空格。参考 yaw（-40~-50）。"},
    {"id": "pos_right45", "name": "右转45° · 说话", "dur": 30,
     "cat": "SPEAKING",
     "inst": "请向右转约45度（大幅侧转），连续说话。\n全程按住空格。参考 yaw（+40~+50）。"},
    {"id": "neg_left15", "name": "左转15° · 静止", "dur": 15,
     "cat": "NOT_SPEAKING",
     "inst": "请向左转约15度，保持静止不说话。\n【不要按空格键】参考 yaw（-10~-20）。"},
    {"id": "neg_right15", "name": "右转15° · 静止", "dur": 15,
     "cat": "NOT_SPEAKING",
     "inst": "请向右转约15度，保持静止不说话。\n【不要按空格键】参考 yaw（+10~+20）。"},
    {"id": "neg_left30", "name": "左转30° · 静止", "dur": 15,
     "cat": "NOT_SPEAKING",
     "inst": "请向左转约30度，保持静止不说话。\n【不要按空格键】参考 yaw（-25~-35）。"},
    {"id": "neg_right30", "name": "右转30° · 静止", "dur": 15,
     "cat": "NOT_SPEAKING",
     "inst": "请向右转约30度，保持静止不说话。\n【不要按空格键】参考 yaw（+25~+35）。"},
    {"id": "neg_left40", "name": "左转40° · 静止", "dur": 15,
     "cat": "NOT_SPEAKING",
     "inst": "请向左转约40度，保持静止不说话。\n【不要按空格键】参考 yaw（-35~-45）。"},
    {"id": "neg_right40", "name": "右转40° · 静止", "dur": 15,
     "cat": "NOT_SPEAKING",
     "inst": "请向右转约40度，保持静止不说话。\n【不要按空格键】参考 yaw（+35~+45）。"},
]


# ======================================================================
# Chinese text
# ======================================================================

_font_cache: dict = {}

def _get_font(size):
    if size not in _font_cache:
        import sys
        candidates = []
        if sys.platform == "win32":
            candidates = [
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/msyhbd.ttc",
                "C:/Windows/Fonts/simhei.ttf",
            ]
        elif sys.platform == "darwin":
            candidates = [
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/STHeiti Light.ttc",
                "/Library/Fonts/Arial Unicode.ttf",
            ]
        else:
            candidates = [
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            ]
        for p in candidates:
            if os.path.exists(p):
                _font_cache[size] = ImageFont.truetype(p, size)
                return _font_cache[size]
        _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]

def draw_cn(img, text, pos, size=22, color=(255, 255, 255)):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(pos, text, font=_get_font(size), fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ======================================================================
# Helpers
# ======================================================================

def extract_yaw_pitch_roll(matrix):
    m = np.array(matrix).reshape(4, 4)
    yaw = float(np.degrees(np.arctan2(m[0, 2], m[0, 0])))
    sy = np.sqrt(m[0, 0]**2 + m[1, 0]**2)
    pitch = float(np.degrees(np.arctan2(-m[2, 0], sy)))
    roll = float(np.degrees(np.arctan2(m[2, 1], m[2, 2])))
    return yaw, pitch, roll

def extract_mouth_roi(frame, pts, w, h):
    left = pts[LEFT_MOUTH_CORNER]
    right = pts[RIGHT_MOUTH_CORNER]
    top = pts[UPPER_LIP_OUTER]
    bottom = pts[LOWER_LIP_OUTER]
    cx = (left[0] + right[0]) / 2
    cy = (top[1] + bottom[1]) / 2
    mw = abs(right[0] - left[0]) * 1.5
    mh = abs(bottom[1] - top[1]) * 2.0
    sz = max(mw, mh)
    x1 = int(max(0, cx - sz / 2))
    y1 = int(max(0, cy - sz / 2))
    x2 = int(min(w, cx + sz / 2))
    y2 = int(min(h, cy + sz / 2))
    return frame[y1:y2, x1:x2], (x2 - x1), (y2 - y1)

def run_vsdlm(sess, input_name, mouth_crop):
    if mouth_crop.size == 0:
        return 0.0
    resized = cv2.resize(mouth_crop, (48, 30))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)
    out = sess.run(None, {input_name: blob})
    return float(out[0][0])


# ======================================================================
# Record one scenario
# ======================================================================

def record_scenario(sc_idx, sc, cap, landmarker, vsdlm, vsdlm_input_name,
                    out_dir, frame_w, frame_h, total_scenarios):
    sc_id = sc["id"]
    sc_name = sc["name"]
    sc_dur = sc["dur"]
    sc_cat = sc["cat"]
    inst_lines = sc["inst"].split("\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sc_dir = out_dir / f"{sc_idx + 1:02d}_{sc_id}_{ts}"
    sc_dir.mkdir(parents=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_w = cv2.VideoWriter(str(sc_dir / "video.mp4"), fourcc, 30.0,
                            (frame_w, frame_h))
    roi_w = cv2.VideoWriter(str(sc_dir / "mouth_roi.mp4"), fourcc, 30.0,
                            (128, 80))

    bs_names = []
    csv_file = open(sc_dir / "features.csv", "w", newline="", encoding="utf-8")
    csv_writer = None
    events_file = open(sc_dir / "events.jsonl", "w", encoding="utf-8")

    t0 = time.perf_counter()
    frame_id = 0
    prev_key = False
    fps_est = 30.0
    t_prev = t0

    def ts_ms():
        return (time.perf_counter() - t0) * 1000

    def log_event(etype, **kw):
        ev = {"type": etype, "timestamp_ms": round(ts_ms(), 1),
              "frame_id": frame_id, **kw}
        events_file.write(json.dumps(ev, ensure_ascii=False) + "\n")

    # ── 5-second countdown with instructions on screen ──
    cat_labels = {"SPEAKING": "▶ 说话场景 — 说话时按住空格",
                  "NOT_SPEAKING": "■ 静默场景 — 不要按空格",
                  "IGNORE": "✕ 忽略场景", "MIXED": "◆ 混合场景 — 按需切换"}
    cat_colors = {"SPEAKING": (0, 255, 0), "NOT_SPEAKING": (200, 200, 200),
                  "IGNORE": (0, 140, 255), "MIXED": (0, 255, 255)}

    for cd in range(5, 0, -1):
        ret, frame = cap.read()
        if ret:
            disp = frame.copy()
            overlay = disp.copy()
            cv2.rectangle(overlay, (0, 0), (frame_w, frame_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, disp, 0.45, 0, disp)
            disp = draw_cn(disp,
                           f"场景 {sc_idx+1}/{total_scenarios}: {sc_name}",
                           (20, 20), size=30, color=(255, 255, 0))
            disp = draw_cn(disp,
                           f"{cat_labels.get(sc_cat, '')}    时长: {sc_dur}秒",
                           (20, 65), size=20,
                           color=cat_colors.get(sc_cat, (255, 255, 255)))
            y_inst = 110
            for il in inst_lines:
                c = (255, 180, 80) if "★" in il else (255, 255, 255)
                if "不要按空格" in il:
                    c = (100, 180, 255)
                disp = draw_cn(disp, il, (30, y_inst), size=22, color=c)
                y_inst += 32
            disp = draw_cn(disp, str(cd), (frame_w - 100, frame_h - 100),
                           size=72, color=(0, 255, 255))
            cv2.imshow("Recording", disp)
        cv2.waitKey(1000)

    # ── Record ──
    log_event("scenario_start", scenario_id=sc_id, scenario_name=sc_name,
              scenario_cat=sc_cat)
    sc_start = time.perf_counter()
    aborted = False

    while True:
        elapsed = time.perf_counter() - sc_start
        remaining = sc_dur - elapsed
        if remaining <= 0:
            break

        ret, frame = cap.read()
        if not ret:
            break

        t_now = time.perf_counter()
        dt = t_now - t_prev
        if dt > 0:
            fps_est = fps_est * 0.9 + (1.0 / dt) * 0.1
        t_prev = t_now
        cur_ts = ts_ms()

        key_pressed = keyboard.is_pressed("space")
        if key_pressed and not prev_key:
            log_event("key_down")
        elif not key_pressed and prev_key:
            log_event("key_up")
        prev_key = key_pressed

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)

        mesh_ok = 0
        yaw = pitch = roll = 0.0
        vsdlm_prob = 0.0
        roi_ww = roi_hh = roi_area = 0
        blur_score = 0.0
        bs_dict = {}
        mouth_roi_frame = np.zeros((80, 128, 3), dtype=np.uint8)

        if result.face_landmarks:
            mesh_ok = 1
            lm = result.face_landmarks[0]
            pts = np.array([(l.x * w, l.y * h) for l in lm], dtype=np.float32)

            if result.facial_transformation_matrixes:
                yaw, pitch, roll = extract_yaw_pitch_roll(
                    result.facial_transformation_matrixes[0])

            if result.face_blendshapes:
                for bs in result.face_blendshapes[0]:
                    bs_dict[bs.category_name] = float(bs.score)
                if not bs_names:
                    bs_names = [bs.category_name
                                for bs in result.face_blendshapes[0]]

            mouth_crop, roi_ww, roi_hh = extract_mouth_roi(frame, pts, w, h)
            roi_area = roi_ww * roi_hh
            if mouth_crop.size > 0:
                gray_roi = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2GRAY)
                blur_score = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
                vsdlm_prob = run_vsdlm(vsdlm, vsdlm_input_name, mouth_crop)
                if mouth_crop.shape[0] > 0 and mouth_crop.shape[1] > 0:
                    mouth_roi_frame = cv2.resize(mouth_crop, (128, 80))

        if csv_writer is None and bs_names:
            fixed = ["frame_id", "timestamp_ms", "fps_estimate",
                     "scenario_id", "scenario_name", "scenario_cat",
                     "key_pressed", "mesh_ok", "yaw", "pitch", "roll",
                     "vsdlm_prob_open", "mouth_roi_w", "mouth_roi_h",
                     "mouth_roi_area", "mouth_blur_score"]
            fields = fixed + [f"bs_{n}" for n in bs_names]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fields)
            csv_writer.writeheader()

        vid_w.write(frame)
        roi_w.write(mouth_roi_frame)

        if csv_writer is not None:
            row = {"frame_id": frame_id, "timestamp_ms": f"{cur_ts:.1f}",
                   "fps_estimate": f"{fps_est:.1f}", "scenario_id": sc_id,
                   "scenario_name": sc_name, "scenario_cat": sc_cat,
                   "key_pressed": 1 if key_pressed else 0, "mesh_ok": mesh_ok,
                   "yaw": f"{yaw:.2f}", "pitch": f"{pitch:.2f}",
                   "roll": f"{roll:.2f}", "vsdlm_prob_open": f"{vsdlm_prob:.4f}",
                   "mouth_roi_w": roi_ww, "mouth_roi_h": roi_hh,
                   "mouth_roi_area": roi_area,
                   "mouth_blur_score": f"{blur_score:.1f}"}
            for n in bs_names:
                row[f"bs_{n}"] = f"{bs_dict.get(n, 0.0):.4f}"
            csv_writer.writerow(row)

        frame_id += 1

        # ── Display ──
        disp = frame.copy()
        cv2.rectangle(disp, (0, 0), (frame_w, 65), (30, 30, 30), -1)
        cv2.putText(disp, f"[{sc_idx+1}/{total_scenarios}]", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1,
                    cv2.LINE_AA)
        disp = draw_cn(disp, sc_name, (90, 2), size=20, color=(255, 255, 0))
        cv2.putText(disp, f"{int(remaining)}s", (frame_w - 70, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                    cv2.LINE_AA)
        pct = min(elapsed / sc_dur, 1.0)
        bw = frame_w - 20
        cv2.rectangle(disp, (10, 35), (10 + int(bw * pct), 48),
                      (0, 200, 200), -1)
        cv2.rectangle(disp, (10, 35), (10 + bw, 48), (80, 80, 80), 1)
        short_inst = inst_lines[0][:30] + ("..." if len(inst_lines[0]) > 30 else "")
        disp = draw_cn(disp, short_inst, (10, 50), size=14, color=(200, 200, 180))

        if key_pressed:
            cv2.rectangle(disp, (0, frame_h - 45), (frame_w, frame_h),
                          (0, 100, 0), -1)
            cv2.putText(disp, ">>> SPEAKING [SPACE] <<<",
                        (frame_w // 2 - 160, frame_h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,
                        cv2.LINE_AA)
        else:
            cv2.rectangle(disp, (0, frame_h - 45), (frame_w, frame_h),
                          (50, 50, 50), -1)
            cv2.putText(disp, "SILENT", (frame_w // 2 - 45, frame_h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 120, 120), 2,
                        cv2.LINE_AA)

        jaw = bs_dict.get("jawOpen", 0.0)

        # YAW: big, right side, black outline + colored text
        yaw_abs = abs(yaw)
        if yaw_abs < 15:
            yaw_color = (0, 255, 0)
        elif yaw_abs < 30:
            yaw_color = (0, 255, 255)
        elif yaw_abs < 45:
            yaw_color = (0, 165, 255)
        else:
            yaw_color = (0, 0, 255)
        yaw_text = f"YAW {yaw:+.0f}"
        yaw_x = frame_w - 220
        yaw_y = frame_h // 2 + 10
        cv2.putText(disp, yaw_text, (yaw_x, yaw_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(disp, yaw_text, (yaw_x, yaw_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, yaw_color, 3, cv2.LINE_AA)

        info = f"fps:{fps_est:.0f}  jaw:{jaw:.2f}  vsdlm:{vsdlm_prob:.2f}"
        cv2.putText(disp, info, (10, frame_h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
                    cv2.LINE_AA)

        if mouth_roi_frame is not None:
            rh, rw = mouth_roi_frame.shape[:2]
            if rh > 0 and rw > 0:
                disp[frame_h - 45 - rh - 5:frame_h - 45 - 5,
                     frame_w - rw - 10:frame_w - 10] = mouth_roi_frame

        cv2.imshow("Recording", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            aborted = True
            break
        elif key == 13:
            break

    log_event("scenario_end", scenario_id=sc_id)

    vid_w.release()
    roi_w.release()
    csv_file.close()
    events_file.close()

    if aborted:
        ans = input(f"\n  场景 {sc_name} 被中止。保留数据？(y/n) [y]: ").strip().lower()
        if ans == "n":
            shutil.rmtree(sc_dir, ignore_errors=True)
            print(f"  已删除: {sc_dir.name}")
            return "aborted_deleted"
        return "aborted_kept"

    print(f"  ✓ 场景 {sc_idx+1} 完成: {sc_name} ({frame_id} 帧)")
    return "done"


# ======================================================================
# Main
# ======================================================================

def main():
    rec_root = Path("data/recordings")
    rec_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  说话检测数据录制工具 v2")
    print(f"{'=' * 60}")
    print()

    out_dir = rec_root
    name = ""

    # ── Show menu ──
    print("  场景列表:")
    for i, sc in enumerate(SCENARIOS):
        if sc["cat"] == "SKIP":
            continue
        cat_mark = {"SPEAKING": "▶", "NOT_SPEAKING": "■"}.get(sc["cat"], " ")
        print(f"    {i + 1:2d}. {cat_mark} {sc['name']:20s}  ({sc['dur']}s)  [{sc['cat']}]")

    total_dur = sum(s["dur"] for s in SCENARIOS)
    print(f"\n  共 {len(SCENARIOS)} 个场景, 总时长 {total_dur}s ({total_dur / 60:.1f}分钟)")
    print()
    print("  输入方式:")
    print("    数字      → 录制单个场景 (如 '4')")
    print("    范围      → 录制范围 (如 '7-12')")
    print("    多个      → 逗号分隔 (如 '1,4,7')")
    print("    all       → 录制全部")
    print("    q         → 退出")
    print()

    choice = input("  请选择要录制的场景: ").strip().lower()
    if choice == "q":
        return

    indices = []
    if choice == "all":
        indices = list(range(len(SCENARIOS)))
    else:
        try:
            for part in choice.split(","):
                part = part.strip()
                if "-" in part:
                    a, b = part.split("-", 1)
                    indices.extend(range(int(a) - 1, int(b)))
                else:
                    indices.append(int(part) - 1)
        except ValueError:
            print("  输入格式错误。")
            return

    indices = [i for i in indices
               if 0 <= i < len(SCENARIOS) and SCENARIOS[i]["cat"] != "SKIP"]
    if not indices:
        print("  没有有效的场景编号。")
        return

    print(f"\n  将录制 {len(indices)} 个场景:")
    for i in indices:
        print(f"    {i + 1}. {SCENARIOS[i]['name']}")

    print(f"\n  输出目录: {out_dir}")
    print(f"\n  操作: 空格=说话  Enter=提前结束  ESC=中止")
    print(f"\n  按 Enter 开始...")
    input()

    # ── Load models ──
    print("  正在加载模型...")
    base_options = mp_python.BaseOptions(
        model_asset_path="models/face_landmarker.task")
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 2
    vsdlm = ort.InferenceSession("models/speaking/vsdlm_s.onnx",
                                  sess_options=sess_opts)
    vsdlm_input_name = vsdlm.get_inputs()[0].name

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ERROR: 无法打开摄像头")
        return
    ret, test_frame = cap.read()
    if not ret:
        print("  ERROR: 无法读取摄像头")
        return
    frame_h, frame_w = test_frame.shape[:2]
    print("  模型加载完成！\n")

    # ── Record each selected scenario ──
    for idx in indices:
        sc = SCENARIOS[idx]
        result = record_scenario(idx, sc, cap, landmarker, vsdlm,
                                 vsdlm_input_name, out_dir, frame_w,
                                 frame_h, len(SCENARIOS))
        if result == "aborted_deleted":
            ans = input("  继续录制下一个场景？(y/n) [y]: ").strip().lower()
            if ans == "n":
                break
        elif result == "aborted_kept":
            ans = input("  继续录制下一个场景？(y/n) [y]: ").strip().lower()
            if ans == "n":
                break

    # ── Cleanup ──
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    # ── Summary ──

    print(f"\n{'=' * 60}")
    print(f"  录制完成！")
    print(f"  输出目录: {out_dir}")
    print(f"  已录场景:")
    for d in sorted(out_dir.iterdir()):
        if d.is_dir():
            csv_p = d / "features.csv"
            if csv_p.exists():
                lines = sum(1 for _ in open(csv_p, encoding="utf-8")) - 1
                print(f"    {d.name}: {lines} 帧")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
