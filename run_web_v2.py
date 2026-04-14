#!/usr/bin/env python3
"""
心流元素人脸识别系统 —— 干净版 (v2)

端口: 5001 (与旧版 5000 完全隔离)

用法:
  python run_web_v2.py                     # http://localhost:5001
  python run_web_v2.py --port 8080         # 自定义端口
  python run_web_v2.py --host 0.0.0.0      # 局域网访问
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import webbrowser
import threading

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8")
            except Exception:
                pass

from src.web.server import app, ensure_detector


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Face Recognition System v2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--model", default="models/det_10g.onnx")
    args = parser.parse_args()

    print("[v2] 预加载检测模型...")
    ensure_detector(model_path=args.model)

    url = f"http://{args.host}:{args.port}"
    print(f"[v2] 启动服务: {url}")
    print("[v2] 按 Ctrl+C 停止")

    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
