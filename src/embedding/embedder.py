# Copyright (c) 2024
# ArcFace Embedder using ONNX Runtime
"""
ArcFace 人脸嵌入提取器

模型说明:
- 输入: 已对齐的 112x112 BGR 人脸图像
- 输出: 512 维归一化嵌入向量
- 模型: buffalo_l 系列的 w600k_r50.onnx (InsightFace 预训练)

预处理:
1. 输入图像必须是 112x112 (与 alignment 模块输出一致)
2. BGR -> RGB 转换
3. 归一化: (pixel - 127.5) / 127.5, 范围 [-1, 1]
4. 转置为 NCHW 格式: [1, 3, 112, 112]

后处理:
1. L2 归一化嵌入向量
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger(__name__)


class ArcFaceEmbedder:
    """ArcFace 人脸嵌入提取器 (ONNX Runtime)"""

    # 模型默认参数
    DEFAULT_INPUT_SIZE = (112, 112)  # 必须与 alignment 输出一致
    EMBEDDING_DIM = 512

    def __init__(
        self,
        model_path: str = "models/w600k_r50.onnx",
        providers: Optional[List[str]] = None,
    ):
        """
        初始化 ArcFace Embedder

        Args:
            model_path: ONNX 模型路径
            providers: ONNX Runtime 执行提供者列表
                       默认优先使用 CUDA, 回退到 CPU
        """
        if ort is None:
            raise ImportError("请安装 onnxruntime: pip install onnxruntime")

        self.model_path = model_path
        self.input_size = self.DEFAULT_INPUT_SIZE

        # 检查模型文件
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ArcFace 模型文件不存在: {model_path}\n"
                f"请运行: python scripts/download_arcface.py"
            )

        # 设置执行提供者
        if providers is None:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        # 创建推理会话 (限制线程防止与 detector 竞争 CPU)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )

        # 获取输入输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # 获取输入形状
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) == 4:
            _, _, h, w = input_shape
            if isinstance(h, int) and isinstance(w, int):
                self.input_size = (w, h)

    def preprocess(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        预处理对齐后的人脸图像

        Args:
            aligned_face: 112x112 BGR 图像 (uint8)

        Returns:
            预处理后的张量 [1, 3, 112, 112] (float32)
        """
        if aligned_face is None:
            raise ValueError("输入图像为空")

        # 验证尺寸
        h, w = aligned_face.shape[:2]
        if (w, h) != self.input_size:
            # 如果尺寸不对，resize (不建议，应保证 alignment 输出正确)
            aligned_face = cv2.resize(aligned_face, self.input_size)

        # BGR -> RGB
        img = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

        # 归一化到 [-1, 1]
        img = (img.astype(np.float32) - 127.5) / 127.5

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        # 添加 batch 维度
        img = np.expand_dims(img, axis=0)

        return img

    def extract(self, aligned_face: np.ndarray) -> Optional[np.ndarray]:
        """
        从对齐的人脸图像提取嵌入向量

        Args:
            aligned_face: 112x112 BGR 图像 (uint8)

        Returns:
            512 维归一化嵌入向量 (float32), 失败返回 None
        """
        try:
            # 预处理
            input_tensor = self.preprocess(aligned_face)

            # 推理
            outputs = self.session.run(
                [self.output_name], {self.input_name: input_tensor}
            )
            embedding = outputs[0][0]  # [512]

            # L2 归一化
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding.astype(np.float32)

        except Exception as e:
            logger.exception("[Embedder] 嵌入提取失败: %s", e)
            return None

    def extract_batch(
        self, aligned_faces: List[np.ndarray]
    ) -> List[Optional[np.ndarray]]:
        """
        批量提取嵌入向量

        Args:
            aligned_faces: 对齐人脸图像列表

        Returns:
            嵌入向量列表
        """
        return [self.extract(face) for face in aligned_faces]

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        计算两个嵌入向量的余弦相似度

        Args:
            emb1, emb2: 归一化的嵌入向量

        Returns:
            相似度 [-1, 1], 越高越相似
        """
        # 由于已经 L2 归一化，直接点积即可
        return float(np.dot(emb1, emb2))

    @staticmethod
    def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        计算两个嵌入向量的欧氏距离

        Args:
            emb1, emb2: 嵌入向量

        Returns:
            欧氏距离，越小越相似
        """
        return float(np.linalg.norm(emb1 - emb2))


def download_arcface_model(model_dir: str = "models") -> str:
    """
    下载 ArcFace ONNX 模型

    Returns:
        模型文件路径
    """
    import urllib.request

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "w600k_r50.onnx")

    if os.path.exists(model_path):
        logger.info("[Download] 模型已存在: %s", model_path)
        return model_path

    # InsightFace buffalo_l 模型 (从 GitHub release 下载)
    url = (
        "https://github.com/deepinsight/insightface/releases/download/"
        "v0.7/buffalo_l.zip"
    )

    logger.info("[Download] 正在下载 ArcFace 模型...")
    logger.info("[Download] URL: %s", url)

    # 下载并解压
    zip_path = os.path.join(model_dir, "buffalo_l.zip")
    urllib.request.urlretrieve(url, zip_path)

    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        # 只提取 w600k_r50.onnx
        for name in zf.namelist():
            if name.endswith("w600k_r50.onnx"):
                data = zf.read(name)
                with open(model_path, "wb") as f:
                    f.write(data)
                logger.info("[Download] 已保存: %s", model_path)
                break

    # 清理 zip 文件
    os.remove(zip_path)

    return model_path


if __name__ == "__main__":
    # 测试代码

    # 下载模型
    model_path = download_arcface_model()

    # 创建 embedder
    embedder = ArcFaceEmbedder(model_path)
    logger.info("[Test] Embedder 初始化成功")
    logger.info("[Test] 输入尺寸: %s", embedder.input_size)
    logger.info("[Test] 嵌入维度: %s", embedder.EMBEDDING_DIM)

    # 测试随机图像
    fake_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    emb = embedder.extract(fake_face)
    if emb is not None:
        logger.info("[Test] 嵌入向量形状: %s", emb.shape)
        logger.info("[Test] 嵌入向量范数: %.4f", np.linalg.norm(emb))
    else:
        logger.warning("[Test] 嵌入提取失败")
