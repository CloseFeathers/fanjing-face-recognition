# Copyright (c) 2024
# ArcFace Embedder using ONNX Runtime
"""
ArcFace Face Embedding Extractor

Model description:
- Input: Aligned 112x112 BGR face image
- Output: 512-dimensional normalized embedding vector
- Model: w600k_r50.onnx from buffalo_l series (InsightFace pretrained)

Preprocessing:
1. Input image must be 112x112 (consistent with alignment module output)
2. BGR -> RGB conversion
3. Normalization: (pixel - 127.5) / 127.5, range [-1, 1]
4. Transpose to NCHW format: [1, 3, 112, 112]

Postprocessing:
1. L2 normalize embedding vector
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
    """ArcFace face embedding extractor (ONNX Runtime)."""

    # Model default parameters
    DEFAULT_INPUT_SIZE = (112, 112)  # Must match alignment output
    EMBEDDING_DIM = 512

    def __init__(
        self,
        model_path: str = "models/w600k_r50.onnx",
        providers: Optional[List[str]] = None,
    ):
        """
        Initialize ArcFace Embedder

        Args:
            model_path: ONNX model path
            providers: ONNX Runtime execution provider list
                       Default: prefer CUDA, fallback to CPU
        """
        if ort is None:
            raise ImportError("Please install onnxruntime: pip install onnxruntime")

        self.model_path = model_path
        self.input_size = self.DEFAULT_INPUT_SIZE

        # Check model file
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ArcFace model file not found: {model_path}\n"
                f"Please run: python scripts/download_arcface.py"
            )

        # Set execution providers
        if providers is None:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        # Create inference session (limit threads to avoid CPU contention with detector)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Get input shape
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) == 4:
            _, _, h, w = input_shape
            if isinstance(h, int) and isinstance(w, int):
                self.input_size = (w, h)

    def preprocess(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Preprocess aligned face image

        Args:
            aligned_face: 112x112 BGR image (uint8)

        Returns:
            Preprocessed tensor [1, 3, 112, 112] (float32)
        """
        if aligned_face is None:
            raise ValueError("Input image is empty")

        # Validate size
        h, w = aligned_face.shape[:2]
        if (w, h) != self.input_size:
            # Resize if size mismatch (not recommended, alignment output should be correct)
            aligned_face = cv2.resize(aligned_face, self.input_size)

        # BGR -> RGB
        img = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

        # Normalize to [-1, 1]
        img = (img.astype(np.float32) - 127.5) / 127.5

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img

    def extract(self, aligned_face: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract embedding vector from aligned face image

        Args:
            aligned_face: 112x112 BGR image (uint8)

        Returns:
            512-dimensional normalized embedding vector (float32), None on failure
        """
        try:
            # Preprocess
            input_tensor = self.preprocess(aligned_face)

            # Inference
            outputs = self.session.run(
                [self.output_name], {self.input_name: input_tensor}
            )
            embedding = outputs[0][0]  # [512]

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding.astype(np.float32)

        except Exception as e:
            logger.exception("[Embedder] Embedding extraction failed: %s", e)
            return None

    def extract_batch(
        self, aligned_faces: List[np.ndarray]
    ) -> List[Optional[np.ndarray]]:
        """
        Batch extract embedding vectors

        Args:
            aligned_faces: List of aligned face images

        Returns:
            List of embedding vectors
        """
        return [self.extract(face) for face in aligned_faces]

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embedding vectors

        Args:
            emb1, emb2: Normalized embedding vectors

        Returns:
            Similarity [-1, 1], higher is more similar
        """
        # Since already L2 normalized, dot product is sufficient
        return float(np.dot(emb1, emb2))

    @staticmethod
    def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two embedding vectors

        Args:
            emb1, emb2: Embedding vectors

        Returns:
            Euclidean distance, smaller is more similar
        """
        return float(np.linalg.norm(emb1 - emb2))


def download_arcface_model(model_dir: str = "models") -> str:
    """
    Download ArcFace ONNX model

    Returns:
        Model file path
    """
    import urllib.request

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "w600k_r50.onnx")

    if os.path.exists(model_path):
        logger.info("[Download] Model already exists: %s", model_path)
        return model_path

    # InsightFace buffalo_l model (download from GitHub release)
    url = (
        "https://github.com/deepinsight/insightface/releases/download/"
        "v0.7/buffalo_l.zip"
    )

    logger.info("[Download] Downloading ArcFace model...")
    logger.info("[Download] URL: %s", url)

    # Download and extract
    zip_path = os.path.join(model_dir, "buffalo_l.zip")
    urllib.request.urlretrieve(url, zip_path)

    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Only extract w600k_r50.onnx
        for name in zf.namelist():
            if name.endswith("w600k_r50.onnx"):
                data = zf.read(name)
                with open(model_path, "wb") as f:
                    f.write(data)
                logger.info("[Download] Saved: %s", model_path)
                break

    # Clean up zip file
    os.remove(zip_path)

    return model_path


if __name__ == "__main__":
    # Test code

    # Download model
    model_path = download_arcface_model()

    # Create embedder
    embedder = ArcFaceEmbedder(model_path)
    logger.info("[Test] Embedder initialized successfully")
    logger.info("[Test] Input size: %s", embedder.input_size)
    logger.info("[Test] Embedding dimension: %s", embedder.EMBEDDING_DIM)

    # Test with random image
    fake_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    emb = embedder.extract(fake_face)
    if emb is not None:
        logger.info("[Test] Embedding vector shape: %s", emb.shape)
        logger.info("[Test] Embedding vector norm: %.4f", np.linalg.norm(emb))
    else:
        logger.warning("[Test] Embedding extraction failed")
