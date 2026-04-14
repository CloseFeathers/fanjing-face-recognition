"""
Face Aligner —— 基于 5 点关键点 (kps5) 的仿射对齐。

使用 Umeyama 算法估计相似变换 (旋转+缩放+平移)，
将检测到的 kps5 映射到 ArcFace 标准参考点，
并将原图 warpAffine 到固定 112x112 输出。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

# ArcFace 标准参考点 (112x112 坐标系)
ARCFACE_REF_112 = np.array([
    [38.2946, 51.6963],   # 左眼中心
    [73.5318, 51.5014],   # 右眼中心
    [56.0252, 71.7366],   # 鼻尖
    [41.5493, 92.3655],   # 左嘴角
    [70.7299, 92.2041],   # 右嘴角
], dtype=np.float32)


class FaceAligner:
    """5 点仿射对齐器。"""

    def __init__(self, output_size: Tuple[int, int] = (112, 112)) -> None:
        self.output_size = output_size
        scale = output_size[0] / 112.0
        self.ref_pts = ARCFACE_REF_112.copy() * scale

    def align(
        self,
        image: np.ndarray,
        kps5: List[List[float]],
    ) -> Optional[np.ndarray]:
        """对齐并裁剪人脸。

        Args:
            image: BGR 原图
            kps5:  5x2 关键点 [[x,y], ...]
        Returns:
            112x112 BGR 对齐人脸, 失败返回 None
        """
        src_pts = np.array(kps5, dtype=np.float32)
        if src_pts.shape != (5, 2):
            return None

        M = _umeyama(src_pts, self.ref_pts, estimate_scale=True)
        if M is None:
            return None

        aligned = cv2.warpAffine(
            image,
            M[:2],                          # 取前两行 (2x3)
            self.output_size,
            flags=cv2.INTER_LINEAR,
            borderValue=(0, 0, 0),
        )
        return aligned


# ======================================================================
# Umeyama 相似变换估计
# ======================================================================

def _umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    estimate_scale: bool = True,
) -> Optional[np.ndarray]:
    """Umeyama 算法: 从 src→dst 估计最优相似变换 (最小二乘)。

    Returns:
        3x3 齐次变换矩阵, 失败返回 None
    """
    num = src.shape[0]
    dim = src.shape[1]

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = dst_demean.T @ src_demean / num

    U, S, Vt = np.linalg.svd(A)

    d = np.ones(dim, dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)

    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return None

    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(Vt) > 0:
            T[:dim, :dim] = U @ Vt
        else:
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ Vt
    else:
        T[:dim, :dim] = U @ np.diag(d) @ Vt

    if estimate_scale:
        src_var = src_demean.var(axis=0).sum()
        if src_var < 1e-10:
            return None
        scale = (S * d).sum() / src_var
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean)
    T[:dim, :dim] *= scale

    return T
