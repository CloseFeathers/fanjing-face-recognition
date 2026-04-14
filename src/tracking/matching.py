"""
匹配工具 —— IoU 计算 + 线性分配。

优先使用 scipy.optimize.linear_sum_assignment (匈牙利算法)；
若 scipy 不可用则退回贪心匹配（对少量人脸足够）。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ======================================================================
# IoU
# ======================================================================

def iou_batch(
    bboxes_a: np.ndarray,
    bboxes_b: np.ndarray,
) -> np.ndarray:
    """计算两组 bbox 的 IoU 矩阵。

    Args:
        bboxes_a: (N, 4) xyxy
        bboxes_b: (M, 4) xyxy
    Returns:
        (N, M) IoU 矩阵
    """
    x1 = np.maximum(bboxes_a[:, 0:1], bboxes_b[:, 0].T)
    y1 = np.maximum(bboxes_a[:, 1:2], bboxes_b[:, 1].T)
    x2 = np.minimum(bboxes_a[:, 2:3], bboxes_b[:, 2].T)
    y2 = np.minimum(bboxes_a[:, 3:4], bboxes_b[:, 3].T)

    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / np.maximum(union, 1e-7)


# ======================================================================
# 线性分配
# ======================================================================

def linear_assignment(
    cost_matrix: np.ndarray,
    thresh: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """求解线性分配问题。

    Args:
        cost_matrix: (N, M) 代价矩阵 (越小越好, 通常 = 1 - IoU)
        thresh:      最大允许代价 (超过则不匹配)
    Returns:
        (matches, unmatched_rows, unmatched_cols)
    """
    if cost_matrix.size == 0:
        return (
            [],
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )

    if _HAS_SCIPY:
        row_idx, col_idx = _scipy_lsa(cost_matrix)
    else:
        row_idx, col_idx = _greedy_assignment(cost_matrix)

    matches = []
    unmatched_rows = set(range(cost_matrix.shape[0]))
    unmatched_cols = set(range(cost_matrix.shape[1]))

    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
            unmatched_rows.discard(r)
            unmatched_cols.discard(c)

    return matches, sorted(unmatched_rows), sorted(unmatched_cols)


def _greedy_assignment(
    cost_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """贪心分配 (scipy 不可用时的后备方案)。"""
    rows, cols = [], []
    used_r, used_c = set(), set()
    flat = cost_matrix.flatten()
    order = flat.argsort()

    for idx in order:
        r, c = divmod(int(idx), cost_matrix.shape[1])
        if r in used_r or c in used_c:
            continue
        rows.append(r)
        cols.append(c)
        used_r.add(r)
        used_c.add(c)
        if len(rows) == min(cost_matrix.shape):
            break

    return np.array(rows, dtype=int), np.array(cols, dtype=int)


# ======================================================================
# 关联入口
# ======================================================================

def associate(
    track_bboxes: np.ndarray,
    det_bboxes: np.ndarray,
    iou_threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int], List[float]]:
    """IoU 关联。

    Returns:
        (matches, unmatched_tracks, unmatched_dets, match_ious)
    """
    if len(track_bboxes) == 0 or len(det_bboxes) == 0:
        return (
            [],
            list(range(len(track_bboxes))),
            list(range(len(det_bboxes))),
            [],
        )

    iou_mat = iou_batch(
        np.asarray(track_bboxes, dtype=np.float64),
        np.asarray(det_bboxes, dtype=np.float64),
    )
    cost_mat = 1.0 - iou_mat
    matches, u_tracks, u_dets = linear_assignment(cost_mat, 1.0 - iou_threshold)
    match_ious = [float(iou_mat[r, c]) for r, c in matches]
    return matches, u_tracks, u_dets, match_ious
