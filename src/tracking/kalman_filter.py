"""
8-dimensional Kalman Filter — For bounding box motion estimation.

State vector x = [cx, cy, w, h, vx, vy, vw, vh]
Observation vector z = [cx, cy, w, h]
Motion model: constant velocity

This implementation is a stateless utility class: mean/covariance held by STrack,
KalmanFilter only provides initiate / predict / update pure functions.
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """Stateless Kalman filter, shared by all tracks."""

    _std_weight_position = 1.0 / 20
    _std_weight_velocity = 1.0 / 160

    def __init__(self) -> None:
        ndim = 4
        dt = 1.0

        # State transition matrix F (8x8)
        self._F = np.eye(2 * ndim, dtype=np.float64)
        for i in range(ndim):
            self._F[i, ndim + i] = dt

        # Observation matrix H (4x8)
        self._H = np.eye(ndim, 2 * ndim, dtype=np.float64)

    # ------------------------------------------------------------------

    def initiate(self, measurement: np.ndarray):
        """Initialize state from first observation.

        Args:
            measurement: [cx, cy, w, h]
        Returns:
            (mean, covariance)
        """
        mean_pos = measurement.astype(np.float64)
        mean_vel = np.zeros(4, dtype=np.float64)
        mean = np.concatenate([mean_pos, mean_vel])

        h = measurement[3]
        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Predict next frame state."""
        h = max(mean[3], 1.0)
        std_pos = self._std_weight_position * h
        std_vel = self._std_weight_velocity * h
        Q = np.diag([
            std_pos ** 2, std_pos ** 2, std_pos ** 2, std_pos ** 2,
            std_vel ** 2, std_vel ** 2, std_vel ** 2, std_vel ** 2,
        ])

        mean = self._F @ mean
        covariance = self._F @ covariance @ self._F.T + Q

        # Ensure w, h > 0
        mean[2] = max(mean[2], 1.0)
        mean[3] = max(mean[3], 1.0)
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray):
        """Correct state with observation."""
        h = max(mean[3], 1.0)
        std = self._std_weight_position * h
        R = np.diag([std ** 2] * 4)

        projected_mean = self._H @ mean
        projected_cov = self._H @ covariance @ self._H.T + R

        # Kalman gain
        K = covariance @ self._H.T @ np.linalg.inv(projected_cov)

        innovation = measurement.astype(np.float64) - projected_mean
        new_mean = mean + K @ innovation
        new_covariance = (np.eye(8) - K @ self._H) @ covariance

        new_mean[2] = max(new_mean[2], 1.0)
        new_mean[3] = max(new_mean[3], 1.0)
        return new_mean, new_covariance
