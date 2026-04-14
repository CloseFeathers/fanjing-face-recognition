# Copyright (c) 2024
# Track Template Manager
"""
Track Template 管理器

功能:
1. 收集单个 track 的多个 embedding 样本
2. 聚合生成 track template (平均 / 质量加权平均)
3. 管理 template 生命周期

聚合策略:
- simple_mean: 简单平均，所有样本权重相等
- quality_weighted: 质量加权平均，quality_score 作为权重
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np


@dataclass
class EmbeddingSample:
    """单个 embedding 样本"""

    frame_id: int
    timestamp_ms: float
    embedding: np.ndarray  # [512]
    quality_score: float
    image_path: Optional[str] = None


@dataclass
class TrackTemplate:
    """Track 的聚合 template"""

    track_id: int
    template: np.ndarray  # [512] 归一化向量
    sample_count: int
    avg_quality: float
    first_frame_id: int
    last_frame_id: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # 可选: 关联的 person_id
    person_id: Optional[int] = None
    similarity_to_person: Optional[float] = None

    def to_dict(self) -> Dict:
        """转换为可序列化的字典"""
        return {
            "track_id": self.track_id,
            "template": self.template.tolist(),
            "sample_count": self.sample_count,
            "avg_quality": round(self.avg_quality, 4),
            "first_frame_id": self.first_frame_id,
            "last_frame_id": self.last_frame_id,
            "created_at": self.created_at,
            "person_id": self.person_id,
            "similarity_to_person": (
                round(self.similarity_to_person, 4)
                if self.similarity_to_person is not None
                else None
            ),
        }


class TrackTemplateManager:
    """
    Track Template 管理器

    管理多个 track 的 embedding 样本收集和 template 聚合
    """

    def __init__(
        self,
        min_samples: int = 3,
        max_samples: int = 10,
        aggregation: str = "quality_weighted",
        log_path: Optional[str] = "output/track_templates.jsonl",
    ):
        """
        初始化 Track Template Manager

        Args:
            min_samples: 生成 template 的最小样本数
            max_samples: 每个 track 保留的最大样本数
            aggregation: 聚合方式 ("simple_mean" 或 "quality_weighted")
            log_path: JSONL 日志路径
        """
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.aggregation = aggregation
        self.log_path = log_path

        # track_id -> List[EmbeddingSample]
        self._samples: Dict[int, List[EmbeddingSample]] = {}

        # track_id -> TrackTemplate (已生成的 template)
        self._templates: Dict[int, TrackTemplate] = {}

        # 日志文件
        self._log_file = None

    def open(self) -> "TrackTemplateManager":
        """打开日志文件"""
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            self._log_file = open(self.log_path, "a", encoding="utf-8")
        return self

    def close(self) -> None:
        """关闭日志文件"""
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def add_sample(
        self,
        track_id: int,
        frame_id: int,
        timestamp_ms: float,
        embedding: np.ndarray,
        quality_score: float,
        image_path: Optional[str] = None,
    ) -> Optional[TrackTemplate]:
        """
        添加一个 embedding 样本

        Args:
            track_id: 轨迹 ID
            frame_id: 帧 ID
            timestamp_ms: 时间戳
            embedding: 嵌入向量 [512]
            quality_score: 质量分数
            image_path: 保存的图像路径

        Returns:
            如果达到 min_samples 并更新了 template，返回 TrackTemplate
        """
        if embedding is None or len(embedding) == 0:
            return None

        # 创建样本
        sample = EmbeddingSample(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            embedding=embedding,
            quality_score=quality_score,
            image_path=image_path,
        )

        # 添加到样本列表
        if track_id not in self._samples:
            self._samples[track_id] = []

        samples = self._samples[track_id]
        samples.append(sample)

        # 如果超过 max_samples，按质量排序保留最好的
        if len(samples) > self.max_samples:
            samples.sort(key=lambda s: s.quality_score, reverse=True)
            self._samples[track_id] = samples[: self.max_samples]

        # 检查是否可以生成/更新 template
        if len(self._samples[track_id]) >= self.min_samples:
            template = self._generate_template(track_id)
            if template:
                self._templates[track_id] = template
                self._log_template(template)
                return template

        return None

    def _generate_template(self, track_id: int) -> Optional[TrackTemplate]:
        """
        聚合样本生成 template

        Args:
            track_id: 轨迹 ID

        Returns:
            TrackTemplate 或 None
        """
        samples = self._samples.get(track_id, [])
        if len(samples) < self.min_samples:
            return None

        embeddings = np.array([s.embedding for s in samples])  # [N, 512]
        qualities = np.array([s.quality_score for s in samples])  # [N]

        if self.aggregation == "simple_mean":
            # 简单平均
            template_vec = np.mean(embeddings, axis=0)
        elif self.aggregation == "quality_weighted":
            # 质量加权平均
            # 避免权重为 0
            weights = np.clip(qualities, 0.01, None)
            weights = weights / weights.sum()
            template_vec = np.average(embeddings, axis=0, weights=weights)
        else:
            # 默认简单平均
            template_vec = np.mean(embeddings, axis=0)

        # L2 归一化
        norm = np.linalg.norm(template_vec)
        if norm > 0:
            template_vec = template_vec / norm

        return TrackTemplate(
            track_id=track_id,
            template=template_vec.astype(np.float32),
            sample_count=len(samples),
            avg_quality=float(np.mean(qualities)),
            first_frame_id=min(s.frame_id for s in samples),
            last_frame_id=max(s.frame_id for s in samples),
        )

    def get_template(self, track_id: int) -> Optional[TrackTemplate]:
        """获取指定 track 的 template"""
        return self._templates.get(track_id)

    def get_all_templates(self) -> Dict[int, TrackTemplate]:
        """获取所有已生成的 templates"""
        return self._templates.copy()

    def get_sample_count(self, track_id: int) -> int:
        """获取指定 track 的样本数"""
        return len(self._samples.get(track_id, []))

    def _log_template(self, template: TrackTemplate) -> None:
        """写入 JSONL 日志"""
        if self._log_file:
            # 不在日志中保存完整的 512 维向量，只保存元数据
            log_entry = {
                "track_id": template.track_id,
                "sample_count": template.sample_count,
                "avg_quality": round(template.avg_quality, 4),
                "first_frame_id": template.first_frame_id,
                "last_frame_id": template.last_frame_id,
                "created_at": template.created_at,
                "person_id": template.person_id,
                "similarity_to_person": template.similarity_to_person,
                "template_norm": round(float(np.linalg.norm(template.template)), 4),
            }
            self._log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            self._log_file.flush()

    def clear_track(self, track_id: int) -> None:
        """清除指定 track 的数据"""
        self._samples.pop(track_id, None)
        self._templates.pop(track_id, None)

    def reset(self) -> None:
        """重置所有数据"""
        self._samples.clear()
        self._templates.clear()
