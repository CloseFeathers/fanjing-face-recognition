# Copyright (c) 2024
# Session-level Person Registry
"""
Session-level Person Registry

功能:
1. 管理会话内的 person 身份
2. 接收新的 track template，决定匹配已有 person 还是创建新 person
3. 支持 margin 条件: top1_similarity - top2_similarity > margin

匹配逻辑:
1. 计算新 template 与所有 person templates 的相似度
2. 如果 top1_similarity > threshold 且 margin 条件满足:
   - 绑定到 top1 person
   - 更新 person template (可选: 滑动平均)
3. 否则创建新 person

阈值说明:
- similarity_threshold: 0.4 (默认, 余弦相似度)
  - ArcFace 的典型阈值在 0.3-0.5 之间
  - 同一人相似度通常 > 0.5
  - 不同人相似度通常 < 0.3
- margin_threshold: 0.1 (默认)
  - 确保 top1 明显优于 top2
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from .track_template import TrackTemplate


@dataclass
class PersonTemplate:
    """Person 的聚合 template"""

    person_id: int
    template: np.ndarray  # [512] 归一化向量
    track_ids: List[int]  # 关联的 track_id 列表
    sample_count: int
    first_seen_frame: int
    last_seen_frame: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """转换为可序列化的字典"""
        return {
            "person_id": self.person_id,
            "track_ids": self.track_ids,
            "sample_count": self.sample_count,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "created_at": self.created_at,
        }


@dataclass
class PersonAssignment:
    """Track 到 Person 的分配结果"""

    track_id: int
    person_id: int
    is_new_person: bool
    top1_similarity: float
    top2_similarity: Optional[float]
    margin: Optional[float]
    threshold: float
    margin_threshold: float
    decision: str  # "matched", "new_person", "margin_fail"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """转换为可序列化的字典"""
        return {
            "track_id": self.track_id,
            "person_id": self.person_id,
            "is_new_person": self.is_new_person,
            "top1_similarity": round(self.top1_similarity, 4),
            "top2_similarity": (
                round(self.top2_similarity, 4)
                if self.top2_similarity is not None
                else None
            ),
            "margin": round(self.margin, 4) if self.margin is not None else None,
            "threshold": self.threshold,
            "margin_threshold": self.margin_threshold,
            "decision": self.decision,
            "timestamp": self.timestamp,
        }


class PersonRegistry:
    """
    Session-level Person Registry

    管理会话内的 person 身份，实现 track 到 person 的匹配
    """

    def __init__(
        self,
        similarity_threshold: float = 0.4,
        margin_threshold: float = 0.1,
        update_template: bool = True,
        update_weight: float = 0.3,
        log_path: Optional[str] = "output/person_assignments.jsonl",
    ):
        """
        初始化 Person Registry

        Args:
            similarity_threshold: 匹配阈值 (余弦相似度)
            margin_threshold: margin 阈值 (top1 - top2 > margin)
            update_template: 是否在匹配后更新 person template
            update_weight: 更新权重 (new = old * (1-w) + track * w)
            log_path: 分配日志路径
        """
        self.similarity_threshold = similarity_threshold
        self.margin_threshold = margin_threshold
        self.update_template = update_template
        self.update_weight = update_weight
        self.log_path = log_path

        # person_id -> PersonTemplate
        self._persons: Dict[int, PersonTemplate] = {}

        # track_id -> person_id (映射表)
        self._track_to_person: Dict[int, int] = {}

        # 下一个 person_id
        self._next_person_id: int = 1

        # 日志文件
        self._log_file = None

    def open(self) -> "PersonRegistry":
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

    def assign(self, track_template: TrackTemplate) -> PersonAssignment:
        """
        将 track template 分配给 person

        Args:
            track_template: Track 的聚合 template

        Returns:
            PersonAssignment 分配结果
        """
        track_id = track_template.track_id
        track_vec = track_template.template

        # 如果已经分配过，返回现有分配
        if track_id in self._track_to_person:
            person_id = self._track_to_person[track_id]
            return PersonAssignment(
                track_id=track_id,
                person_id=person_id,
                is_new_person=False,
                top1_similarity=-1.0,  # 已分配,未重新计算
                top2_similarity=None,
                margin=None,
                threshold=self.similarity_threshold,
                margin_threshold=self.margin_threshold,
                decision="already_assigned",
            )

        # 计算与所有 person 的相似度
        similarities = []
        for pid, person in self._persons.items():
            sim = float(np.dot(track_vec, person.template))
            similarities.append((pid, sim))

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 获取 top1 和 top2
        top1_pid, top1_sim = similarities[0] if similarities else (None, -1.0)
        top2_sim = similarities[1][1] if len(similarities) > 1 else None
        margin = (top1_sim - top2_sim) if top2_sim is not None else None

        # 决策逻辑
        if top1_sim >= self.similarity_threshold:
            if margin is None or margin >= self.margin_threshold:
                # 匹配成功
                person_id = top1_pid
                is_new = False
                decision = "matched"

                # 更新 person template
                if self.update_template:
                    self._update_person_template(person_id, track_template)

                # 添加 track 到 person 的关联
                self._persons[person_id].track_ids.append(track_id)
                self._persons[person_id].last_seen_frame = track_template.last_frame_id
            else:
                # margin 不足，创建新 person
                person_id = self._create_person(track_template)
                is_new = True
                decision = "margin_fail"
        else:
            # 相似度不足，创建新 person
            person_id = self._create_person(track_template)
            is_new = True
            decision = "new_person"

        # 记录映射
        self._track_to_person[track_id] = person_id

        # 更新 track_template 中的 person 信息
        track_template.person_id = person_id
        track_template.similarity_to_person = top1_sim if not is_new else -1.0

        # 创建分配结果
        assignment = PersonAssignment(
            track_id=track_id,
            person_id=person_id,
            is_new_person=is_new,
            top1_similarity=top1_sim if top1_pid is not None else -1.0,
            top2_similarity=top2_sim,
            margin=margin,
            threshold=self.similarity_threshold,
            margin_threshold=self.margin_threshold,
            decision=decision,
        )

        # 写入日志
        self._log_assignment(assignment)

        return assignment

    def _create_person(self, track_template: TrackTemplate) -> int:
        """创建新 person"""
        person_id = self._next_person_id
        self._next_person_id += 1

        person = PersonTemplate(
            person_id=person_id,
            template=track_template.template.copy(),
            track_ids=[track_template.track_id],
            sample_count=track_template.sample_count,
            first_seen_frame=track_template.first_frame_id,
            last_seen_frame=track_template.last_frame_id,
        )
        self._persons[person_id] = person

        return person_id

    def _update_person_template(
        self, person_id: int, track_template: TrackTemplate
    ) -> None:
        """更新 person template (滑动平均)"""
        person = self._persons.get(person_id)
        if person is None:
            return

        # 滑动平均: new = old * (1-w) + track * w
        w = self.update_weight
        new_template = person.template * (1 - w) + track_template.template * w

        # L2 归一化
        norm = np.linalg.norm(new_template)
        if norm > 0:
            new_template = new_template / norm

        person.template = new_template.astype(np.float32)
        person.sample_count += track_template.sample_count

    def get_person_id(self, track_id: int) -> Optional[int]:
        """获取 track 对应的 person_id"""
        return self._track_to_person.get(track_id)

    def get_person(self, person_id: int) -> Optional[PersonTemplate]:
        """获取 person template"""
        return self._persons.get(person_id)

    def get_all_persons(self) -> Dict[int, PersonTemplate]:
        """获取所有 persons"""
        return self._persons.copy()

    def get_person_count(self) -> int:
        """获取当前 person 数量"""
        return len(self._persons)

    def _log_assignment(self, assignment: PersonAssignment) -> None:
        """写入分配日志"""
        if self._log_file:
            self._log_file.write(
                json.dumps(assignment.to_dict(), ensure_ascii=False) + "\n"
            )
            self._log_file.flush()

    def reset(self) -> None:
        """重置会话"""
        self._persons.clear()
        self._track_to_person.clear()
        self._next_person_id = 1
