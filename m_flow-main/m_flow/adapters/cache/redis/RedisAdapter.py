"""
Redis cache adapter for M-flow.

Provides distributed locking and session storage using Redis.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import redis
import redis.asyncio as aioredis

from m_flow.adapters.cache.cache_db_interface import CacheDBInterface
from m_flow.adapters.exceptions import CacheConnectionError
from m_flow.shared.logging_utils import get_logger

_log = get_logger("RedisAdapter")


class RedisAdapter(CacheDBInterface):
    """
    Redis-based cache and locking implementation.

    Provides:
      - Distributed locking for concurrent operations
      - Session-based Q&A storage with TTL support
    """

    def __init__(
        self,
        host: str,
        port: int,
        lock_name: str = "default_lock",
        username: str | None = None,
        password: str | None = None,
        timeout: int = 240,
        blocking_timeout: int = 300,
        connection_timeout: int = 30,
    ) -> None:
        super().__init__(host, port, lock_name)

        self.connection_timeout = connection_timeout
        self.timeout = timeout
        self.blocking_timeout = blocking_timeout
        self.lock = None

        try:
            self.sync_redis = redis.Redis(
                host=host,
                port=port,
                username=username,
                password=password,
                socket_connect_timeout=connection_timeout,
                socket_timeout=connection_timeout,
            )
            self.async_redis = aioredis.Redis(
                host=host,
                port=port,
                username=username,
                password=password,
                decode_responses=True,
                socket_connect_timeout=connection_timeout,
            )

            self._ping_check()
            _log.info(f"Connected to Redis at {host}:{port}")

        except (redis.ConnectionError, redis.TimeoutError) as e:
            raise CacheConnectionError(f"Redis connection failed ({host}:{port}): {e}") from e
        except Exception as e:
            raise CacheConnectionError(f"Redis init error: {e}") from e

    def _ping_check(self) -> None:
        """Verify connectivity."""
        try:
            self.sync_redis.ping()
        except (redis.ConnectionError, redis.TimeoutError) as e:
            raise CacheConnectionError(f"Redis ping failed: {e}") from e

    def acquire_lock(self) -> Any:
        """Acquire distributed lock (sync, for Kuzu compatibility)."""
        self.lock = self.sync_redis.lock(
            name=self.lock_key,
            timeout=self.timeout,
            blocking_timeout=self.blocking_timeout,
        )

        if not self.lock.acquire():
            raise RuntimeError(f"Lock acquisition failed: {self.lock_key}")

        return self.lock

    def release_lock(self) -> None:
        """Release distributed lock if held."""
        if self.lock:
            try:
                self.lock.release()
            except redis.exceptions.LockError:
                pass
            finally:
                self.lock = None

    @contextmanager
    def hold_lock(self):
        """Context manager for lock lifecycle."""
        self.acquire_lock()
        try:
            yield
        finally:
            self.release_lock()

    async def add_qa(
        self,
        user_id: str,
        session_id: str,
        question: str,
        context: str,
        answer: str,
        ttl: int | None = 86400,
    ) -> None:
        """
        Store Q&A entry in session list.

        Args:
            user_id: User identifier.
            session_id: Session identifier.
            question: User query.
            context: Context used for answer.
            answer: Generated response.
            ttl: Expiration in seconds (default: 24h).
        """
        try:
            key = f"agent_sessions:{user_id}:{session_id}"

            entry = {
                "time": datetime.utcnow().isoformat(),
                "question": question,
                "context": context,
                "answer": answer,
            }

            await self.async_redis.rpush(key, json.dumps(entry, default=str))

            if ttl:
                await self.async_redis.expire(key, ttl)

        except (redis.ConnectionError, redis.TimeoutError) as e:
            raise CacheConnectionError(f"Redis add_qa failed: {e}") from e

    async def get_latest_qa(
        self,
        user_id: str,
        session_id: str,
        last_n: int = 5,
    ) -> list[dict]:
        """Retrieve recent Q&A entries."""
        key = f"agent_sessions:{user_id}:{session_id}"

        if last_n == 1:
            data = await self.async_redis.lindex(key, -1)
            return [json.loads(data)] if data else []

        entries = await self.async_redis.lrange(key, -last_n, -1)
        return [json.loads(e) for e in entries] if entries else []

    async def get_all_qas(self, user_id: str, session_id: str) -> list[dict]:
        """Retrieve all session Q&A entries."""
        key = f"agent_sessions:{user_id}:{session_id}"
        entries = await self.async_redis.lrange(key, 0, -1)
        return [json.loads(e) for e in entries]

    async def close(self) -> None:
        """Close async Redis connection."""
        await self.async_redis.aclose()
