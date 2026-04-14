"""
Cache engine factory.

Creates appropriate cache backend (Redis, filesystem) based on configuration.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from m_flow.adapters.cache.cache_db_interface import CacheDBInterface
from m_flow.adapters.cache.config import get_cache_config
from m_flow.adapters.cache.fscache.FsCacheAdapter import FSCacheAdapter

_cfg = get_cache_config()


@lru_cache
def create_cache_engine(
    cache_host: str,
    cache_port: int,
    cache_username: str,
    cache_password: str,
    lock_key: str,
    agentic_lock_expire: int = 240,
    agentic_lock_timeout: int = 300,
) -> CacheDBInterface | None:
    """
    Instantiate cache coordination backend.

    Args:
        cache_host: Cache server hostname.
        cache_port: Connection port.
        cache_username: Auth username.
        cache_password: Auth password.
        lock_key: Lock resource identifier.
        agentic_lock_expire: Lock hold duration.
        agentic_lock_timeout: Max lock wait time.

    Returns:
        Cache adapter instance or None if caching disabled.
    """
    if not _cfg.caching:
        return None

    backend = _cfg.cache_backend

    if backend == "redis":
        from m_flow.adapters.cache.redis.RedisAdapter import RedisAdapter

        return RedisAdapter(
            host=cache_host,
            port=cache_port,
            username=cache_username,
            password=cache_password,
            lock_name=lock_key,
            timeout=agentic_lock_expire,
            blocking_timeout=agentic_lock_timeout,
        )

    if backend == "fs":
        return FSCacheAdapter()

    raise ValueError(f"Unknown cache backend: '{backend}' (supported: redis, fs)")


def get_cache_engine(lock_key: Optional[str] = None) -> CacheDBInterface | None:
    """Get cache adapter with current config."""
    return create_cache_engine(
        cache_host=_cfg.cache_host,
        cache_port=_cfg.cache_port,
        cache_username=_cfg.cache_username,
        cache_password=_cfg.cache_password,
        lock_key=lock_key,
        agentic_lock_expire=_cfg.agentic_lock_expire,
        agentic_lock_timeout=_cfg.agentic_lock_timeout,
    )
