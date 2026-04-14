"""
M-Flow caching layer configuration.

Centralises all tunables for the caching and distributed-lock subsystem.
Two storage backends are supported out of the box:

* **Redis** – for horizontally-scaled deployments that require a shared
  cache across multiple process instances.
* **Filesystem** – lightweight default for single-node or development
  environments.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Literal, Optional

from pydantic_settings import BaseSettings
from m_flow.config.env_compat import MflowSettings, SettingsConfigDict

BACKEND_REDIS = "redis"
BACKEND_FILESYSTEM = "fs"


class CacheConfig(MflowSettings):
    """Runtime settings for the M-Flow caching subsystem.

    All fields are populated from environment variables (or a ``.env``
    file) following the standard ``pydantic-settings`` resolution order.

    Fields
    ------
    cache_backend
        Active backend identifier (``"redis"`` or ``"fs"``).
    caching
        Global toggle — when ``False`` the cache layer is bypassed.
    shared_kuzu_lock
        Activates cross-process locking for the embedded Kùzu store.
    cache_host / cache_port
        Network coordinates of the Redis instance (ignored for ``fs``).
    cache_username / cache_password
        Optional Redis AUTH credentials.
    agentic_lock_expire
        Seconds before an acquired distributed lock auto-releases.
    agentic_lock_timeout
        Maximum seconds to wait when acquiring a distributed lock.
    """

    cache_backend: Literal["redis", "fs"] = BACKEND_FILESYSTEM
    caching: bool = False
    shared_kuzu_lock: bool = False
    cache_host: str = "localhost"
    cache_port: int = 6379
    cache_username: Optional[str] = None
    cache_password: Optional[str] = None
    agentic_lock_expire: int = 240
    agentic_lock_timeout: int = 300

    model_config = SettingsConfigDict(env_prefix="MFLOW_", env_file=".env", extra="allow")

    def to_dict(self) -> Dict[str, Any]:
        """Materialise the full configuration as a plain dictionary."""
        return {
            "cache_backend": self.cache_backend,
            "caching": self.caching,
            "shared_kuzu_lock": self.shared_kuzu_lock,
            "cache_host": self.cache_host,
            "cache_port": self.cache_port,
            "cache_username": self.cache_username,
            "cache_password": self.cache_password,
            "agentic_lock_expire": self.agentic_lock_expire,
            "agentic_lock_timeout": self.agentic_lock_timeout,
        }

    @property
    def uses_redis(self) -> bool:
        """``True`` when the active backend is Redis."""
        return self.cache_backend == BACKEND_REDIS


@lru_cache(maxsize=1)
def get_cache_config() -> CacheConfig:
    """Return the process-wide ``CacheConfig`` singleton.

    The instance is constructed on first call and memoised for the
    remainder of the process lifetime.
    """
    return CacheConfig()
