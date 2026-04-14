"""
Abstract interface for distributed cache backends (Redis, Memcached, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, List, Optional


class CacheDBInterface(ABC):
    """
    Contract for cache coordination backends.

    Subclasses must implement lock primitives and Q/A session storage.
    """

    __slots__ = ("_host", "_port", "_key", "_lock")

    def __init__(self, host: str, port: int, lock_key: str) -> None:
        self._host = host
        self._port = port
        self._key = lock_key
        self._lock: Any = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def lock_key(self) -> str:
        return self._key

    @property
    def lock(self) -> Any:
        return self._lock

    @lock.setter
    def lock(self, val: Any) -> None:
        self._lock = val

    # -------------------------------------------------------------------------
    # Lock primitives
    # -------------------------------------------------------------------------

    @abstractmethod
    def acquire_lock(self) -> bool:
        """Try to acquire the distributed lock. Return True on success."""
        ...

    @abstractmethod
    def release_lock(self) -> None:
        """Release the lock if currently held."""
        ...

    # Aliases
    def acquire(self) -> bool:
        return self.acquire_lock()

    def release(self) -> None:
        return self.release_lock()

    @contextmanager
    def hold_lock(self):
        """Context manager wrapping acquire/release."""
        self.acquire_lock()
        try:
            yield
        finally:
            self.release_lock()

    # -------------------------------------------------------------------------
    # Q/A session storage
    # -------------------------------------------------------------------------

    @abstractmethod
    async def add_qa(
        self,
        user_id: str,
        session_id: str,
        question: str,
        context: str,
        answer: str,
        ttl: Optional[int] = 86400,
    ) -> None:
        """Persist a question/answer/context triplet."""
        ...

    @abstractmethod
    async def get_latest_qa(
        self,
        user_id: str,
        session_id: str,
        limit: int = 5,
    ) -> List[Any]:
        """Retrieve the N most recent Q/A triplets."""
        ...

    @abstractmethod
    async def get_all_qas(
        self,
        user_id: str,
        session_id: str,
    ) -> List[Any]:
        """Retrieve all Q/A triplets for the session."""
        ...

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    @abstractmethod
    async def close(self) -> None:
        """Release any underlying connections."""
        ...
