"""
缓存配置单元测试
"""

from __future__ import annotations

import pytest
from m_flow.adapters.cache.config import CacheConfig, get_cache_config


class TestCacheConfigDefaults:
    """测试默认配置"""

    def test_backend(self):
        assert CacheConfig().cache_backend == "fs"

    def test_caching_disabled(self):
        assert CacheConfig().caching is False

    def test_kuzu_lock_disabled(self):
        assert CacheConfig().shared_kuzu_lock is False

    def test_host(self):
        assert CacheConfig().cache_host == "localhost"

    def test_port(self):
        assert CacheConfig().cache_port == 6379

    def test_lock_expire(self):
        assert CacheConfig().agentic_lock_expire == 240

    def test_lock_timeout(self):
        assert CacheConfig().agentic_lock_timeout == 300


class TestCacheConfigCustom:
    """测试自定义配置"""

    @pytest.fixture
    def custom_config(self):
        return CacheConfig(
            cache_backend="redis",
            caching=True,
            shared_kuzu_lock=True,
            cache_host="redis.test.local",
            cache_port=6380,
            agentic_lock_expire=120,
            agentic_lock_timeout=180,
        )

    def test_backend(self, custom_config):
        assert custom_config.cache_backend == "redis"

    def test_caching_enabled(self, custom_config):
        assert custom_config.caching is True

    def test_kuzu_lock(self, custom_config):
        assert custom_config.shared_kuzu_lock is True

    def test_host(self, custom_config):
        assert custom_config.cache_host == "redis.test.local"

    def test_port(self, custom_config):
        assert custom_config.cache_port == 6380


class TestCacheConfigDict:
    """测试字典序列化"""

    def test_to_dict_keys(self):
        cfg = CacheConfig(
            cache_backend="fs",
            caching=True,
            cache_host="host1",
            cache_port=7000,
            agentic_lock_expire=100,
            agentic_lock_timeout=200,
        )
        d = cfg.to_dict()
        assert d["cache_backend"] == "fs"
        assert d["caching"] is True
        assert d["cache_host"] == "host1"
        assert d["cache_port"] == 7000
        assert "cache_username" in d
        assert "cache_password" in d


class TestCacheConfigSingleton:
    """测试单例模式"""

    def test_same_instance(self):
        cfg1 = get_cache_config()
        cfg2 = get_cache_config()
        assert cfg1 is cfg2


class TestCacheConfigExtra:
    """测试额外字段"""

    def test_extra_fields_allowed(self):
        cfg = CacheConfig(custom_key="val", num=42)
        assert cfg.custom_key == "val"
        assert cfg.num == 42


class TestCacheConfigBooleans:
    """测试布尔值解析"""

    def test_truthy_strings(self):
        cfg = CacheConfig(caching="true", shared_kuzu_lock="yes")
        assert cfg.caching is True
        assert cfg.shared_kuzu_lock is True

    def test_falsy_strings(self):
        cfg = CacheConfig(caching="false", shared_kuzu_lock="no")
        assert cfg.caching is False
        assert cfg.shared_kuzu_lock is False
