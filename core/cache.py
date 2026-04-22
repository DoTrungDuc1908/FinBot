"""
core/cache.py
Redis cache layer with JSON serialization, TTL support, and async compatibility.
"""
import json
import hashlib
from typing import Any, Optional
from functools import wraps

import redis
from loguru import logger

from config.settings import settings


class CacheClient:
    """Synchronous Redis cache client with JSON serialization."""

    def __init__(self) -> None:
        self._client: redis.Redis | None = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password or None,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
        return self._client

    def get(self, key: str) -> Any | None:
        try:
            raw = self.client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"Cache GET error for key={key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        try:
            serialized = json.dumps(value, ensure_ascii=False, default=str)
            self.client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.warning(f"Cache SET error for key={key}: {e}")
            return False

    def delete(self, key: str) -> None:
        try:
            self.client.delete(key)
        except Exception as e:
            logger.warning(f"Cache DELETE error for key={key}: {e}")

    def exists(self, key: str) -> bool:
        try:
            return bool(self.client.exists(key))
        except Exception:
            return False

    def ping(self) -> bool:
        try:
            return self.client.ping()
        except Exception:
            return False

    @staticmethod
    def build_key(*parts: str) -> str:
        """Build a namespaced cache key."""
        raw = ":".join(str(p) for p in parts)
        return f"finbot:{raw}"

    @staticmethod
    def hash_key(data: Any) -> str:
        """Hash arbitrary data to a short cache key suffix."""
        s = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()[:12]


# Singleton instance
cache = CacheClient()


def cached(ttl: int = 300, key_prefix: str = ""):
    """
    Decorator to cache function return values in Redis.
    Cache key is built from prefix + function name + hashed args.

    Usage:
        @cached(ttl=300, key_prefix="stock")
        def get_price(ticker: str) -> dict: ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            prefix = key_prefix or func.__qualname__
            arg_hash = CacheClient.hash_key({"args": args, "kwargs": kwargs})
            key = CacheClient.build_key(prefix, func.__name__, arg_hash)

            cached_val = cache.get(key)
            if cached_val is not None:
                logger.debug(f"Cache HIT: {key}")
                return cached_val

            result = func(*args, **kwargs)
            if result is not None:
                cache.set(key, result, ttl=ttl)
                logger.debug(f"Cache SET: {key} (ttl={ttl}s)")
            return result
        return wrapper
    return decorator
