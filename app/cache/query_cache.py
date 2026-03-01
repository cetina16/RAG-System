"""In-memory LRU cache for RAG pipeline results."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from cachetools import TTLCache

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_cache: TTLCache | None = None


def _get_cache() -> TTLCache:
    global _cache
    if _cache is None:
        settings = get_settings()
        _cache = TTLCache(
            maxsize=settings.cache_max_size,
            ttl=settings.cache_ttl_seconds,
        )
    return _cache


def _make_key(query: str, top_k: int) -> str:
    payload = json.dumps({"query": query.strip().lower(), "top_k": top_k}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_get(query: str, top_k: int) -> Any | None:
    """Return cached result or None on miss."""
    key = _make_key(query, top_k)
    result = _get_cache().get(key)
    if result is not None:
        logger.info("cache_hit", query_preview=query[:60])
    return result


def cache_set(query: str, top_k: int, value: Any) -> None:
    """Store result in cache."""
    key = _make_key(query, top_k)
    _get_cache()[key] = value
    logger.info("cache_set", query_preview=query[:60], key=key[:12])


def cache_stats() -> dict[str, Any]:
    c = _get_cache()
    return {
        "size":     len(c),
        "maxsize":  c.maxsize,
        "ttl":      c.ttl,
    }
