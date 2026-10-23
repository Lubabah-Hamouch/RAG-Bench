"""Request-level protections: log redaction, rate limiting, in-flight guard, HTTPS and key checks.

Nothing here stores or derives state from an API key. Rate limits and the in-flight guard are
keyed by client IP only, so no key (or hash of a key) is ever held in memory beyond the request.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

from fastapi import Request

from ragbench.api.settings import ApiSettings
from ragbench.llm.client import scrub

KEY_HEADER = "X-Groq-Api-Key"
_KEY_OK = re.compile(r"^[A-Za-z0-9_\-.]{20,256}$")

_installed = False


def valid_key_shape(key: str | None) -> bool:
    """Plausible bearer token: 20-256 URL-safe characters, no whitespace or control characters."""
    return bool(key and _KEY_OK.match(key))


# ---- log redaction ------------------------------------------------------------------------------


class RedactionFilter(logging.Filter):
    """Masks key-shaped strings in a log record's message and arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = scrub(record.msg) if isinstance(record.msg, str) else record.msg
        if record.args:
            args = record.args if isinstance(record.args, tuple) else (record.args,)
            record.args = tuple(scrub(a) if isinstance(a, str) else a for a in args)
        return True


def install_log_redaction() -> None:
    """Scrub every log record at creation (all loggers and handlers). Idempotent."""
    global _installed
    if _installed:
        return
    previous = logging.getLogRecordFactory()

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        RedactionFilter().filter(record)
        return record

    logging.setLogRecordFactory(factory)
    _installed = True


# ---- rate limiting ------------------------------------------------------------------------------


class RateLimiter:
    """Sliding-window limiter keyed by an opaque string (the client IP)."""

    def __init__(self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, float]:
        """Record a hit. Returns (allowed, seconds_until_a_slot_frees_if_denied)."""
        now = self._clock()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, max(0.0, self.window_s - (now - hits[0]))
            hits.append(now)
            if not hits:  # pragma: no cover - defensive
                self._hits.pop(key, None)
            return True, 0.0


class InflightGuard:
    """At most one run in flight per client, and a global cap on concurrent runs."""

    def __init__(self, max_total: int):
        self.max_total = max_total
        self._active: set[str] = set()
        self._lock = threading.Lock()

    def acquire(self, client: str) -> str | None:
        """Returns None on success, or the reason it was refused."""
        with self._lock:
            if client in self._active:
                return "run_in_progress"
            if len(self._active) >= self.max_total:
                return "server_busy"
            self._active.add(client)
            return None

    def release(self, client: str) -> None:
        with self._lock:
            self._active.discard(client)


# ---- request helpers ----------------------------------------------------------------------------


def client_ip(request: Request, settings: ApiSettings) -> str:
    if settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def is_secure(request: Request, settings: ApiSettings) -> bool:
    if request.url.scheme == "https":
        return True
    return settings.trust_proxy_headers and request.headers.get("x-forwarded-proto") == "https"
