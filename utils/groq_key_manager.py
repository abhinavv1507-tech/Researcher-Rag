"""
utils/groq_key_manager.py

Production-grade Groq API key manager.

Features:
  - Round-robin key rotation across all 12 keys
  - Per-key cooldown when a 429 (rate-limit) is detected
  - Automatic skip to next available key
  - Thread-safe counter with simple lock
  - Logs every key rotation and cooldown event
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

from dotenv import load_dotenv

from utils.logging_config import get_logger

load_dotenv()
log = get_logger(__name__)

# ---------------------------------------------------------------------------
# How long (seconds) a key is skipped after hitting a 429
# ---------------------------------------------------------------------------
_KEY_COOLDOWN_SECONDS: float = float(os.getenv("GROQ_KEY_COOLDOWN_S", "60"))


class GroqKeyManager:
    """
    Thread-safe round-robin Groq API key manager with rate-limit cooldown.

    Usage::

        mgr = GroqKeyManager()
        key = mgr.next_key()          # get next key
        mgr.mark_rate_limited(key)    # cool this key down for 60 s
        mgr.mark_failed(key)          # permanently skip this key

    All public methods are thread-safe.
    """

    def __init__(self) -> None:
        self._keys: list[str] = _load_keys()
        self._lock = threading.Lock()
        self._index: int = 0
        # Per-key state: None = available, float = resume_at timestamp
        self._cooldown_until: dict[str, Optional[float]] = {k: None for k in self._keys}
        self._failed: set[str] = set()
        log.info("groq_key_manager.init", total_keys=len(self._keys))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_key(self) -> str:
        """
        Return the next available API key in round-robin order.
        Skips keys that are in cooldown or marked failed.
        Raises RuntimeError if no key is available.
        """
        with self._lock:
            n = len(self._keys)
            for attempt in range(n):
                key = self._keys[self._index % n]
                self._index += 1
                if self._is_available(key):
                    log.debug(
                        "groq_key_manager.key_selected",
                        key_suffix=key[-6:],
                        index=(self._index - 1) % n,
                    )
                    return key

            # All keys are in cooldown — return the one that recovers soonest
            soonest_key = min(
                (k for k in self._keys if k not in self._failed),
                key=lambda k: self._cooldown_until.get(k) or 0,
                default=None,
            )
            if soonest_key is None:
                raise RuntimeError("All Groq API keys have permanently failed.")

            wait = (self._cooldown_until[soonest_key] or 0) - time.monotonic()
            log.warning(
                "groq_key_manager.all_keys_cooling",
                wait_s=round(max(wait, 0), 1),
                key_suffix=soonest_key[-6:],
            )
            if wait > 0:
                time.sleep(wait)
            self._cooldown_until[soonest_key] = None
            return soonest_key

    def mark_rate_limited(self, key: str) -> None:
        """Put key in cooldown for GROQ_KEY_COOLDOWN_S seconds."""
        with self._lock:
            resume = time.monotonic() + _KEY_COOLDOWN_SECONDS
            self._cooldown_until[key] = resume
            log.warning(
                "groq_key_manager.key_rate_limited",
                key_suffix=key[-6:],
                cooldown_s=_KEY_COOLDOWN_SECONDS,
            )

    def mark_failed(self, key: str) -> None:
        """Permanently remove key from rotation (e.g. invalid key)."""
        with self._lock:
            self._failed.add(key)
            log.error("groq_key_manager.key_failed_permanently", key_suffix=key[-6:])

    @property
    def available_count(self) -> int:
        """Number of keys currently not in cooldown or failed."""
        with self._lock:
            return sum(1 for k in self._keys if self._is_available(k))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_available(self, key: str) -> bool:
        if key in self._failed:
            return False
        cooldown = self._cooldown_until.get(key)
        if cooldown is not None and time.monotonic() < cooldown:
            return False
        # Reset expired cooldown
        if cooldown is not None:
            self._cooldown_until[key] = None
        return True


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: Optional[GroqKeyManager] = None
_manager_lock = threading.Lock()


def get_key_manager() -> GroqKeyManager:
    """Return the process-wide GroqKeyManager singleton (created once)."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = GroqKeyManager()
    return _manager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_keys() -> list[str]:
    """
    Load Groq API keys from environment.
    Supports GROQ_API_KEYS (preferred, comma-separated) or GROQ_API_KEY.
    """
    raw = os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", "")).strip()
    if not raw:
        raise RuntimeError(
            "No Groq API keys found. "
            "Set GROQ_API_KEYS=key1,key2,... in your .env file."
        )
    keys = [k.strip() for k in raw.replace(";", ",").split(",") if k.strip()]
    if not keys:
        raise RuntimeError("GROQ_API_KEYS is set but contains no valid keys.")
    return keys
