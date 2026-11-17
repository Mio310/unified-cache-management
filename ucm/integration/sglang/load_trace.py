"""Opt-in request attribution for Direct Linker loads (no cache identity changes)."""

from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar
from uuid import uuid4

logger = logging.getLogger(__name__)
_request = ContextVar("ucm_sglang_trace_request", default=None)


def enabled() -> bool:
    return os.environ.get("UCM_SGLANG_LOAD_TRACE", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def emit(event: str, context: dict | None, **fields) -> None:
    if context is None:
        return
    # Observability must never turn a successful storage operation into a failure.
    try:
        logger.info(
            "[UCMTrace] %s",
            json.dumps(
                dict(context, **fields, event=event), ensure_ascii=True, sort_keys=True
            ),
        )
    except Exception:
        pass


def new_attempt(rid: str, **ranks) -> dict | None:
    if not enabled():
        return None
    request = _request.get()
    session_id = None
    if request is not None and getattr(request, "rid", None) == rid:
        session_id = getattr(request, "session_id", None)
    return dict(session_id=session_id, rid=rid, load_attempt_id=uuid4().hex, **ranks)


class TracedLoad(list):
    """Keep one attempt attached to queued transfers, including cancellation."""

    def __init__(self, transfers, trace):
        super().__init__(transfers)
        self.trace = trace


def install_load_trace() -> None:
    """Scope request metadata at both SGLang wrapper entry points on every rank."""
    if not enabled():
        return
    from functools import wraps

    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        UnifiedCacheLinkerWrapper,
    )

    from ucm.integration.sglang.ucm_direct_linker import UcmDirectLinker

    def wrap(original, request_index):
        @wraps(original)
        def traced(self, *args, **kwargs):
            if not isinstance(self.cache_linker, UcmDirectLinker):
                return original(self, *args, **kwargs)
            req = kwargs.get(
                "req", args[request_index] if len(args) > request_index else None
            )
            token = _request.set(req)
            try:
                result = original(self, *args, **kwargs)
                if original.__name__ == "match":
                    emit(
                        "request_match",
                        {
                            "rid": getattr(req, "rid", None),
                            "session_id": getattr(req, "session_id", None),
                            "tp_rank": self.cache_linker._tp_rank,
                        },
                    )
                return result
            finally:
                _request.reset(token)

        traced._ucm_load_trace = True
        return traced

    methods = (("match", 1), ("load_back", 0))
    for name, _ in methods:
        if not callable(getattr(UnifiedCacheLinkerWrapper, name, None)):
            raise RuntimeError(f"SGLang load tracing requires wrapper.{name}")
    for name, index in methods:
        original = getattr(UnifiedCacheLinkerWrapper, name)
        if not getattr(original, "_ucm_load_trace", False):
            setattr(UnifiedCacheLinkerWrapper, name, wrap(original, index))
