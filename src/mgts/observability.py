"""Логи запуска и измерение длительности этапов."""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, MutableMapping
from uuid import uuid4

_LOG_CONTEXT: ContextVar[dict[str, str]] = ContextVar("mgts_log_context", default={})
_STANDARD_RECORD_FIELDS = set(logging.makeLogRecord({}).__dict__)


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        context = _LOG_CONTEXT.get()
        for field in ("run_id", "command", "stage", "frame_id"):
            if not hasattr(record, field):
                setattr(record, field, context.get(field, "-"))
        excluded = _STANDARD_RECORD_FIELDS | {"run_id", "command", "stage", "frame_id", "details"}
        details = {key: value for key, value in record.__dict__.items() if key not in excluded}
        record.details = json.dumps(details, ensure_ascii=False, default=str, sort_keys=True) if details else ""
        return True


def configure_logging() -> None:
    """Включить человекочитаемые логи; уровень задаётся MGTS_LOG_LEVEL."""
    root = logging.getLogger()
    level = getattr(logging, os.getenv("MGTS_LOG_LEVEL", "INFO").upper(), logging.INFO)
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    for handler in root.handlers:
        if not any(isinstance(item, _ContextFilter) for item in handler.filters):
            handler.addFilter(_ContextFilter())
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s run=%(run_id)s cmd=%(command)s stage=%(stage)s frame=%(frame_id)s %(message)s %(details)s"
        ))


def new_run_id() -> str:
    return uuid4().hex[:12]


def current_run_id() -> str | None:
    return _LOG_CONTEXT.get().get("run_id")


@contextmanager
def log_context(**fields: str) -> Iterator[None]:
    token = _LOG_CONTEXT.set({**_LOG_CONTEXT.get(), **fields})
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


@contextmanager
def trace_span(
    logger: logging.Logger,
    stage: str,
    *,
    level: int = logging.INFO,
    timings: MutableMapping[str, float] | None = None,
    timing_key: str | None = None,
    log_traceback: bool = True,
    **fields: object,
) -> Iterator[None]:
    """Записать начало/итог этапа; при ошибке — traceback и пробросить исключение."""
    extra = {**_LOG_CONTEXT.get(), "stage": stage, **fields}
    started = time.perf_counter()
    logger.log(level, "stage.start", extra=extra)
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - started
        if timings is not None and timing_key:
            timings[timing_key] = timings.get(timing_key, 0.0) + elapsed
        if log_traceback:
            logger.exception("stage.failed duration_s=%.3f", elapsed, extra=extra)
        else:
            logger.error("stage.failed duration_s=%.3f", elapsed, extra=extra)
        raise
    else:
        elapsed = time.perf_counter() - started
        if timings is not None and timing_key:
            timings[timing_key] = timings.get(timing_key, 0.0) + elapsed
        logger.log(level, "stage.completed duration_s=%.3f", elapsed, extra=extra)
