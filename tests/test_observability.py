from __future__ import annotations

import json
import logging

import pytest

from mgts.observability import _ContextFilter, log_context, trace_span


def test_trace_span_logs_context_and_elapsed_time(caplog):
    logger = logging.getLogger("mgts.test.observability")
    timings = {}

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        with log_context(run_id="run-123", command="run", frame_id="frame-0042"):
            with trace_span(logger, "frame.inference", level=logging.DEBUG, timings=timings,
                            timing_key="inference", detections=2):
                pass

    records = [record for record in caplog.records if record.name == logger.name]
    assert [record.getMessage().split()[0] for record in records] == ["stage.start", "stage.completed"]
    assert all(record.run_id == "run-123" and record.frame_id == "frame-0042" for record in records)
    assert records[0].stage == "frame.inference" and records[0].detections == 2
    assert timings["inference"] >= 0


def test_trace_span_logs_traceback_and_timing_on_failure(caplog):
    logger = logging.getLogger("mgts.test.observability.failure")
    timings = {}

    with caplog.at_level(logging.ERROR, logger=logger.name), pytest.raises(ValueError, match="broken stage"):
        with trace_span(logger, "pipeline.export", timings=timings, timing_key="export"):
            raise ValueError("broken stage")

    failure = next(record for record in caplog.records if record.getMessage().startswith("stage.failed"))
    assert failure.exc_info and failure.exc_info[0] is ValueError
    assert failure.stage == "pipeline.export" and timings["export"] >= 0


def test_context_filter_formats_custom_fields_as_json():
    logger = logging.getLogger("mgts.test.observability.format")
    record = logger.makeRecord(logger.name, logging.INFO, __file__, 1, "stage.completed", (), None,
                               extra={"run_id": "run-123", "command": "run", "stage": "frame.inference",
                                      "frame_id": "frame-0042", "detections": 2})

    assert _ContextFilter().filter(record)
    assert json.loads(record.details) == {"detections": 2}
