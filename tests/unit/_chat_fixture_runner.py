# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tiny helper that loads a synthetic OpenAI Chat Completion SSE fixture
from ``tests/unit/fixtures/openai_chat/`` and replays it through
``OpenAIChatCompletionAggregator``.

Why this exists
---------------
Most aggregator unit tests construct ``ChatCompletionChunk(...)`` literals
inline — fine for a one-off, but the boilerplate dwarfs the actual
behaviour under test. This loader lets a test point at a small ``.sse``
file (committed alongside the tests) and assert on the emitted Event
list + built ChatCompletion. The fixture format mirrors the live
recordings used by ``tests/aggregator_parity/`` so the SSE parser is
shared (no second implementation to drift).

Use this when the test is "feed a chunk sequence in, observe events" —
i.e. anything you could capture from a real provider stream. Keep
inline-chunk tests for cases that mutate aggregator internal state
directly.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from nexau.archs.llm.llm_aggregators import (
    Event,
    OpenAIChatCompletionAggregator,
)
from tests.aggregator_parity.sse_loader import _parse_sse_blocks

_FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures" / "openai_chat"


def run_chat_fixture(
    scenario: str,
    *,
    on_event: Callable[[Event], None] | None = None,
    run_id: str = "test-run",
) -> tuple[list[Event], ChatCompletion]:
    """Replay ``tests/unit/fixtures/openai_chat/<scenario>.sse`` through
    :class:`OpenAIChatCompletionAggregator`.

    Returns ``(emitted_events, built_response)``. Events are captured in
    emission order so callers can assert on START → ARGS → END
    ordering, count occurrences, etc.

    Parameters
    ----------
    scenario:
        File stem under ``fixtures/openai_chat/``. The file must be SSE
        with one ``data: <json>`` line per chunk, terminated by
        ``data: [DONE]`` (lines starting with ``#`` are silently
        skipped — useful for inline scenario commentary).
    on_event:
        Optional secondary observer (e.g. a Mock) — fires alongside the
        captured list, not in place of it.
    run_id:
        Run id passed to the aggregator constructor; defaults to
        ``"test-run"`` since most tests don't care about its value.
    """
    path = _FIXTURES_ROOT / f"{scenario}.sse"
    if not path.is_file():
        raise FileNotFoundError(f"Synthetic fixture not found: {path}")
    raw = path.read_text(encoding="utf-8")
    chunk_dicts = _parse_sse_blocks(raw)
    chunks = [ChatCompletionChunk.model_validate(d) for d in chunk_dicts]

    captured: list[Event] = []

    def _capture(event: Event) -> None:
        captured.append(event)
        if on_event is not None:
            on_event(event)

    aggregator = OpenAIChatCompletionAggregator(on_event=_capture, run_id=run_id)
    for chunk in chunks:
        aggregator.aggregate(chunk)
    response = aggregator.build()
    return captured, response
