# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Synthetic-input tests for aggregator code paths that real provider
streams don't exercise.

Three categories of paths covered here (each has good reasons to be
exercised by synthetic input rather than recordings):

1. **`clear()` reset methods** — designed for aggregator reuse across
   multiple streams. Not exercised by parity tests (which create a fresh
   aggregator per fixture). These tests just call clear() to bump
   coverage and assert state actually resets.

2. **`build()` / `finalize()` error paths** — RuntimeError raised when
   no chunks were ever consumed. Real streams always have at least one
   chunk, so this path is otherwise dead. Defensive-by-design.

3. **Eager-streaming pathology** — Anthropic's
   ``_pending_tool_deltas`` / ``_flush_pending_with_synthetic`` machinery
   that triggers when ``input_json_delta`` arrives BEFORE
   ``content_block_start``. Real Anthropic API rarely does this in the
   wild (only certain provider routings); the gateway emulator masks
   it with the "duplicate content_block_start" pathology. We synthesize
   the case here to lock in the recovery behavior.

These are NOT parity tests — they exercise Set A (or Set B) in isolation
to ensure defensive code paths still work as documented.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from anthropic.types import (
    InputJSONDelta,
    Message,
    RawContentBlockDeltaEvent,
    RawContentBlockStopEvent,
    RawMessageStartEvent,
    Usage,
)

from nexau.archs.llm.llm_aggregators import (
    AnthropicEventAggregator,
    GeminiRestEventAggregator,
    OpenAIChatCompletionAggregator,
    OpenAIResponsesAggregator,
)
from nexau.archs.main_sub.execution.llm_caller import (
    AnthropicStreamAggregator,
    GeminiRestStreamAggregator,
    OpenAIChatStreamAggregator,
)

# ============================================================================
# (1) clear() reset methods — Set A aggregators
# ============================================================================


def test_anthropic_clear_resets_state() -> None:
    """AnthropicEventAggregator.clear() must reset all internal state."""
    agg = AnthropicEventAggregator(on_event=Mock(), run_id="run-1")
    # Feed a message_start to populate state
    agg.aggregate(
        RawMessageStartEvent(
            type="message_start",
            message=Message(
                id="msg_x",
                type="message",
                role="assistant",
                content=[],
                model="claude-3",
                stop_reason=None,
                stop_sequence=None,
                usage=Usage(input_tokens=1, output_tokens=0),
            ),
        )
    )
    assert agg._message_id == "msg_x"
    assert agg._started is True

    agg.clear()
    assert agg._message_id == ""
    assert agg._started is False
    assert not agg._block_types
    assert not agg._tool_ids


def test_oac_clear_resets_state() -> None:
    """OpenAIChatCompletionAggregator.clear() resets state for reuse."""
    agg = OpenAIChatCompletionAggregator(on_event=Mock(), run_id="run-1")
    # No need to feed anything — just verify clear() runs and resets _value
    agg.clear()
    assert agg._value.id == ""
    assert agg._value.choices == []
    assert not agg._choice_aggregators


def test_responses_clear_resets_state() -> None:
    """OpenAIResponsesAggregator.clear() resets state for reuse."""
    agg = OpenAIResponsesAggregator(on_event=Mock(), run_id="run-1")
    agg.clear()
    assert agg._value.id == ""
    assert agg._value.output == []
    assert not agg._output_aggregators


def test_gemini_clear_resets_state() -> None:
    """GeminiRestEventAggregator.clear() resets state for reuse."""
    agg = GeminiRestEventAggregator(on_event=Mock(), run_id="run-1")
    # Populate some state
    agg.aggregate({"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})
    assert agg._started

    agg.clear()
    assert agg._started is False
    assert agg._text_started is False
    assert agg._thinking_started is False


# ============================================================================
# (2) build() / finalize() error paths — Set A aggregators on empty input
# ============================================================================


def test_oac_build_raises_on_empty_stream() -> None:
    """OpenAIChatCompletionAggregator.build() raises RuntimeError if no
    valid chunks were consumed (no choice aggregators registered)."""
    agg = OpenAIChatCompletionAggregator(on_event=Mock(), run_id="run-1")
    with pytest.raises(RuntimeError, match="did not receive any valid chunks"):
        agg.build()


# ============================================================================
# (2b) finalize() error paths — Set B aggregators
# ============================================================================


def test_anthropic_set_b_finalize_raises_on_empty_stream() -> None:
    """AnthropicStreamAggregator.finalize() raises if no chunks consumed."""
    agg = AnthropicStreamAggregator()
    with pytest.raises(RuntimeError, match="No stream chunks were received"):
        agg.finalize()


def test_gemini_set_b_finalize_raises_on_empty_stream() -> None:
    """GeminiRestStreamAggregator.finalize() raises if no chunks consumed."""
    agg = GeminiRestStreamAggregator()
    with pytest.raises(RuntimeError, match="No stream chunks were received"):
        agg.finalize()


# ============================================================================
# (3) Anthropic eager-streaming pathology: input_json_delta before content_block_start
# ============================================================================


def test_anthropic_eager_streaming_input_json_delta_before_start() -> None:
    """Anthropic Set A's _pending_tool_deltas / _flush_pending_with_synthetic.

    When ``input_json_delta`` arrives at index N before any
    ``content_block_start`` for that index, AnthropicEventAggregator
    buffers the fragment in ``_pending_tool_deltas[N]``. When the start
    eventually arrives (with id/name), _register_tool_and_flush emits
    Start + Args (replaying buffered fragments) + ... in proper order.

    Synthesizes this pathology and asserts:
    1. No events are emitted while the delta is pending
    2. When start arrives, ToolCallStart fires with the real id/name
    3. Buffered fragment is replayed via ToolCallArgs

    Real production streams rarely do this; the parity recordings from
    the gateway showed the OPPOSITE pathology (duplicate content_block_start
    with empty id/name), so this code path was otherwise uncovered.
    """
    events: list = []
    agg = AnthropicEventAggregator(on_event=events.append, run_id="run-1")

    agg.aggregate(
        RawMessageStartEvent(
            type="message_start",
            message=Message(
                id="msg_eager",
                type="message",
                role="assistant",
                content=[],
                model="claude-3",
                stop_reason=None,
                stop_sequence=None,
                usage=Usage(input_tokens=1, output_tokens=0),
            ),
        )
    )

    # Delta arrives BEFORE any content_block_start at the same index
    agg.aggregate(
        RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=InputJSONDelta(type="input_json_delta", partial_json='{"loc":"BJ"}'),
        )
    )

    # No tool events emitted yet (buffered)
    tool_starts_pre = [e for e in events if type(e).__name__ == "ToolCallStartEvent"]
    assert tool_starts_pre == []
    assert agg._pending_tool_deltas.get(0) == ['{"loc":"BJ"}']

    # content_block_stop arrives WITHOUT preceding content_block_start at all
    # → triggers _flush_pending_with_synthetic, which synthesizes a tool id
    agg.aggregate(RawContentBlockStopEvent(type="content_block_stop", index=0))

    # Now ToolCallStart should have fired with synthetic id, plus the
    # buffered fragment as ToolCallArgs, plus ToolCallEnd
    tool_starts = [e for e in events if type(e).__name__ == "ToolCallStartEvent"]
    tool_args = [e for e in events if type(e).__name__ == "ToolCallArgsEvent"]
    tool_ends = [e for e in events if type(e).__name__ == "ToolCallEndEvent"]
    assert len(tool_starts) == 1
    assert tool_starts[0].tool_call_id.startswith("toolu_late_")
    assert tool_starts[0].tool_call_name == ""  # synthetic name is empty
    assert len(tool_args) == 1
    assert tool_args[0].delta == '{"loc":"BJ"}'
    assert len(tool_ends) == 1


def test_anthropic_eager_streaming_buffered_then_real_start() -> None:
    """Variant: delta buffered, then a REAL content_block_start arrives.

    Tests the alternate path where _flush_pending_with_synthetic is
    NOT called — instead _register_tool_and_flush replays the buffer
    with the real id/name from the late start event.
    """
    from anthropic.types import RawContentBlockStartEvent
    from anthropic.types import ToolUseBlock as AnthropicToolUseBlock

    events: list = []
    agg = AnthropicEventAggregator(on_event=events.append, run_id="run-1")

    agg.aggregate(
        RawMessageStartEvent(
            type="message_start",
            message=Message(
                id="msg_eager2",
                type="message",
                role="assistant",
                content=[],
                model="claude-3",
                stop_reason=None,
                stop_sequence=None,
                usage=Usage(input_tokens=1, output_tokens=0),
            ),
        )
    )

    # Delta arrives first
    agg.aggregate(
        RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=InputJSONDelta(type="input_json_delta", partial_json='{"x":1}'),
        )
    )
    assert agg._pending_tool_deltas.get(0) == ['{"x":1}']

    # Real start arrives late
    agg.aggregate(
        RawContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=AnthropicToolUseBlock(type="tool_use", id="toolu_real", name="get_x", input={}),
        )
    )

    # _register_tool_and_flush replays buffered fragments with real id/name
    tool_starts = [e for e in events if type(e).__name__ == "ToolCallStartEvent"]
    tool_args = [e for e in events if type(e).__name__ == "ToolCallArgsEvent"]
    assert len(tool_starts) == 1
    assert tool_starts[0].tool_call_id == "toolu_real"
    assert tool_starts[0].tool_call_name == "get_x"
    assert len(tool_args) == 1
    assert tool_args[0].delta == '{"x":1}'
    # Buffer drained
    assert 0 not in agg._pending_tool_deltas


# ============================================================================
# (4) Set B reset: AnthropicStreamAggregator's accumulator state lifecycle
# ============================================================================


def test_anthropic_set_b_handles_unknown_block_type() -> None:
    """Anthropic Set B should handle unknown block types gracefully.

    Tests the defensive branch where a content_block_start arrives with
    an unrecognized block type — Set B should not crash; it just keeps
    going (real future block types fall through this path).
    """
    agg = AnthropicStreamAggregator()
    agg.consume({"type": "message_start", "message": {"role": "assistant", "model": "claude-3"}})
    agg.consume(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "future_unknown_block_type", "data": "..."},
        }
    )
    agg.consume({"type": "content_block_stop", "index": 0})
    agg.consume({"type": "message_stop"})
    # Add a valid text block so finalize doesn't raise
    agg.consume({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}})
    agg.consume({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "ok"}})
    agg.consume({"type": "content_block_stop", "index": 1})
    result = agg.finalize()
    assert result["role"] == "assistant"


# ============================================================================
# (5) OpenAI Chat Set B variant: usage-only chunk + reasoning_content edge cases
# ============================================================================


def test_oac_set_b_usage_only_chunk() -> None:
    """OpenAIChatStreamAggregator.consume should accept usage-only chunks
    (no choices, just usage). Real wire format puts usage in a final chunk."""
    agg = OpenAIChatStreamAggregator()
    # Initial chunk with content
    agg.consume(
        {
            "model": "gpt-4o",
            "choices": [{"delta": {"role": "assistant", "content": "Hi"}}],
        }
    )
    # Usage-only terminal chunk
    agg.consume(
        {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "total_tokens": 11,
            },
        }
    )
    result = agg.finalize()
    assert result["content"] == "Hi"
    assert result["usage"]["prompt_tokens"] == 10


def test_oac_set_b_clear() -> None:
    """OpenAIChatStreamAggregator state is per-instance; verify a fresh
    instance has clean state independently of any prior one."""
    agg1 = OpenAIChatStreamAggregator()
    agg1.consume({"choices": [{"delta": {"role": "assistant", "content": "Hi"}}]})
    assert agg1.finalize()["content"] == "Hi"

    agg2 = OpenAIChatStreamAggregator()
    agg2.consume({"choices": [{"delta": {"role": "assistant", "content": "Bye"}}]})
    assert agg2.finalize()["content"] == "Bye"
    assert agg1.finalize()["content"] == "Hi"


# ============================================================================
# (6) Refusal content_part path — modern models tend to return text-form
# refusals; the structured ``refusal`` content_part code path is otherwise
# uncovered by recordings.
# ============================================================================


def test_oac_set_a_handles_refusal_delta() -> None:
    """OpenAIChatCompletionAggregator handles ``delta.refusal`` chunks."""
    from openai.types.chat import ChatCompletionChunk

    events: list = []
    agg = OpenAIChatCompletionAggregator(on_event=events.append, run_id="run-1")
    agg.aggregate(
        ChatCompletionChunk.model_validate(
            {
                "id": "chatcmpl-x",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "gpt-x",
                "choices": [{"index": 0, "delta": {"role": "assistant", "refusal": "I cannot"}, "finish_reason": None}],
            }
        )
    )
    agg.aggregate(
        ChatCompletionChunk.model_validate(
            {
                "id": "chatcmpl-x",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "gpt-x",
                "choices": [{"index": 0, "delta": {"refusal": " help with that"}, "finish_reason": "stop"}],
            }
        )
    )
    result = agg.build()
    assert result.choices[0].message.refusal == "I cannot help with that"


def test_oac_set_a_handles_logprobs() -> None:
    """OpenAIChatCompletionAggregator accumulates logprobs across chunks."""
    from openai.types.chat import ChatCompletionChunk

    events: list = []
    agg = OpenAIChatCompletionAggregator(on_event=events.append, run_id="run-1")
    agg.aggregate(
        ChatCompletionChunk.model_validate(
            {
                "id": "chatcmpl-l",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "gpt-x",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hi"},
                        "logprobs": {"content": [{"token": "Hi", "logprob": -0.1, "top_logprobs": []}]},
                        "finish_reason": None,
                    }
                ],
            }
        )
    )
    agg.aggregate(
        ChatCompletionChunk.model_validate(
            {
                "id": "chatcmpl-l",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "gpt-x",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " there"},
                        "logprobs": {"content": [{"token": " there", "logprob": -0.2, "top_logprobs": []}]},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
    )
    result = agg.build()
    assert result.choices[0].message.content == "Hi there"
    logprobs = result.choices[0].logprobs
    assert logprobs is not None
    assert logprobs.content is not None
    assert len(logprobs.content) == 2


# ============================================================================
# (7) Gemini chunk shape edge cases — defensive early-return paths
# ============================================================================


def test_gemini_handles_chunk_without_candidates() -> None:
    """Gemini Set A/B both gracefully skip chunks with no candidates."""
    agg_a = GeminiRestEventAggregator(on_event=Mock(), run_id="run-1")
    agg_a.aggregate({})
    agg_a.aggregate({"usageMetadata": {"promptTokenCount": 5}})
    agg_a.aggregate({"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})

    agg_b = GeminiRestStreamAggregator()
    agg_b.consume({})
    agg_b.consume({"usageMetadata": {"promptTokenCount": 5}})
    agg_b.consume({"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})
    result = agg_b.finalize()
    assert "hi" in str(result)


def test_gemini_handles_chunk_without_content() -> None:
    """Gemini chunk with finishReason but no content (final-chunk pattern)."""
    agg = GeminiRestEventAggregator(on_event=Mock(), run_id="run-1")
    agg.aggregate({"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})
    agg.aggregate({"candidates": [{"finishReason": "STOP"}]})


def test_gemini_handles_malformed_parts() -> None:
    """Gemini parts that are not dicts / non-list parts are skipped.

    These payloads deliberately violate ``GeminiResponse`` to exercise
    the aggregator's defensive runtime guards; cast to satisfy mypy.
    """
    from typing import cast

    from nexau.archs.llm.llm_aggregators.gemini_rest.gemini_rest_event_aggregator import GeminiResponse

    agg = GeminiRestEventAggregator(on_event=Mock(), run_id="run-1")
    agg.aggregate(cast(GeminiResponse, {"candidates": [{"content": {"parts": ["not_a_dict", {"text": "hi"}]}}]}))
    agg.aggregate(cast(GeminiResponse, {"candidates": [{"content": {"parts": "not-a-list"}}]}))
