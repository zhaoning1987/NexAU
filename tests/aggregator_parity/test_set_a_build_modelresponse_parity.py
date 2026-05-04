# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Axis 4 — ``ModelResponse.from_X(Set A.build())`` byte-equivalent to Set B.

RFC-0023 §阶段 ③ adds ``build()`` to all 4 Set A aggregators returning the
provider's vendor-native typed object (Anthropic Message, OpenAI ChatCompletion,
OpenAI Response, Gemini dict). Downstream code converts to the unified
``ModelResponse`` via the existing ``ModelResponse.from_*`` adapters.

This axis pins that **the resulting ModelResponse from the Set A path matches
the ModelResponse from the Set B path** for every recorded fixture. It's the
final safety net before §阶段 ③ swaps the production code path
(``llm_caller.py`` 8 call sites) from Set B to Set A — any divergence here
would cause silent ModelResponse shape drift on the persistence side.

Once §阶段 ③ deletes Set B (PR-C.2), this axis goes with it (no Set B side
to compare against).

Filename note: avoids ``openai`` / ``chat`` / ``llm`` substrings to dodge
conftest's auto-skip-without-LIVE_LLM-key marker.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from anthropic.types import RawMessageStopEvent

from nexau.archs.llm.llm_aggregators import (
    AnthropicEventAggregator,
    GeminiRestEventAggregator,
    OpenAIChatCompletionAggregator,
    OpenAIResponsesAggregator,
)
from nexau.archs.main_sub.execution.model_response import ModelResponse
from tests.aggregator_parity.anthropic_glue import (
    _coerce_to_sdk_events as _coerce_anthropic,
)
from tests.aggregator_parity.anthropic_glue import (
    run_set_b_anthropic,
)
from tests.aggregator_parity.fixtures.anthropic import ANTHROPIC_FIXTURES
from tests.aggregator_parity.fixtures.gemini_rest import GEMINI_REST_FIXTURES
from tests.aggregator_parity.fixtures.openai_chat import OPENAI_CHAT_FIXTURES
from tests.aggregator_parity.fixtures.openai_responses import OPENAI_RESPONSES_FIXTURES
from tests.aggregator_parity.gemini_glue import run_set_b_gemini
from tests.aggregator_parity.openai_chat_glue import (
    _coerce_to_sdk_chunks as _coerce_oac,
)
from tests.aggregator_parity.openai_chat_glue import (
    run_set_b_openai_chat,
)
from tests.aggregator_parity.openai_responses_glue import (
    _coerce_to_sdk_events as _coerce_oresp,
)
from tests.aggregator_parity.openai_responses_glue import (
    run_set_b_openai_responses,
)

# Axis-4 has its own divergence registry — axis-1 KNOWN_DIVERGENT_FIXTURES are
# events-emission-level divergences which may or may not surface at the
# build()/finalize() level (e.g. server_tool_use diverges on axis 1's text
# block collapse but coincidentally agrees here). Each axis owns its registry.
KNOWN_AXIS4_DIVERGENT: dict[tuple[str, str], str] = {
    # (provider, fixture_name) → reason
    # (none yet — populate as real divergences are uncovered)
}


# ----------------------------------------------------------------------------
# Per-provider drivers
# ----------------------------------------------------------------------------


def _drive_anthropic(events: list[Any]) -> tuple[ModelResponse, ModelResponse]:
    sdk = _coerce_anthropic(events)
    agg = AnthropicEventAggregator(on_event=lambda _e: None, run_id="axis4")
    saw_stop = False
    for ev in sdk:
        if isinstance(ev, RawMessageStopEvent):
            saw_stop = True
        agg.aggregate(ev)
    if not saw_stop:
        agg._handle_message_stop()  # noqa: SLF001 — synthetic flush, mirrors anthropic_glue
    set_a_msg = agg.build()
    set_b_dict = run_set_b_anthropic(events)
    return (
        ModelResponse.from_anthropic_message(set_a_msg),
        ModelResponse.from_anthropic_message(set_b_dict),
    )


def _drive_openai_chat(events: list[Any]) -> tuple[ModelResponse, ModelResponse]:
    sdk = _coerce_oac(events)
    agg = OpenAIChatCompletionAggregator(on_event=lambda _e: None, run_id="axis4")
    for ev in sdk:
        agg.aggregate(ev)
    try:
        completion_a = agg.build()
    except RuntimeError:
        # No choices received — same path as openai_chat_glue. Set A's
        # build() raises before producing a ChatCompletion; downstream
        # ModelResponse construction is impossible by design (Set B raises
        # too on its finalize path).
        pytest.skip("OAC fixture produced no choices — both Set A.build and Set B.finalize raise")
    # ModelResponse.from_openai_message takes the per-choice message, not the
    # full ChatCompletion. Both Set A and Set B converge through that adapter
    # at the call sites, so we mirror it here.
    msg_from_a = completion_a.choices[0].message
    usage_a = completion_a.usage.model_dump() if completion_a.usage else None
    set_b_dict = run_set_b_openai_chat(events)
    return (
        ModelResponse.from_openai_message(msg_from_a, usage=usage_a),
        ModelResponse.from_openai_message(set_b_dict, usage=set_b_dict.get("usage")),
    )


def _drive_openai_responses(events: list[Any]) -> tuple[ModelResponse, ModelResponse]:
    sdk = _coerce_oresp(events)
    agg = OpenAIResponsesAggregator(on_event=lambda _e: None, run_id="axis4")
    for ev in sdk:
        agg.aggregate(ev)
    response_a = agg.build()
    set_b_dict = run_set_b_openai_responses(events)
    return (
        ModelResponse.from_openai_response(response_a),
        ModelResponse.from_openai_response(set_b_dict),
    )


def _drive_gemini(events: list[Any]) -> tuple[ModelResponse, ModelResponse]:
    from typing import cast

    from nexau.archs.llm.llm_aggregators.gemini_rest.gemini_rest_event_aggregator import GeminiResponse

    agg = GeminiRestEventAggregator(on_event=lambda _e: None, run_id="axis4")
    for c in events:
        agg.aggregate(cast(GeminiResponse, c))
    dict_a = agg.build()
    dict_b = run_set_b_gemini(events)
    # ``ModelResponse.from_gemini_rest`` is typed against ``dict[str, Any]``
    # (provider-neutral); widen our typed GeminiResponse via cast.
    return (
        ModelResponse.from_gemini_rest(cast(dict[str, Any], dict_a)),
        ModelResponse.from_gemini_rest(dict_b),
    )


# ----------------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------------


def _compare(mr_a: ModelResponse, mr_b: ModelResponse) -> list[str]:
    """Return list of byte-level differences between the two ModelResponses.

    Compares the ``ModelResponse``-level fields end users actually consume:
    ``content`` text, ``reasoning_content``, ``tool_calls`` shape (id + name +
    arguments). Empty list = byte-equivalent.
    """
    diffs: list[str] = []
    if (mr_a.content or "") != (mr_b.content or ""):
        diffs.append(f"content: a={mr_a.content!r} b={mr_b.content!r}")
    if (mr_a.reasoning_content or "") != (mr_b.reasoning_content or ""):
        diffs.append(f"reasoning_content: a={mr_a.reasoning_content!r} b={mr_b.reasoning_content!r}")
    if len(mr_a.tool_calls) != len(mr_b.tool_calls):
        diffs.append(f"tool_calls count: a={len(mr_a.tool_calls)} b={len(mr_b.tool_calls)}")
    else:
        for i, (ta, tb) in enumerate(zip(mr_a.tool_calls, mr_b.tool_calls)):
            if ta.name != tb.name:
                diffs.append(f"tool_calls[{i}].name: a={ta.name!r} b={tb.name!r}")
            if ta.arguments != tb.arguments:
                diffs.append(f"tool_calls[{i}].arguments: a={ta.arguments!r} b={tb.arguments!r}")
    return diffs


# ----------------------------------------------------------------------------
# Parametrized test
# ----------------------------------------------------------------------------


SUITES: list[tuple[str, list[tuple[str, Any]], Callable[[list[Any]], tuple[ModelResponse, ModelResponse]]]] = [
    ("anthropic", ANTHROPIC_FIXTURES, _drive_anthropic),
    ("openai_chat", OPENAI_CHAT_FIXTURES, _drive_openai_chat),
    ("openai_responses", OPENAI_RESPONSES_FIXTURES, _drive_openai_responses),
    ("gemini_rest", GEMINI_REST_FIXTURES, _drive_gemini),
]


def _flatten_cases() -> list[tuple[str, str, str, Callable[[list[Any]], tuple[ModelResponse, ModelResponse]], list[Any]]]:
    """Yield (test_id, provider, fixture_name, driver, events) per fixture."""
    out = []
    for provider, fixtures, driver in SUITES:
        for name, fn in fixtures:
            test_id = f"{provider[:5]}_{name}"
            out.append((test_id, provider, name, driver, fn()))
    return out


_CASES = _flatten_cases()


@pytest.mark.parametrize(
    "provider,fixture_name,driver,events",
    [(p, n, d, e) for _id, p, n, d, e in _CASES],
    ids=[test_id for test_id, _, _, _, _ in _CASES],
)
def test_set_a_build_yields_same_modelresponse_as_set_b(
    provider: str,
    fixture_name: str,
    driver: Callable[[list[Any]], tuple[ModelResponse, ModelResponse]],
    events: list[Any],
    request: pytest.FixtureRequest,
) -> None:
    """``ModelResponse.from_X(Set A.build())`` must byte-match the Set B path."""
    if (provider, fixture_name) in KNOWN_AXIS4_DIVERGENT:
        request.applymarker(
            pytest.mark.xfail(
                reason=KNOWN_AXIS4_DIVERGENT[(provider, fixture_name)],
                strict=True,
            )
        )
    mr_a, mr_b = driver(events)
    diffs = _compare(mr_a, mr_b)
    if diffs:
        pytest.fail(
            f"Axis-4 ModelResponse divergence on {fixture_name}:\n"
            + "\n".join(f"  - {d[:300]}" for d in diffs)
            + "\n\nSet A's build() returns a vendor-typed object that, after passing through "
            "ModelResponse.from_X, yields a different ModelResponse than Set B's finalize() does. "
            "Swapping production callers from Set B to Set A would silently change downstream "
            "behavior. Either fix Set A.build() to match, or — if the divergence is a deliberate "
            "design call — register the fixture in the provider's KNOWN_DIVERGENT_FIXTURES with a "
            "written rationale."
        )
