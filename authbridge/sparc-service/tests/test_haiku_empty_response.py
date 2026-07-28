"""Reproduce: IBM LiteLLM proxy + Haiku intermittently returns empty content
when response_format (structured output) mode is used.

Hypothesis: ALTK calls generate_async with schema_field="response_format",
which tells LiteLLM to use structured output mode. The IBM proxy occasionally
returns a response where choices[0].message.content is None or "" and
tool_calls is also empty, causing:
  ValueError: No content or tool calls found in response

This test sends the exact same call N times and counts how many return
empty content — proving the failure is intermittent and tied to response_format.

Requirements:
    pip install litellm python-dotenv
    OAIKEY and OAIBASE must be set in /root/.env or env vars.

Usage:
    python -m pytest tests/test_haiku_empty_response.py -v -s
    # or directly:
    python tests/test_haiku_empty_response.py
"""

from __future__ import annotations

import json
import os
import asyncio
from pathlib import Path

# Load credentials from /root/.env if present
_env_file = Path("/root/.env")
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip().removeprefix("export").strip()
        v = v.strip().strip("\"'")
        os.environ.setdefault(k, v)

API_KEY = os.environ.get("OAIKEY", "")
API_BASE = os.environ.get("OAIBASE", "")
MODEL = "claude-haiku-4-5-20251001"

# The exact schema ALTK sends for general_hallucination_check (from production log)
HALLUCINATION_SCHEMA = {
    "title": "general_hallucination_check",
    "description": "Assessment of tool call grounding accuracy, following the rubric defined in the task description.",
    "type": "object",
    "properties": {
        "evidence": {"type": "string"},
        "explanation": {"type": "string"},
        "output": {"type": "integer", "minimum": 1, "maximum": 5},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "correction": {"type": "object", "additionalProperties": True},
    },
    "required": ["confidence", "correction", "evidence", "explanation", "output"],
    "additionalProperties": False,
}

# Minimal but realistic prompt — matches the structure ALTK sends
PROMPT = [
    {
        "role": "system",
        "content": (
            "Evaluate whether each parameter value in the function call is correct "
            "and directly supported by the provided conversation history.\n\n"
            "Your output must conform to the following JSON schema:\n"
            + json.dumps(HALLUCINATION_SCHEMA)
        ),
    },
    {
        "role": "user",
        "content": (
            "Conversation context:\n"
            '[{"role": "user", "content": "Cancel my flight reservation XEHM4B. Reason: change of plan."}]\n\n'
            "Tool Specification:\n"
            '[{"type": "function", "function": {"name": "cancel_reservation", '
            '"description": "Cancel the whole reservation.", '
            '"parameters": {"type": "object", "properties": '
            '{"reservation_id": {"type": "string"}}, "required": ["reservation_id"]}}}]\n\n'
            'Proposed tool call:\n{"id": "1", "type": "function", "function": '
            '{"name": "cancel_reservation", "arguments": "{\\"reservation_id\\": \\"XEHM4B\\"}"}}\n\n'
            "Return a JSON object as specified in the system prompt."
        ),
    },
]


def _call_with_response_format() -> dict | None:
    """Single synchronous call using response_format — exactly what ALTK does."""
    import litellm

    response = litellm.completion(
        model=MODEL,
        messages=PROMPT,
        api_key=API_KEY,
        api_base=API_BASE,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "general_hallucination_check",
                "schema": HALLUCINATION_SCHEMA,
                "strict": True,
            },
        },
        timeout=30,
    )
    msg = response.choices[0].message
    content = getattr(msg, "content", None)
    tool_calls = getattr(msg, "tool_calls", None)
    return {
        "content": content,
        "tool_calls": tool_calls,
        "empty": not content and not tool_calls,
    }


def _call_with_system_prompt() -> dict | None:
    """Single call with schema injected in system prompt — the proposed fix."""
    import litellm

    response = litellm.completion(
        model=MODEL,
        messages=PROMPT,  # schema already embedded in system prompt above
        api_key=API_KEY,
        api_base=API_BASE,
        timeout=30,
    )
    msg = response.choices[0].message
    content = getattr(msg, "content", None)
    tool_calls = getattr(msg, "tool_calls", None)
    return {
        "content": content,
        "tool_calls": tool_calls,
        "empty": not content and not tool_calls,
    }


def test_response_format_intermittent_empty(n_calls: int = 10):
    """Send the same call N times with response_format and count empty responses.

    If the hypothesis is correct, at least some calls will return empty content,
    proving the IBM proxy is intermittently broken in response_format mode.
    """
    empty_count = 0
    results = []

    for i in range(n_calls):
        result = _call_with_response_format()
        results.append(result)
        status = "EMPTY" if result["empty"] else "OK"
        print(f"  call {i+1:02d}: {status}  content_len={len(result['content'] or '')}")
        if result["empty"]:
            empty_count += 1

    print(f"\nSummary: {empty_count}/{n_calls} calls returned empty content")

    # The test proves the hypothesis if at least 1 call is empty.
    # If 0 are empty, the IBM proxy may have been fixed or the prompt is too simple.
    assert empty_count > 0, (
        f"All {n_calls} calls succeeded — IBM proxy may be stable now, "
        f"or the prompt needs to be longer to trigger the failure."
    )


def test_system_prompt_mode_no_empty(n_calls: int = 10):
    """Same N calls but with schema in system prompt instead of response_format.

    If the fix works, zero calls should return empty content.
    """
    empty_count = 0

    for i in range(n_calls):
        result = _call_with_system_prompt()
        status = "EMPTY" if result["empty"] else "OK"
        print(f"  call {i+1:02d}: {status}  content_len={len(result['content'] or '')}")
        if result["empty"]:
            empty_count += 1

    print(f"\nSummary: {empty_count}/{n_calls} calls returned empty content")

    assert empty_count == 0, (
        f"{empty_count}/{n_calls} calls returned empty content even with system-prompt mode"
    )


if __name__ == "__main__":
    if not API_KEY or not API_BASE:
        print("ERROR: OAIKEY and OAIBASE must be set in /root/.env or environment")
        raise SystemExit(1)

    N = 10
    print(f"=== Test 1: response_format mode ({N} calls) ===")
    empty = 0
    for i in range(N):
        r = _call_with_response_format()
        status = "EMPTY" if r["empty"] else f"OK ({len(r['content'] or '')} chars)"
        print(f"  call {i+1:02d}: {status}")
        if r["empty"]:
            empty += 1
    print(f"Result: {empty}/{N} empty\n")

    print(f"=== Test 2: system-prompt mode ({N} calls) ===")
    empty = 0
    for i in range(N):
        r = _call_with_system_prompt()
        status = "EMPTY" if r["empty"] else f"OK ({len(r['content'] or '')} chars)"
        print(f"  call {i+1:02d}: {status}")
        if r["empty"]:
            empty += 1
    print(f"Result: {empty}/{N} empty")
