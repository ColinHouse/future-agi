"""Tests for strip_code_fence — the shared Markdown-fence unwrapper — and
extract_dict_from_string, which parses LLM-judge verdicts."""

import pytest

from agentic_eval.core.utils.json_utils import (
    extract_dict_from_string,
    strip_code_fence,
)


def test_multiline_json_fence():
    assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_bare_fence_no_language():
    assert strip_code_fence('```\n{"a": 1}\n```') == '{"a": 1}'


def test_single_line_fence_is_not_dropped():
    # Regression: a naive line-split returns "" here, dropping the payload.
    assert strip_code_fence('```json{"a": 1}```') == '{"a": 1}'


def test_no_fence_returned_unchanged():
    assert strip_code_fence('{"a": 1}') == '{"a": 1}'


def test_truncated_opening_fence():
    assert strip_code_fence('```json\n{"a": 1}') == '{"a": 1}'


def test_none_and_empty_are_safe():
    assert strip_code_fence(None) == ""
    assert strip_code_fence("") == ""


def test_multiline_body_preserved():
    text = "```\nline1\nline2\n```"
    assert strip_code_fence(text) == "line1\nline2"


@pytest.mark.parametrize(
    "verdict, expected",
    [
        # LaTeX in an explanation: \\t and \\r are escaped backslashes, not tabs.
        (
            r'{"result": "Pass", "explanation": "uses \\times and \\rho"}',
            {"result": "Pass", "explanation": "uses \\times and \\rho"},
        ),
        (
            r'{"result": "Fail", "explanation": "wrote C:\\temp\\results.txt"}',
            {"result": "Fail", "explanation": "wrote C:\\temp\\results.txt"},
        ),
        # A real \t escape is a tab.
        (
            r'{"result": "Pass", "explanation": "col1\tcol2"}',
            {"result": "Pass", "explanation": "col1\tcol2"},
        ),
    ],
)
def test_valid_judge_json_keeps_its_escapes(verdict, expected):
    assert extract_dict_from_string(f"My verdict:\n{verdict}") == expected


def test_raw_newline_inside_a_string_is_still_repaired():
    # Not valid JSON (a control character inside a string): the cleanup
    # drops the newline so the verdict still parses.
    verdict = '{"result": "Pass", "explanation": "line one\nline two"}'
    assert extract_dict_from_string(verdict) == {
        "result": "Pass",
        "explanation": "line oneline two",
    }


def test_single_quoted_verdict_is_still_repaired():
    assert extract_dict_from_string("{'result': 'Pass'}") == {"result": "Pass"}
