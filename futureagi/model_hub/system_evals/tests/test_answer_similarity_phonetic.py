"""
Golden tests for answer_similarity's PhoneticSimilarity comparator.

Soundex only encodes English letters, so numbers and words in other scripts
have to be compared as written rather than collapsing to one shared code.
Soundex codes used below are worked out by hand: the → T000, balance → B452,
is → I200, dollars → D462, Robert / Rupert → R163.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest
import yaml

YAML_PATH = (
    Path(__file__).resolve().parent.parent / "function" / "answer_similarity.yaml"
)


def _phonetic(response, expected_response):
    code = yaml.safe_load(YAML_PATH.read_text())["config"]["code"]
    ns: dict = {}
    vars(builtins)["exec"](compile(code, str(YAML_PATH), "exec"), ns)
    return ns["evaluate"](
        None,
        response,
        expected_response,
        None,
        response=response,
        expected_response=expected_response,
        comparator="PhoneticSimilarity",
    )["score"]


def test_different_numbers_do_not_match():
    # {T000, B452, I200, 500, D462} vs {T000, B452, I200, 900, D462}: 4 shared of 6.
    score = _phonetic("the balance is 500 dollars", "the balance is 900 dollars")
    assert score == pytest.approx(4 / 6)


def test_same_numbers_match():
    assert _phonetic("call 911 now", "call 911 now") == 1.0


def test_unspaced_japanese_is_compared_as_written():
    # One token each, and they differ (晴れ vs 雨): nothing shared.
    assert _phonetic("今日は晴れです", "今日は雨です") == 0.0


@pytest.mark.parametrize(
    "response, expected_response",
    [
        ("Robert", "Rupert"),  # both R163
        ("café", "cafe"),  # accented letters carry no Soundex code: both C100
    ],
)
def test_english_sound_alikes_still_match(response, expected_response):
    assert _phonetic(response, expected_response) == 1.0
