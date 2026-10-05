"""
Golden tests for the ``normalize`` / ``tokenize`` options of the WER/CER family.

The defaults must keep today's scores. ``normalize: basic`` and
``tokenize: auto | char`` stop punctuation, full-width forms and spacing from
counting as recognition errors, which is what makes these metrics usable for
Chinese, Japanese and Korean transcripts. Expected values are worked out by
hand in the comments.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest
import yaml

YAML_DIR = Path(__file__).resolve().parent.parent / "function"
WORD_LEVEL = [
    "word_error_rate",
    "match_error_rate",
    "word_info_lost",
    "word_info_preserved",
]


def _load_eval(name: str):
    """Load YAML, materialize the code body, return the evaluate callable."""
    path = YAML_DIR / f"{name}.yaml"
    code = yaml.safe_load(path.read_text())["config"]["code"]
    ns: dict = {}
    vars(builtins)["exec"](compile(code, str(path), "exec"), ns)
    return ns["evaluate"]


def _run(name, reference, hypothesis, **params):
    return _load_eval(name)(
        None,
        hypothesis,
        reference,
        None,
        reference=reference,
        hypothesis=hypothesis,
        **params,
    )


def test_defaults_keep_todays_scores():
    # "hello," and "today?" don't match their reference words: 2 edits / 5 words.
    r = _run("word_error_rate", "hello how are you today", "Hello, how are you today?")
    assert r["reason"] == "WER: 40.0% (2 edits / 5 words)"
    # Two inserted punctuation marks over 5 reference characters.
    r = _run("character_error_rate", "我们明天见", "我们，明天见。")
    assert r["reason"] == "CER: 40.0% (2 edits / 5 chars)"


def test_basic_normalization_ignores_case_and_punctuation():
    r = _run(
        "word_error_rate",
        "hello how are you today",
        "Hello, how are you today?",
        normalize="basic",
    )
    assert r["reason"] == "WER: 0.0% (0 edits / 5 words)"


@pytest.mark.parametrize(
    "reference, hypothesis",
    [
        ("我们明天见", "我们，明天见。"),  # full-width comma, ideographic full stop
        ("今日は良い天気です", "今日 は 良い 天気 です"),  # spaced ASR output
        ("ジョン・スミス", "ジョンスミス"),  # katakana middle dot
        ("予約番号は1234です", "予約番号は１２３４です"),  # full-width digits
    ],
)
def test_basic_cer_ignores_formatting(reference, hypothesis):
    assert (
        _run("character_error_rate", reference, hypothesis, normalize="basic")["score"]
        == 1.0
    )


@pytest.mark.parametrize(
    "reference, hypothesis, score",
    [
        (
            "時々雨",
            "時雨",
            pytest.approx(2 / 3),
        ),  # 々 is part of the word: 1 deletion / 3
        ("コーヒー", "コヒー", 0.75),  # ー is a letter, not a dash: 1 deletion / 4
    ],
)
def test_basic_normalization_keeps_cjk_word_characters(reference, hypothesis, score):
    assert (
        _run("character_error_rate", reference, hypothesis, normalize="basic")["score"]
        == score
    )


def test_auto_tokenization_scores_cjk_per_character():
    # Spacing alone no longer counts: both sides are the same 9 characters.
    r = _run(
        "word_error_rate",
        "今日は良い天気です",
        "今日 は 良い 天気 です",
        tokenize="auto",
    )
    assert r["reason"] == "WER: 0.0% (0 edits / 9 words)"
    # One substituted character (良 → 悪) out of 9.
    r = _run(
        "word_error_rate", "今日は良い天気です", "今日は悪い天気です", tokenize="auto"
    )
    assert r["reason"] == "WER: 11.1% (1 edits / 9 words)"


def test_auto_tokenization_keeps_latin_and_digit_runs_as_words():
    # 予 約 番 号 は 1234 で す: 8 tokens once the full-width digits map to 1234.
    r = _run(
        "word_error_rate",
        "予約番号は1234です",
        "予約番号は１２３４です",
        normalize="basic",
        tokenize="auto",
    )
    assert r["reason"] == "WER: 0.0% (0 edits / 8 words)"


def test_char_tokenization_ignores_spaces():
    # "bookatablefortwo" vs "bookacablefortwo": 1 substitution / 16 characters.
    r = _run(
        "word_error_rate",
        "book a table for two",
        "book a cable for two",
        tokenize="char",
    )
    assert r["score"] == pytest.approx(1 - 1 / 16)


@pytest.mark.parametrize(
    "name, reason, score",
    [
        # 8 hits and 1 substitution over 9 characters on each side.
        ("match_error_rate", "MER: 11.1% (H=8, S=1, D=0, I=0)", pytest.approx(8 / 9)),
        (
            "word_info_lost",
            "WIL: 0.2099 (H=8, |ref|=9, |hyp|=9)",
            pytest.approx(64 / 81),
        ),
        ("word_info_preserved", "WIP: 0.7901", pytest.approx(64 / 81)),
    ],
)
def test_alignment_metrics_accept_auto_tokenization(name, reason, score):
    r = _run(name, "今日は良い天気です", "今日は悪い天気です", tokenize="auto")
    assert r["reason"] == reason
    assert r["score"] == score


@pytest.mark.parametrize("name", WORD_LEVEL)
def test_unknown_tokenize_value_is_rejected(name):
    r = _run(name, "a b", "a b", tokenize="sentence")
    assert r["score"] == 0.0
    assert "tokenize" in r["reason"]


@pytest.mark.parametrize("name", WORD_LEVEL + ["character_error_rate"])
def test_unknown_normalize_value_is_rejected(name):
    r = _run(name, "a b", "a b", normalize="nfkc")
    assert r["score"] == 0.0
    assert "normalize" in r["reason"]


@pytest.mark.parametrize(
    "name, reason",
    [
        # One substituted character (良 → 悪) out of 9, as in the tests above.
        ("word_error_rate", "WER: 11.1% (1 edits / 9 words)"),
        ("character_error_rate", "CER: 11.1% (1 edits / 9 chars)"),
        ("match_error_rate", "MER: 11.1% (H=8, S=1, D=0, I=0)"),
        ("word_info_lost", "WIL: 0.2099 (H=8, |ref|=9, |hyp|=9)"),
        ("word_info_preserved", "WIP: 0.7901"),
    ],
)
def test_bodies_run_in_the_local_fallback_runner(name, reason, settings, monkeypatch):
    """A self-hosted install without the code executor can opt into the local
    RestrictedPython runner, which rejects names starting with "_" and needs
    its own guard for tuple unpacking. The bodies have to work there too."""
    from agentic_eval.core_evals.fi_utils import sandbox

    settings.CLOUD_DEPLOYMENT = ""
    settings.CODE_EXECUTOR_LOCAL_FALLBACK = True
    monkeypatch.setattr(sandbox, "CODE_EXECUTOR_URL", "")
    code = yaml.safe_load((YAML_DIR / f"{name}.yaml").read_text())["config"]["code"]
    inputs = {
        "reference": "今日は良い天気です",
        "hypothesis": "今日は悪い天気です",
        "normalize": "basic",
        "tokenize": "auto",
    }

    result = sandbox.execute_sandboxed_python(code, inputs, timeout=10)

    assert result["status"] == "success", result
    assert result["data"]["reason"] == reason
