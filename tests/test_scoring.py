"""Tests for the quality scoring module."""

import pytest

from local_inferbench.scoring import (
    ResponseScore,
    ScoringSummary,
    compute_scoring_summary,
    score_category,
    score_coherence,
    score_completeness,
    score_length,
    score_relevance,
    score_response,
)


class TestScoreLength:
    def test_empty_response(self):
        score, _ = score_length("", "code", 512)
        assert score == 0.0

    def test_appropriate_length(self):
        response = "def is_valid_ipv4(s):\n    parts = s.split('.')\n" + "x" * 200
        score, _ = score_length(response, "code", 512)
        assert score > 0.7

    def test_very_short_response(self):
        score, _ = score_length("ok", "reasoning", 512)
        assert score < 0.3

    def test_category_aware(self):
        # Extraction expects shorter responses than reasoning
        text = "March 15, 2001 - founded"
        score_ext, _ = score_length(text, "extraction", 512)
        score_reason, _ = score_length(text, "reasoning", 512)
        assert score_ext >= score_reason


class TestScoreCoherence:
    def test_empty_response(self):
        score, _ = score_coherence("")
        assert score < 0.2

    def test_normal_text(self):
        text = "The quick brown fox jumps over the lazy dog. This is a well-structured sentence with proper grammar."
        score, details = score_coherence(text)
        assert score > 0.5
        assert details["has_structure"] is True

    def test_highly_repetitive(self):
        text = "the cat sat " * 50
        score, details = score_coherence(text)
        assert score < 0.6
        assert details["trigram_unique_ratio"] < 0.3

    def test_code_counts_as_structured(self):
        text = "def hello():\n    print('world')\n    return True"
        score, details = score_coherence(text)
        assert details["has_structure"] is True

    def test_low_entropy(self):
        text = "aaaaaaaaaaaaaaaaaaaaaaaaa"
        score, _ = score_coherence(text)
        assert score < 0.6


class TestScoreRelevance:
    def test_relevant_response(self):
        prompt = "Explain why objects fall at the same rate in a vacuum"
        response = "Objects fall at the same rate in a vacuum because gravity accelerates all objects equally, regardless of mass. In a vacuum, there is no air resistance to slow lighter objects."
        score, details = score_relevance(prompt, response)
        assert score > 0.3
        assert details["keyword_coverage"] > 0

    def test_irrelevant_response(self):
        prompt = "Write a Python function for IPv4 validation"
        response = "The weather today is sunny and warm. I enjoy going to the beach on days like this."
        score, _ = score_relevance(prompt, response)
        assert score < 0.3

    def test_echo_penalty(self):
        prompt = "What is the meaning of life?"
        response = "What is the meaning of life? What is the meaning of life?"
        score, details = score_relevance(prompt, response)
        assert details["prompt_echo_similarity"] > 0.5


class TestScoreCompleteness:
    def test_natural_stop(self):
        score, details = score_completeness("Full response here.", "stop")
        assert score == 1.0

    def test_truncated_at_boundary(self):
        score, details = score_completeness("This is a complete thought.", "length")
        assert score == 0.7

    def test_truncated_mid_sentence(self):
        score, details = score_completeness("This is an incompl", "length")
        assert score == 0.3

    def test_unknown_reason(self):
        score, _ = score_completeness("Some text", "unknown")
        assert score == 0.5


class TestScoreCategory:
    def test_code_with_code(self):
        response = "def is_valid_ipv4(s):\n    parts = s.split('.')\n    if len(parts) != 4:\n        return False\n    return True"
        score, details = score_category(response, "code", "Write a function")
        assert score > 0.5
        assert details["has_code_patterns"] is True

    def test_code_without_code(self):
        response = "IPv4 addresses have four octets separated by dots."
        score, details = score_category(response, "code", "Write a function")
        assert score < 0.7

    def test_reasoning_with_steps(self):
        response = "First, we consider gravity. Because of this, objects accelerate equally. Therefore, they fall at the same rate. Thus, mass doesn't matter."
        score, details = score_category(response, "reasoning", "Explain")
        assert score > 0.5
        assert len(details["reasoning_indicators"]) >= 2

    def test_extraction_with_data(self):
        prompt = "Extract dates: founded March 15, 2001. IPO July 22, 2008."
        response = "- March 15, 2001: founded\n- July 22, 2008: IPO"
        score, details = score_category(response, "extraction", prompt)
        assert score > 0.5
        assert details["has_structured_output"] is True

    def test_analysis_with_perspectives(self):
        response = "On one hand, this is beneficial. However, there are disadvantages. The trade-off between speed and quality is significant."
        score, details = score_category(response, "analysis", "Analyze")
        assert score > 0.5

    def test_math_with_numbers(self):
        response = "The probability equals 1/2. Using the formula, we get sum = 100. Therefore the answer is 42."
        score, details = score_category(response, "math", "Calculate")
        assert score > 0.3
        assert details["has_numbers"] is True

    def test_unknown_category(self):
        score, _ = score_category("Some response", "unknown_cat", "prompt")
        assert score == 0.5


class TestScoreResponse:
    def test_returns_response_score(self):
        result = score_response(
            prompt_text="Write a Python function",
            prompt_category="code",
            response_text="def hello():\n    return 'world'",
            finish_reason="stop",
            max_tokens=512,
        )
        assert isinstance(result, ResponseScore)
        assert 0.0 <= result.overall_score <= 1.0
        assert 0.0 <= result.length_score <= 1.0
        assert 0.0 <= result.coherence_score <= 1.0
        assert result.prompt_category == "code"

    def test_empty_response_low_score(self):
        result = score_response(
            prompt_text="Write something",
            prompt_category="creative",
            response_text="",
            finish_reason="stop",
            max_tokens=512,
        )
        assert result.overall_score < 0.3

    def test_to_dict(self):
        result = score_response(
            prompt_text="test",
            prompt_category="reasoning",
            response_text="Because of gravity, objects fall. Therefore they accelerate equally.",
            finish_reason="stop",
        )
        d = result.to_dict()
        assert "overall_score" in d
        assert "length_score" in d
        assert isinstance(d["overall_score"], float)


class TestComputeScoringSummary:
    def test_empty(self):
        summary = compute_scoring_summary([])
        assert summary.mean_overall == 0.0
        assert summary.mean_by_category == {}

    def test_with_scores(self):
        scores = [
            score_response("p1", "code", "def f():\n    return 1", "stop"),
            score_response("p2", "reasoning", "First, because. Therefore, thus the answer.", "stop"),
            score_response("p3", "code", "class Foo:\n    def bar(self):\n        pass", "stop"),
        ]
        summary = compute_scoring_summary(scores)
        assert summary.mean_overall > 0
        assert summary.min_overall <= summary.max_overall
        assert "code" in summary.mean_by_category
        assert "reasoning" in summary.mean_by_category
        assert len(summary.individual_scores) == 3

    def test_to_dict(self):
        scores = [score_response("p", "code", "def f(): pass", "stop")]
        summary = compute_scoring_summary(scores)
        d = summary.to_dict()
        assert "mean_overall" in d
        assert "mean_by_category" in d
