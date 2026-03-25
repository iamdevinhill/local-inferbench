"""Tests for the quality scoring module."""

import pytest

from local_inferbench.scoring import (
    ResponseScore,
    ScoringSummary,
    compute_corpus_idf,
    compute_scoring_summary,
    score_category,
    score_coherence,
    score_completeness,
    score_correctness,
    score_length,
    score_relevance,
    score_response,
    _check_contains,
    _check_numeric,
    _check_code_execution,
    _extract_python_code,
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

    def test_continuous_scoring_no_jumps(self):
        """Verify that similar inputs produce similar scores (no step discontinuities)."""
        # Generate texts with gradually increasing repetition
        scores = []
        for unique_ratio in [0.29, 0.31, 0.49, 0.51, 0.69, 0.71]:
            # Create text that roughly achieves target unique ratio
            base = "the quick brown fox jumps over the lazy dog near the river bank"
            words = base.split()
            text = " ".join(words) + ". " + " ".join(words[:3]) * int(10 * (1 - unique_ratio))
            s, _ = score_coherence(text)
            scores.append(s)
        # Check that adjacent scores don't have huge jumps (old code had 0.3 jumps)
        for i in range(len(scores) - 1):
            assert abs(scores[i] - scores[i + 1]) < 0.3

    def test_sentence_variation(self):
        """Text with varied sentence lengths should score higher than uniform."""
        varied = "Short. This is a medium-length sentence about something. Here is a much longer sentence that goes on for quite a while to test variation."
        uniform = "Five word sentence here. Five word sentence here. Five word sentence here. Five word sentence here."
        score_varied, details_varied = score_coherence(varied)
        score_uniform, details_uniform = score_coherence(uniform)
        # Varied text should have higher sentence_length_cv
        if "sentence_length_cv" in details_varied and "sentence_length_cv" in details_uniform:
            assert details_varied["sentence_length_cv"] > details_uniform["sentence_length_cv"]

    def test_transition_words_detected(self):
        text = "First, we consider the problem. However, there is a complication. Furthermore, the data suggests otherwise. For example, the results show improvement."
        _, details = score_coherence(text)
        assert details["transitions_found"] >= 2


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

    def test_tfidf_weighting(self):
        """IDF weighting should give distinctive terms more importance."""
        corpus = [
            "Write a Python function for IPv4 validation",
            "Write a Python function for sorting",
            "Write a Python function for string reversal",
        ]
        idf = compute_corpus_idf(corpus)
        # "ipv4" appears in 1 doc, "python" appears in 3 — ipv4 should have higher IDF
        assert idf.get("ipv4", 0) > idf.get("python", 0)
        assert idf.get("validation", 0) > idf.get("function", 0)

    def test_tfidf_relevance_scoring(self):
        """TF-IDF weighted scoring should give higher score when distinctive terms match."""
        corpus_idf = compute_corpus_idf([
            "Write a Python function for IPv4 validation",
            "Write a Python function for sorting",
        ])
        prompt = "Write a Python function for IPv4 validation"
        # Response with distinctive term
        good_response = "Here is an IPv4 validation function that checks each octet."
        # Response with only generic terms
        weak_response = "Here is a function written in Python."

        score_good, details_good = score_relevance(prompt, good_response, corpus_idf)
        score_weak, details_weak = score_relevance(prompt, weak_response, corpus_idf)
        assert details_good["scoring_method"] == "tfidf_weighted"
        assert score_good > score_weak


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


class TestScoreCorrectness:
    def test_contains_match(self):
        score, details = _check_contains(
            "The ball costs $0.05 dollars.",
            ["$0.05", "five cents"],
        )
        assert score == 1.0
        assert "$0.05" in details["matched"]

    def test_contains_no_match(self):
        score, details = _check_contains(
            "The ball costs $0.10 dollars.",
            ["$0.05", "five cents"],
        )
        assert score == 0.0
        assert details["matched"] == []

    def test_contains_case_insensitive(self):
        score, _ = _check_contains(
            "The answer involves FIVE CENTS.",
            ["five cents"],
        )
        assert score == 1.0

    def test_numeric_exact(self):
        score, details = _check_numeric("The answer is 0.0707 meters per minute.", 0.0707, 0.001)
        assert score == 1.0

    def test_numeric_close(self):
        score, details = _check_numeric("The answer is approximately 0.071.", 0.0707, 0.001)
        assert score >= 0.5  # Close enough for partial credit

    def test_numeric_wrong(self):
        score, _ = _check_numeric("The answer is 42.", 0.0707, 0.001)
        assert score == 0.0

    def test_numeric_integer(self):
        score, _ = _check_numeric("The expected number of flips is 14.", 14.0, 0.5)
        assert score == 1.0

    def test_no_verification_data(self):
        score, details = score_correctness("Any response")
        assert score == 0.5
        assert details["check"] == "none"


class TestCodeExecution:
    def test_extract_fenced_code(self):
        response = "Here's the code:\n```python\ndef foo():\n    return 42\n```\nThat's it."
        code = _extract_python_code(response)
        assert code is not None
        assert "def foo" in code

    def test_extract_indented_code(self):
        response = "The function:\n\n    def bar(x):\n        return x * 2\n\nDone."
        code = _extract_python_code(response)
        assert code is not None
        assert "def bar" in code

    def test_code_execution_pass(self):
        response = "```python\ndef add(a, b):\n    return a + b\n```"
        score, details = _check_code_execution(response, [
            {"function": "add", "input": [2, 3], "expected": 5},
            {"function": "add", "input": [0, 0], "expected": 0},
        ])
        assert score == 1.0
        assert details["tests_passed"] == 2
        assert details["tests_total"] == 2

    def test_code_execution_partial(self):
        response = "```python\ndef add(a, b):\n    return a + b + 1\n```"
        score, details = _check_code_execution(response, [
            {"function": "add", "input": [2, 3], "expected": 5},  # returns 6, fail
            {"function": "add", "input": [-1, 0], "expected": 0},  # returns 0, pass
        ])
        assert 0.0 < score < 1.0
        assert details["tests_passed"] == 1

    def test_code_execution_syntax_error(self):
        response = "```python\ndef broken(:\n    return\n```"
        score, details = _check_code_execution(response, [
            {"function": "broken", "input": [], "expected": None},
        ])
        assert score == 0.1  # Small credit for attempting code
        assert "syntax_error" in details["error"]

    def test_code_execution_no_code(self):
        response = "I think you should use a for loop to solve this problem."
        score, details = _check_code_execution(response, [
            {"function": "solve", "input": [], "expected": 42},
        ])
        assert score == 0.0
        assert details["error"] == "no_code_found"

    def test_code_execution_runtime_error(self):
        response = "```python\ndef divide(a, b):\n    return a / b\n```"
        score, details = _check_code_execution(response, [
            {"function": "divide", "input": [1, 0], "expected": 0},
        ])
        assert score < 0.5  # Runtime error


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

    def test_code_syntax_check(self):
        """Code category should now include syntax validation."""
        valid_code = "```python\ndef hello():\n    return 'world'\n```"
        invalid_code = "```python\ndef hello(\n    return\n```"
        score_valid, d_valid = score_category(valid_code, "code", "Write a function")
        score_invalid, d_invalid = score_category(invalid_code, "code", "Write a function")
        assert score_valid > score_invalid

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


class TestPerCategoryWeights:
    def test_math_weights_correctness_heavily(self):
        """Math prompts with known answers should weight correctness at 0.50."""
        correct = score_response(
            prompt_text="What is 2+2?",
            prompt_category="math",
            response_text="The answer is 4. We compute 2+2=4 using basic arithmetic.",
            finish_reason="stop",
            answer_type="contains",
            expected_answer=["4"],
        )
        incorrect = score_response(
            prompt_text="What is 2+2?",
            prompt_category="math",
            response_text="The answer is 5. We compute 2+2=5 using basic arithmetic.",
            finish_reason="stop",
            answer_type="contains",
            expected_answer=["4"],
        )
        # Correctness (weight 0.50) should create a significant gap
        assert correct.overall_score > incorrect.overall_score
        assert correct.correctness_score == 1.0
        assert incorrect.correctness_score == 0.0

    def test_creative_weights_coherence_heavily(self):
        """Creative prompts should weight coherence and category more than correctness."""
        result = score_response(
            prompt_text="Write a poem",
            prompt_category="creative",
            response_text="The morning sun rises softly over distant hills.\n\nBirds sing melodies that echo through the valley.\n\nPeace settles like dew upon the grass.",
            finish_reason="stop",
        )
        weights = result.details.get("weights_used", {})
        assert weights.get("coherence", 0) >= 0.25
        assert weights.get("correctness", 0) <= 0.10

    def test_no_correctness_data_redistributes_weight(self):
        """When no correctness data is available, its weight should go to category + coherence."""
        result = score_response(
            prompt_text="Explain something",
            prompt_category="math",
            response_text="The equation shows that x = 5. Therefore the sum equals 10.",
            finish_reason="stop",
            # No answer_type provided
        )
        weights = result.details.get("weights_used", {})
        assert weights["correctness"] == 0.0
        assert weights["category"] > 0.15  # Got bonus from correctness redistribution


class TestCorpusIDF:
    def test_empty_corpus(self):
        idf = compute_corpus_idf([])
        assert idf == {}

    def test_single_document(self):
        idf = compute_corpus_idf(["Write a Python function"])
        assert "python" in idf
        assert "function" in idf

    def test_rare_words_higher_idf(self):
        corpus = [
            "Write a Python function for sorting",
            "Write a Python function for searching",
            "Write a Python function for IPv4 validation",
        ]
        idf = compute_corpus_idf(corpus)
        # "ipv4" only in 1 doc, "python" in all 3
        assert idf["ipv4"] > idf["python"]

    def test_common_words_lower_idf(self):
        corpus = [
            "Explain machine learning concepts",
            "Explain deep learning architectures",
            "Explain reinforcement learning algorithms",
        ]
        idf = compute_corpus_idf(corpus)
        # "explain" and "learning" in all 3, "machine" only in 1
        assert idf.get("machine", 0) > idf.get("learning", 0)


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
        assert 0.0 <= result.correctness_score <= 1.0
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
        assert "correctness_score" in d
        assert isinstance(d["overall_score"], float)

    def test_with_correctness_data(self):
        result = score_response(
            prompt_text="A bat and ball cost $1.10. The bat costs $1.00 more. How much is the ball?",
            prompt_category="reasoning",
            response_text="The ball costs $0.05. If the ball is $0.05, the bat is $1.05, totaling $1.10.",
            finish_reason="stop",
            answer_type="contains",
            expected_answer=["$0.05", "0.05", "five cents"],
        )
        assert result.correctness_score == 1.0
        assert result.overall_score > 0.5

    def test_with_corpus_idf(self):
        idf = compute_corpus_idf([
            "Write Python code for sorting",
            "Write Python code for IPv4 validation",
        ])
        result = score_response(
            prompt_text="Write Python code for IPv4 validation",
            prompt_category="code",
            response_text="def is_valid_ipv4(s):\n    parts = s.split('.')\n    return len(parts) == 4",
            finish_reason="stop",
            corpus_idf=idf,
        )
        assert result.details["relevance"]["scoring_method"] == "tfidf_weighted"

    def test_weights_in_details(self):
        result = score_response(
            prompt_text="test",
            prompt_category="math",
            response_text="The answer is 42.",
            finish_reason="stop",
        )
        assert "weights_used" in result.details


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
