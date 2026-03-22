"""Quality/accuracy scoring for generated responses.

Uses only stdlib — no external LLM judge. Scores responses on length,
coherence, relevance, completeness, and category-specific heuristics.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

# Minimal stopwords for relevance scoring
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "ought",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their", "this",
    "that", "these", "those", "what", "which", "who", "whom", "how",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "if", "than", "too", "very", "just", "also",
})

# Expected length ranges by category: (min_chars, ideal_min, ideal_max, max_chars)
_LENGTH_EXPECTATIONS: dict[str, tuple[int, int, int, int]] = {
    "code": (50, 150, 2000, 5000),
    "reasoning": (100, 200, 2000, 5000),
    "creative": (30, 100, 2000, 5000),
    "extraction": (20, 50, 500, 2000),
    "summarization": (30, 80, 500, 1500),
    "instruction": (50, 150, 1500, 4000),
    "analysis": (100, 200, 2000, 5000),
    "math": (50, 150, 2000, 5000),
    "science": (80, 200, 2000, 5000),
    "long_generation": (200, 500, 5000, 10000),
}

_DEFAULT_LENGTH = (20, 100, 2000, 5000)

# Weights for composite score
_WEIGHTS = {
    "length": 0.15,
    "coherence": 0.25,
    "relevance": 0.20,
    "completeness": 0.15,
    "category": 0.25,
}


@dataclass
class ResponseScore:
    """Quality score for a single generated response."""

    prompt_text: str
    prompt_category: str
    response_text: str

    length_score: float
    coherence_score: float
    relevance_score: float
    completeness_score: float
    category_score: float

    overall_score: float

    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_category": self.prompt_category,
            "length_score": round(self.length_score, 3),
            "coherence_score": round(self.coherence_score, 3),
            "relevance_score": round(self.relevance_score, 3),
            "completeness_score": round(self.completeness_score, 3),
            "category_score": round(self.category_score, 3),
            "overall_score": round(self.overall_score, 3),
        }


@dataclass
class ScoringSummary:
    """Aggregated quality scores across all responses in a run."""

    mean_overall: float
    median_overall: float
    min_overall: float
    max_overall: float
    mean_by_category: dict[str, float]
    individual_scores: list[ResponseScore]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_overall": round(self.mean_overall, 3),
            "median_overall": round(self.median_overall, 3),
            "min_overall": round(self.min_overall, 3),
            "max_overall": round(self.max_overall, 3),
            "mean_by_category": {
                k: round(v, 3) for k, v in self.mean_by_category.items()
            },
        }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase words, filtering stopwords."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def _ngrams(words: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def _char_entropy(text: str) -> float:
    """Shannon entropy of character distribution."""
    if not text:
        return 0.0
    counts = Counter(text.lower())
    total = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# --- Individual scorers ---


def score_length(
    response: str, category: str, max_tokens: int
) -> tuple[float, dict[str, Any]]:
    """Score based on whether response length is appropriate for the category."""
    length = len(response)
    details: dict[str, Any] = {"response_length": length, "category": category}

    if length == 0:
        return 0.0, details

    min_c, ideal_min, ideal_max, max_c = _LENGTH_EXPECTATIONS.get(
        category, _DEFAULT_LENGTH
    )

    if ideal_min <= length <= ideal_max:
        score = 1.0
    elif length < min_c:
        score = length / min_c * 0.3
    elif length < ideal_min:
        score = 0.3 + 0.7 * (length - min_c) / max(ideal_min - min_c, 1)
    elif length > max_c:
        score = max(0.3, 1.0 - (length - max_c) / max_c)
    else:  # ideal_max < length <= max_c
        score = 1.0 - 0.3 * (length - ideal_max) / max(max_c - ideal_max, 1)

    return _clamp(score), details


def score_coherence(response: str) -> tuple[float, dict[str, Any]]:
    """Score based on repetition, structure, and entropy."""
    details: dict[str, Any] = {}

    if len(response) < 5:
        return 0.1, details

    # Repetition detection via n-grams
    words = response.lower().split()
    rep_score = 1.0
    if len(words) >= 5:
        trigrams = _ngrams(words, 3)
        if trigrams:
            unique_ratio = len(set(trigrams)) / len(trigrams)
            details["trigram_unique_ratio"] = round(unique_ratio, 3)
            if unique_ratio < 0.3:
                rep_score = 0.2
            elif unique_ratio < 0.5:
                rep_score = 0.5
            elif unique_ratio < 0.7:
                rep_score = 0.8
            else:
                rep_score = 1.0

    # Entropy check
    entropy = _char_entropy(response)
    details["char_entropy"] = round(entropy, 3)
    entropy_score = _clamp(entropy / 4.5)  # Normal English ~4.0-4.5 bits

    # Structure check: has sentences or code patterns
    has_structure = bool(
        re.search(r"[.!?]\s", response)
        or re.search(r"(def |class |function |```|import )", response)
        or re.search(r"\n\s*([-*\d])", response)  # lists
    )
    structure_score = 1.0 if has_structure else 0.5
    details["has_structure"] = has_structure

    score = rep_score * 0.4 + entropy_score * 0.3 + structure_score * 0.3
    return _clamp(score), details


def score_relevance(prompt: str, response: str) -> tuple[float, dict[str, Any]]:
    """Score based on keyword overlap between prompt and response."""
    details: dict[str, Any] = {}

    prompt_keywords = set(_tokenize(prompt))
    response_keywords = set(_tokenize(response))

    if not prompt_keywords:
        return 0.5, details  # Can't assess

    overlap = prompt_keywords & response_keywords
    keyword_coverage = len(overlap) / len(prompt_keywords)
    details["keyword_coverage"] = round(keyword_coverage, 3)
    details["prompt_keywords"] = len(prompt_keywords)
    details["overlapping_keywords"] = len(overlap)

    # Penalize if response is just echoing the prompt
    similarity = SequenceMatcher(None, prompt.lower(), response.lower()[:len(prompt) * 2]).ratio()
    details["prompt_echo_similarity"] = round(similarity, 3)

    echo_penalty = 1.0
    if similarity > 0.8:
        echo_penalty = 0.3
    elif similarity > 0.6:
        echo_penalty = 0.7

    score = keyword_coverage * echo_penalty
    return _clamp(score), details


def score_completeness(
    response: str, finish_reason: str
) -> tuple[float, dict[str, Any]]:
    """Score based on whether the response completed naturally."""
    details: dict[str, Any] = {"finish_reason": finish_reason}

    if finish_reason == "stop":
        return 1.0, details

    # Truncated — check if it ended at a sentence boundary
    stripped = response.rstrip()
    ends_at_boundary = bool(stripped) and stripped[-1] in ".!?}])\""
    details["ends_at_boundary"] = ends_at_boundary

    if finish_reason == "length":
        return (0.7 if ends_at_boundary else 0.3), details

    return 0.5, details


def score_category(
    response: str, category: str, prompt: str
) -> tuple[float, dict[str, Any]]:
    """Category-specific quality heuristics."""
    details: dict[str, Any] = {"category": category}
    response_lower = response.lower()

    if category == "code":
        return _score_code(response, response_lower, details)
    elif category == "reasoning":
        return _score_reasoning(response_lower, details)
    elif category == "creative":
        return _score_creative(response, details)
    elif category == "extraction":
        return _score_extraction(response, prompt, details)
    elif category == "summarization":
        return _score_summarization(response, prompt, details)
    elif category == "instruction":
        return _score_instruction(response_lower, details)
    elif category == "analysis":
        return _score_analysis(response_lower, details)
    elif category == "math":
        return _score_math(response_lower, details)
    elif category == "science":
        return _score_science(response_lower, details)
    else:
        return 0.5, details  # Unknown category, neutral score


def _score_code(
    response: str, lower: str, details: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    signals = 0
    checks = 0

    # Code-like patterns
    checks += 1
    has_code = bool(
        re.search(r"(def |class |function |const |let |var |import |#include)", lower)
        or re.search(r"```", response)
    )
    if has_code:
        signals += 1
    details["has_code_patterns"] = has_code

    # Indentation
    checks += 1
    has_indent = bool(re.search(r"\n    \w", response) or re.search(r"\n\t\w", response))
    if has_indent:
        signals += 1
    details["has_indentation"] = has_indent

    # Balanced brackets
    checks += 1
    parens_balanced = response.count("(") == response.count(")")
    braces_balanced = response.count("{") == response.count("}")
    brackets_ok = parens_balanced and braces_balanced
    if brackets_ok:
        signals += 1
    details["brackets_balanced"] = brackets_ok

    score = signals / checks if checks > 0 else 0.5
    return _clamp(score), details


def _score_reasoning(
    lower: str, details: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    step_indicators = [
        "first", "second", "third", "step", "therefore", "because",
        "since", "thus", "hence", "consequently", "as a result",
        "this means", "in conclusion", "let's",
    ]
    found = [ind for ind in step_indicators if ind in lower]
    details["reasoning_indicators"] = found
    score = _clamp(len(found) / 3)
    return score, details


def _score_creative(
    response: str, details: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    words = response.lower().split()
    if len(words) < 5:
        return 0.2, details

    # Type-token ratio (vocabulary diversity)
    unique_words = set(words)
    ttr = len(unique_words) / len(words)
    details["type_token_ratio"] = round(ttr, 3)

    # Paragraph structure
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    details["paragraph_count"] = len(paragraphs)

    ttr_score = _clamp(ttr / 0.6)  # 0.6 TTR is quite diverse
    structure_score = _clamp(len(paragraphs) / 2)

    return _clamp(ttr_score * 0.6 + structure_score * 0.4), details


def _score_extraction(
    response: str, prompt: str, details: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    # Check if data from prompt appears structured in response
    # Look for dates, numbers, names from the prompt
    prompt_numbers = set(re.findall(r"\d{4}|\d+\.\d+|\d+", prompt))
    response_numbers = set(re.findall(r"\d{4}|\d+\.\d+|\d+", response))

    overlap = prompt_numbers & response_numbers
    coverage = len(overlap) / len(prompt_numbers) if prompt_numbers else 0.5
    details["number_extraction_coverage"] = round(coverage, 3)

    # Check for structured output (lists, key-value pairs)
    has_structure = bool(
        re.search(r"[-*•]\s", response)
        or re.search(r"\d+[.)]\s", response)
        or re.search(r":\s", response)
    )
    details["has_structured_output"] = has_structure

    score = coverage * 0.6 + (1.0 if has_structure else 0.3) * 0.4
    return _clamp(score), details


def _score_summarization(
    response: str, prompt: str, details: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    # Summaries should be shorter than the prompt content
    ratio = len(response) / max(len(prompt), 1)
    details["length_ratio"] = round(ratio, 3)

    # Key terms coverage
    prompt_words = set(_tokenize(prompt))
    response_words = set(_tokenize(response))
    coverage = len(prompt_words & response_words) / max(len(prompt_words), 1)
    details["key_term_coverage"] = round(coverage, 3)

    conciseness = 1.0 if ratio < 5 else _clamp(1.0 - (ratio - 5) / 10)
    score = coverage * 0.6 + conciseness * 0.4
    return _clamp(score), details


def _score_instruction(
    lower: str, details: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    # Look for sequential markers
    seq_patterns = re.findall(
        r"(\d+[.)]\s|step \d|first|second|third|then|next|finally|after that)", lower
    )
    details["sequential_markers"] = len(seq_patterns)
    score = _clamp(len(seq_patterns) / 3)
    return score, details


def _score_analysis(
    lower: str, details: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    perspective_markers = [
        "however", "on the other hand", "alternatively", "conversely",
        "advantage", "disadvantage", "pro", "con", "benefit", "drawback",
        "trade-off", "tradeoff", "while", "whereas", "although",
    ]
    found = [m for m in perspective_markers if m in lower]
    details["perspective_markers"] = found
    score = _clamp(len(found) / 3)
    return score, details


def _score_math(
    lower: str, details: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    # Math responses should have numbers, equations, or mathematical terms
    has_numbers = bool(re.search(r"\d+", lower))
    math_terms = [
        "equation", "formula", "proof", "theorem", "therefore",
        "equals", "sum", "product", "integral", "derivative",
        "probability", "=", "+", "×", "÷",
    ]
    found = [t for t in math_terms if t in lower]
    details["math_terms_found"] = found
    details["has_numbers"] = has_numbers

    score = _clamp((len(found) + (2 if has_numbers else 0)) / 4)
    return score, details


def _score_science(
    lower: str, details: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    # Science responses should be explanatory with technical terms
    explanation_markers = [
        "because", "due to", "causes", "results in", "mechanism",
        "process", "occurs", "involves", "molecules", "atoms",
        "energy", "cells", "reaction", "system", "theory",
    ]
    found = [m for m in explanation_markers if m in lower]
    details["science_markers"] = found
    score = _clamp(len(found) / 3)
    return score, details


# --- Main scoring interface ---


def score_response(
    prompt_text: str,
    prompt_category: str,
    response_text: str,
    finish_reason: str,
    max_tokens: int = 512,
) -> ResponseScore:
    """Score a single response across all quality dimensions."""
    length_s, length_d = score_length(response_text, prompt_category, max_tokens)
    coherence_s, coherence_d = score_coherence(response_text)
    relevance_s, relevance_d = score_relevance(prompt_text, response_text)
    completeness_s, completeness_d = score_completeness(response_text, finish_reason)
    category_s, category_d = score_category(response_text, prompt_category, prompt_text)

    overall = (
        length_s * _WEIGHTS["length"]
        + coherence_s * _WEIGHTS["coherence"]
        + relevance_s * _WEIGHTS["relevance"]
        + completeness_s * _WEIGHTS["completeness"]
        + category_s * _WEIGHTS["category"]
    )

    return ResponseScore(
        prompt_text=prompt_text,
        prompt_category=prompt_category,
        response_text=response_text,
        length_score=length_s,
        coherence_score=coherence_s,
        relevance_score=relevance_s,
        completeness_score=completeness_s,
        category_score=category_s,
        overall_score=round(overall, 3),
        details={
            "length": length_d,
            "coherence": coherence_d,
            "relevance": relevance_d,
            "completeness": completeness_d,
            "category": category_d,
        },
    )


def compute_scoring_summary(scores: list[ResponseScore]) -> ScoringSummary:
    """Aggregate quality scores across all responses."""
    if not scores:
        return ScoringSummary(
            mean_overall=0.0,
            median_overall=0.0,
            min_overall=0.0,
            max_overall=0.0,
            mean_by_category={},
            individual_scores=[],
        )

    overall_values = [s.overall_score for s in scores]

    # Group by category
    by_category: dict[str, list[float]] = {}
    for s in scores:
        by_category.setdefault(s.prompt_category, []).append(s.overall_score)
    mean_by_cat = {cat: statistics.mean(vals) for cat, vals in sorted(by_category.items())}

    return ScoringSummary(
        mean_overall=statistics.mean(overall_values),
        median_overall=statistics.median(overall_values),
        min_overall=min(overall_values),
        max_overall=max(overall_values),
        mean_by_category=mean_by_cat,
        individual_scores=scores,
    )
