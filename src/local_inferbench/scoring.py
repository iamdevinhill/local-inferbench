"""Quality/accuracy scoring for generated responses.

Uses only stdlib — no external LLM judge. Scores responses on length,
coherence, relevance, completeness, correctness (when verifiable),
and category-specific heuristics.
"""

from __future__ import annotations

import logging
import math
import re
import signal
import statistics
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger(__name__)

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

# Default weights for composite score
_DEFAULT_WEIGHTS = {
    "length": 0.10,
    "coherence": 0.20,
    "relevance": 0.15,
    "completeness": 0.10,
    "category": 0.20,
    "correctness": 0.25,
}

# Per-category weight overrides — correctness gets more weight where verifiable,
# form-based scores dominate where correctness can't be checked
_CATEGORY_WEIGHTS: dict[str, dict[str, float]] = {
    "math": {
        "length": 0.05, "coherence": 0.10, "relevance": 0.10,
        "completeness": 0.10, "category": 0.15, "correctness": 0.50,
    },
    "code": {
        "length": 0.05, "coherence": 0.10, "relevance": 0.10,
        "completeness": 0.10, "category": 0.15, "correctness": 0.50,
    },
    "reasoning": {
        "length": 0.05, "coherence": 0.15, "relevance": 0.10,
        "completeness": 0.10, "category": 0.20, "correctness": 0.40,
    },
    "creative": {
        "length": 0.10, "coherence": 0.30, "relevance": 0.10,
        "completeness": 0.10, "category": 0.35, "correctness": 0.05,
    },
    "extraction": {
        "length": 0.05, "coherence": 0.10, "relevance": 0.15,
        "completeness": 0.10, "category": 0.25, "correctness": 0.35,
    },
    "summarization": {
        "length": 0.10, "coherence": 0.20, "relevance": 0.20,
        "completeness": 0.10, "category": 0.25, "correctness": 0.15,
    },
    "instruction": {
        "length": 0.10, "coherence": 0.20, "relevance": 0.15,
        "completeness": 0.10, "category": 0.30, "correctness": 0.15,
    },
    "analysis": {
        "length": 0.10, "coherence": 0.20, "relevance": 0.15,
        "completeness": 0.10, "category": 0.30, "correctness": 0.15,
    },
    "science": {
        "length": 0.05, "coherence": 0.15, "relevance": 0.15,
        "completeness": 0.10, "category": 0.25, "correctness": 0.30,
    },
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
    correctness_score: float

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
            "correctness_score": round(self.correctness_score, 3),
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


def _get_weights(category: str, has_correctness: bool) -> dict[str, float]:
    """Get scoring weights for a category. If no correctness data available,
    redistribute correctness weight to category and coherence."""
    weights = _CATEGORY_WEIGHTS.get(category, _DEFAULT_WEIGHTS).copy()
    if not has_correctness:
        bonus = weights.get("correctness", 0.0)
        weights["correctness"] = 0.0
        weights["category"] = weights.get("category", 0.0) + bonus * 0.6
        weights["coherence"] = weights.get("coherence", 0.0) + bonus * 0.4
    return weights


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
    """Score based on repetition, structure, entropy, and sentence variation."""
    details: dict[str, Any] = {}

    if len(response) < 5:
        return 0.1, details

    # Repetition detection via n-grams — continuous scoring
    words = response.lower().split()
    rep_score = 1.0
    if len(words) >= 5:
        trigrams = _ngrams(words, 3)
        if trigrams:
            unique_ratio = len(set(trigrams)) / len(trigrams)
            details["trigram_unique_ratio"] = round(unique_ratio, 3)
            # Smooth curve: sqrt gives diminishing returns at high uniqueness
            rep_score = _clamp(unique_ratio ** 0.5)

    # Entropy check — normalized by text length
    entropy = _char_entropy(response)
    details["char_entropy"] = round(entropy, 3)
    # Short text naturally has lower entropy; scale threshold accordingly
    # English prose ~4.0-4.5 bits, short text ~3.0-3.5 is still fine
    min_length_chars = 50
    entropy_target = 3.0 + min(1.5, len(response) / 500)  # 3.0 for short, up to 4.5
    entropy_score = _clamp(entropy / entropy_target)

    # Structure check: has sentences or code patterns
    has_structure = bool(
        re.search(r"[.!?]\s", response)
        or re.search(r"(def |class |function |```|import )", response)
        or re.search(r"\n\s*([-*\d])", response)  # lists
    )
    structure_score = 1.0 if has_structure else 0.5
    details["has_structure"] = has_structure

    # Sentence-length variation — uniform sentence lengths suggest formulaic output
    sentences = [s.strip() for s in re.split(r"[.!?]+", response) if s.strip()]
    sent_variation_score = 0.5  # default neutral
    if len(sentences) >= 3:
        sent_lengths = [len(s.split()) for s in sentences]
        mean_len = statistics.mean(sent_lengths)
        if mean_len > 0:
            stdev = statistics.stdev(sent_lengths) if len(sent_lengths) > 1 else 0
            cv = stdev / mean_len  # coefficient of variation
            details["sentence_length_cv"] = round(cv, 3)
            # CV of 0.3-0.6 is typical for natural text
            sent_variation_score = _clamp(cv / 0.4)
    details["sentence_variation_score"] = round(sent_variation_score, 3)

    # Discourse transitions between sentences
    transition_words = {
        "however", "additionally", "furthermore", "moreover", "in contrast",
        "for example", "for instance", "specifically", "consequently",
        "therefore", "nevertheless", "meanwhile", "similarly", "conversely",
        "in addition", "on the other hand", "as a result", "in particular",
    }
    lower = response.lower()
    transitions_found = sum(1 for t in transition_words if t in lower)
    transition_score = _clamp(transitions_found / 2) if len(sentences) >= 3 else 0.5
    details["transitions_found"] = transitions_found

    score = (
        rep_score * 0.30
        + entropy_score * 0.20
        + structure_score * 0.20
        + sent_variation_score * 0.15
        + transition_score * 0.15
    )
    return _clamp(score), details


def score_relevance(
    prompt: str,
    response: str,
    corpus_idf: dict[str, float] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Score based on term-importance-weighted overlap between prompt and response."""
    details: dict[str, Any] = {}

    prompt_keywords = _tokenize(prompt)
    response_keyword_set = set(_tokenize(response))
    prompt_keyword_set = set(prompt_keywords)

    if not prompt_keyword_set:
        return 0.5, details  # Can't assess

    # TF-IDF weighted coverage if corpus IDF is available
    if corpus_idf:
        weighted_overlap = 0.0
        total_weight = 0.0
        for word in prompt_keyword_set:
            weight = corpus_idf.get(word, 1.0)  # rare words default to high weight
            total_weight += weight
            if word in response_keyword_set:
                weighted_overlap += weight
        keyword_coverage = weighted_overlap / total_weight if total_weight > 0 else 0.0
        details["scoring_method"] = "tfidf_weighted"
    else:
        overlap = prompt_keyword_set & response_keyword_set
        keyword_coverage = len(overlap) / len(prompt_keyword_set)
        details["scoring_method"] = "keyword_overlap"

    details["keyword_coverage"] = round(keyword_coverage, 3)
    details["prompt_keywords"] = len(prompt_keyword_set)
    details["overlapping_keywords"] = len(prompt_keyword_set & response_keyword_set)

    # Penalize if response is just echoing the prompt
    similarity = SequenceMatcher(
        None, prompt.lower(), response.lower()[:len(prompt) * 2]
    ).ratio()
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


# --- Correctness verification ---


def _check_contains(response: str, expected: list[str]) -> tuple[float, dict[str, Any]]:
    """Check if response contains any of the expected answer strings."""
    lower = response.lower()
    matched = [ans for ans in expected if ans.lower() in lower]
    return (1.0 if matched else 0.0), {"matched": matched, "check": "contains"}


def _check_numeric(
    response: str, expected: float, tolerance: float = 0.01
) -> tuple[float, dict[str, Any]]:
    """Check if response contains the expected numeric value."""
    # Extract all numbers from response
    numbers = re.findall(r"-?\d+\.?\d*", response)
    parsed = []
    for n in numbers:
        try:
            parsed.append(float(n))
        except ValueError:
            continue

    # Check for fraction patterns like 2/(9π) or 2/(9*pi)
    fractions = re.findall(r"(\d+)\s*/\s*\(?\s*(\d+)\s*[*×]?\s*(?:π|pi|\\pi)\s*\)?", response.lower())
    for num, denom in fractions:
        try:
            parsed.append(float(num) / (float(denom) * math.pi))
        except (ValueError, ZeroDivisionError):
            continue

    for val in parsed:
        if abs(val - expected) <= tolerance:
            return 1.0, {"found_value": val, "expected": expected, "check": "numeric"}

    # Partial credit if close
    if parsed:
        closest = min(parsed, key=lambda x: abs(x - expected))
        relative_error = abs(closest - expected) / max(abs(expected), 1e-9)
        if relative_error < 0.1:
            return 0.5, {"closest": closest, "expected": expected, "check": "numeric"}

    return 0.0, {"numbers_found": parsed[:10], "expected": expected, "check": "numeric"}


def _extract_python_code(response: str) -> str | None:
    """Extract Python code from a response (fenced or indented)."""
    # Try fenced code blocks first
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if fenced:
        return fenced[0].strip()

    # Try indented blocks (4+ spaces or tab after a blank line)
    lines = response.split("\n")
    code_lines: list[str] = []
    in_code = False
    for line in lines:
        if re.match(r"^(    |\t)\S", line) or (in_code and (line.strip() == "" or re.match(r"^(    |\t)", line))):
            code_lines.append(line)
            in_code = True
        elif in_code and line.strip():
            # Non-indented non-empty line while in code — check if it starts a def/class
            if re.match(r"^(def |class |@)", line):
                code_lines.append(line)
            else:
                break

    if code_lines:
        return "\n".join(code_lines).strip()
    return None


class _CodeExecTimeout(Exception):
    pass


def _timeout_handler(signum: int, frame: object) -> None:
    raise _CodeExecTimeout("Code execution timed out")


def _check_code_execution(
    response: str, test_cases: list[dict[str, object]]
) -> tuple[float, dict[str, Any]]:
    """Extract code from response and run test cases against it."""
    details: dict[str, Any] = {"check": "code_execution"}
    code = _extract_python_code(response)
    if code is None:
        details["error"] = "no_code_found"
        return 0.0, details

    # Syntax check
    try:
        compile(code, "<benchmark>", "exec")
    except SyntaxError as e:
        details["error"] = f"syntax_error: {e}"
        return 0.1, details  # Small credit for attempting code

    details["syntax_valid"] = True

    # Execute code in restricted namespace
    namespace: dict[str, Any] = {"__builtins__": {
        "range": range, "len": len, "int": int, "float": float, "str": str,
        "bool": bool, "list": list, "dict": dict, "set": set, "tuple": tuple,
        "max": max, "min": min, "abs": abs, "sum": sum, "enumerate": enumerate,
        "zip": zip, "map": map, "filter": filter, "sorted": sorted,
        "reversed": reversed, "isinstance": isinstance, "type": type,
        "print": lambda *a, **kw: None,  # suppress output
        "True": True, "False": False, "None": None,
        "ValueError": ValueError, "TypeError": TypeError,
        "KeyError": KeyError, "IndexError": IndexError,
        "Exception": Exception, "StopIteration": StopIteration,
    }}

    # Set up timeout (Unix only)
    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(5)  # 5 second timeout
    except (OSError, AttributeError):
        pass  # signal.SIGALRM not available on this platform

    try:
        exec(code, namespace)  # noqa: S102
    except _CodeExecTimeout:
        details["error"] = "execution_timeout"
        return 0.15, details
    except Exception as e:
        details["error"] = f"execution_error: {type(e).__name__}: {e}"
        return 0.15, details
    finally:
        try:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except (OSError, AttributeError):
            pass

    # Run test cases
    passed = 0
    total = len(test_cases)
    test_results = []

    for tc in test_cases:
        func_name = tc.get("function", "")
        args = tc.get("input")
        expected = tc.get("expected")

        if not func_name or func_name not in namespace:
            # Try to find the first callable function defined in the code
            for name, val in namespace.items():
                if callable(val) and not name.startswith("_"):
                    func_name = name
                    break

        func = namespace.get(func_name)
        if not callable(func):
            test_results.append({"status": "skip", "reason": f"function '{func_name}' not found"})
            continue

        try:
            if old_handler is not None:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(2)

            if isinstance(args, list):
                result = func(*args)
            else:
                result = func(args)

            signal.alarm(0)

            if result == expected:
                passed += 1
                test_results.append({"status": "pass"})
            else:
                test_results.append({"status": "fail", "got": str(result), "expected": str(expected)})
        except _CodeExecTimeout:
            test_results.append({"status": "timeout"})
        except Exception as e:
            test_results.append({"status": "error", "error": f"{type(e).__name__}: {e}"})
        finally:
            try:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            except (OSError, AttributeError):
                pass

    details["tests_passed"] = passed
    details["tests_total"] = total
    details["test_results"] = test_results

    if total == 0:
        return 0.3, details  # Code ran but no tests to verify

    # Base score from test pass rate, with credit for valid syntax + execution
    pass_rate = passed / total
    score = 0.2 + 0.8 * pass_rate  # 0.2 baseline for valid, executable code
    return _clamp(score), details


def score_correctness(
    response: str,
    expected_answer: list[str] | None = None,
    answer_type: str | None = None,
    numeric_answer: float | None = None,
    numeric_tolerance: float = 0.01,
    test_cases: list[dict[str, object]] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Score correctness when verification data is available.

    Returns (score, details). When no verification data is provided,
    returns a neutral score of 0.5 (unknown correctness).
    """
    if answer_type == "contains" and expected_answer:
        return _check_contains(response, expected_answer)

    if answer_type == "numeric" and numeric_answer is not None:
        return _check_numeric(response, numeric_answer, numeric_tolerance)

    if answer_type == "code_test" and test_cases:
        return _check_code_execution(response, test_cases)

    # No verification data — return neutral
    return 0.5, {"check": "none"}


# --- Category-specific heuristic scorers ---


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

    # Syntax check — try to compile extracted code
    code = _extract_python_code(response)
    if code is not None:
        checks += 1
        try:
            compile(code, "<benchmark>", "exec")
            signals += 1
            details["syntax_valid"] = True
        except SyntaxError:
            details["syntax_valid"] = False

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
    prompt_numbers = set(re.findall(r"\d{4}|\d+\.\d+|\d+", prompt))
    response_numbers = set(re.findall(r"\d{4}|\d+\.\d+|\d+", response))

    overlap = prompt_numbers & response_numbers
    coverage = len(overlap) / len(prompt_numbers) if prompt_numbers else 0.5
    details["number_extraction_coverage"] = round(coverage, 3)

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
    ratio = len(response) / max(len(prompt), 1)
    details["length_ratio"] = round(ratio, 3)

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
    explanation_markers = [
        "because", "due to", "causes", "results in", "mechanism",
        "process", "occurs", "involves", "molecules", "atoms",
        "energy", "cells", "reaction", "system", "theory",
    ]
    found = [m for m in explanation_markers if m in lower]
    details["science_markers"] = found
    score = _clamp(len(found) / 3)
    return score, details


# --- Corpus IDF computation ---


def compute_corpus_idf(all_prompts: list[str]) -> dict[str, float]:
    """Compute IDF weights from a corpus of prompt texts.

    Words appearing in many prompts get low weight (they're generic),
    words appearing in few prompts get high weight (they're distinctive).
    """
    if not all_prompts:
        return {}

    n_docs = len(all_prompts)
    doc_freq: Counter[str] = Counter()
    for prompt in all_prompts:
        unique_words = set(_tokenize(prompt))
        for word in unique_words:
            doc_freq[word] += 1

    idf: dict[str, float] = {}
    for word, df in doc_freq.items():
        idf[word] = math.log(n_docs / df) + 1.0  # smoothed IDF

    return idf


# --- Main scoring interface ---


def score_response(
    prompt_text: str,
    prompt_category: str,
    response_text: str,
    finish_reason: str,
    max_tokens: int = 512,
    expected_answer: list[str] | None = None,
    answer_type: str | None = None,
    numeric_answer: float | None = None,
    numeric_tolerance: float = 0.01,
    test_cases: list[dict[str, object]] | None = None,
    corpus_idf: dict[str, float] | None = None,
) -> ResponseScore:
    """Score a single response across all quality dimensions."""
    length_s, length_d = score_length(response_text, prompt_category, max_tokens)
    coherence_s, coherence_d = score_coherence(response_text)
    relevance_s, relevance_d = score_relevance(prompt_text, response_text, corpus_idf)
    completeness_s, completeness_d = score_completeness(response_text, finish_reason)
    category_s, category_d = score_category(response_text, prompt_category, prompt_text)
    correctness_s, correctness_d = score_correctness(
        response_text,
        expected_answer=expected_answer,
        answer_type=answer_type,
        numeric_answer=numeric_answer,
        numeric_tolerance=numeric_tolerance,
        test_cases=test_cases,
    )

    has_correctness = answer_type is not None
    weights = _get_weights(prompt_category, has_correctness)

    overall = (
        length_s * weights["length"]
        + coherence_s * weights["coherence"]
        + relevance_s * weights["relevance"]
        + completeness_s * weights["completeness"]
        + category_s * weights["category"]
        + correctness_s * weights["correctness"]
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
        correctness_score=correctness_s,
        overall_score=round(overall, 3),
        details={
            "length": length_d,
            "coherence": coherence_d,
            "relevance": relevance_d,
            "completeness": completeness_d,
            "category": category_d,
            "correctness": correctness_d,
            "weights_used": weights,
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
