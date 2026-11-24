"""LLM Reader with Verifiable Evidence.

Extracts requirement atoms and slot scores using an LLM.
All claims are strictly verified against the prompt text using verbatim substring checks.
Claims lacking verbatim substring evidence are automatically discarded.
Negated claims (negated=True) are never asserted.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import app.env  # noqa: F401
from app.linter.rubrics import (
    ATOM_RUBRICS,
    RUBRIC_VERSION,
    SCHEMA_VERSION,
    SLOT_RUBRICS,
)

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / "data" / "llm_cache"
LOG_DIR = ROOT / "data" / "llm_logs"
DISCARDS_DEV_LOG = ROOT / "data" / "eval" / "diagnostics" / "reader_discards_dev.jsonl"

ATOM_NAMES = [
    "very_easy",
    "expert_level",
    "no_basic",
    "ordered_easy_to_hard",
    "hidden_answers",
    "answer_after_each_q",
    "single_correct",
    "multi_interp",
    "confusing_language",
]

SLOT_NAMES = [
    "role",
    "audience",
    "topic",
    "count",
    "n_options",
    "labeling",
    "marking",
    "explanation",
    "ordering",
]

# Six worked examples taken strictly from data/gold/dev_authored/dev_prompts.json
WORKED_EXAMPLES: list[dict[str, Any]] = [
    {
        "prompt": "Act as an academic instructor. Write 5 MCQs on CPU scheduling algorithms. All questions must be arranged in order of increasing difficulty from easiest to hardest. Provide four options per question and mark the right choice.",
        "analysis": {
            "atoms": {
                "ordered_easy_to_hard": {
                    "present": True,
                    "negated": False,
                    "evidence": "arranged in order of increasing difficulty from easiest to hardest",
                },
                "very_easy": {"present": False, "negated": False, "evidence": ""},
                "expert_level": {"present": False, "negated": False, "evidence": ""},
                "no_basic": {"present": False, "negated": False, "evidence": ""},
                "hidden_answers": {"present": False, "negated": False, "evidence": ""},
                "answer_after_each_q": {"present": False, "negated": False, "evidence": ""},
                "single_correct": {"present": False, "negated": False, "evidence": ""},
                "multi_interp": {"present": False, "negated": False, "evidence": ""},
                "confusing_language": {"present": False, "negated": False, "evidence": ""},
            },
            "slots": {
                "role": {"score": 1.0, "evidence": "Act as an academic instructor"},
                "audience": {"score": 0.0, "evidence": ""},
                "topic": {"score": 1.0, "evidence": "on CPU scheduling algorithms"},
                "count": {"score": 1.0, "evidence": "Write 5 MCQs"},
                "n_options": {"score": 1.0, "evidence": "four options per question"},
                "labeling": {"score": 0.0, "evidence": ""},
                "marking": {"score": 1.0, "evidence": "mark the right choice"},
                "explanation": {"score": 0.0, "evidence": ""},
                "ordering": {
                    "score": 1.0,
                    "evidence": "arranged in order of increasing difficulty from easiest to hardest",
                },
            },
        },
    },
    {
        "prompt": "Create 10 multiple choice questions about database management systems for bachelor students. All questions must be very easy. Give four options for each question.",
        "analysis": {
            "atoms": {
                "very_easy": {"present": True, "negated": False, "evidence": "All questions must be very easy"},
                "expert_level": {"present": False, "negated": False, "evidence": ""},
                "no_basic": {"present": False, "negated": False, "evidence": ""},
                "ordered_easy_to_hard": {"present": False, "negated": False, "evidence": ""},
                "hidden_answers": {"present": False, "negated": False, "evidence": ""},
                "answer_after_each_q": {"present": False, "negated": False, "evidence": ""},
                "single_correct": {"present": False, "negated": False, "evidence": ""},
                "multi_interp": {"present": False, "negated": False, "evidence": ""},
                "confusing_language": {"present": False, "negated": False, "evidence": ""},
            },
            "slots": {
                "role": {"score": 0.0, "evidence": ""},
                "audience": {"score": 1.0, "evidence": "for bachelor students"},
                "topic": {"score": 1.0, "evidence": "about database management systems"},
                "count": {"score": 1.0, "evidence": "Create 10 multiple choice questions"},
                "n_options": {"score": 1.0, "evidence": "four options for each question"},
                "labeling": {"score": 0.0, "evidence": ""},
                "marking": {"score": 0.0, "evidence": ""},
                "explanation": {"score": 0.0, "evidence": ""},
                "ordering": {"score": 0.0, "evidence": ""},
            },
        },
    },
    {
        "prompt": "Act as a professor. Create 10 MCQs on machine learning. Questions must require expert-level knowledge. Absolutely no basic questions. Four choices per question.",
        "analysis": {
            "atoms": {
                "expert_level": {
                    "present": True,
                    "negated": False,
                    "evidence": "Questions must require expert-level knowledge",
                },
                "no_basic": {"present": True, "negated": False, "evidence": "no basic questions"},
                "very_easy": {"present": False, "negated": False, "evidence": ""},
                "ordered_easy_to_hard": {"present": False, "negated": False, "evidence": ""},
                "hidden_answers": {"present": False, "negated": False, "evidence": ""},
                "answer_after_each_q": {"present": False, "negated": False, "evidence": ""},
                "single_correct": {"present": False, "negated": False, "evidence": ""},
                "multi_interp": {"present": False, "negated": False, "evidence": ""},
                "confusing_language": {"present": False, "negated": False, "evidence": ""},
            },
            "slots": {
                "role": {"score": 1.0, "evidence": "Act as a professor"},
                "audience": {"score": 0.0, "evidence": ""},
                "topic": {"score": 1.0, "evidence": "on machine learning"},
                "count": {"score": 1.0, "evidence": "Create 10 MCQs"},
                "n_options": {"score": 1.0, "evidence": "Four choices per question"},
                "labeling": {"score": 0.0, "evidence": ""},
                "marking": {"score": 0.0, "evidence": ""},
                "explanation": {"score": 0.0, "evidence": ""},
                "ordering": {"score": 0.0, "evidence": ""},
            },
        },
    },
    {
        "prompt": "Act as an academic instructor. Prepare 8 multiple-choice questions on software testing. Four options labeled A to D for each question. Do not show answers.",
        "analysis": {
            "atoms": {
                "hidden_answers": {"present": True, "negated": False, "evidence": "Do not show answers"},
                "very_easy": {"present": False, "negated": False, "evidence": ""},
                "expert_level": {"present": False, "negated": False, "evidence": ""},
                "no_basic": {"present": False, "negated": False, "evidence": ""},
                "ordered_easy_to_hard": {"present": False, "negated": False, "evidence": ""},
                "answer_after_each_q": {"present": False, "negated": False, "evidence": ""},
                "single_correct": {"present": False, "negated": False, "evidence": ""},
                "multi_interp": {"present": False, "negated": False, "evidence": ""},
                "confusing_language": {"present": False, "negated": False, "evidence": ""},
            },
            "slots": {
                "role": {"score": 1.0, "evidence": "Act as an academic instructor"},
                "audience": {"score": 0.0, "evidence": ""},
                "topic": {"score": 1.0, "evidence": "on software testing"},
                "count": {"score": 1.0, "evidence": "Prepare 8 multiple-choice questions"},
                "n_options": {"score": 1.0, "evidence": "Four options labeled A to D"},
                "labeling": {"score": 1.0, "evidence": "labeled A to D"},
                "marking": {"score": 1.0, "evidence": "Do not show answers"},
                "explanation": {"score": 0.0, "evidence": ""},
                "ordering": {"score": 0.0, "evidence": ""},
            },
        },
    },
    {
        "prompt": "Act as a university professor. Prepare 20 MCQs on distributed systems for postgraduate students. All items must demand expert-level understanding. Absolutely no basic or easy questions are permitted. Each question must have four options A to D with exactly one right answer. After every question show the answer and explain why it is correct.",
        "analysis": {
            "atoms": {
                "expert_level": {
                    "present": True,
                    "negated": False,
                    "evidence": "demand expert-level understanding",
                },
                "no_basic": {
                    "present": True,
                    "negated": False,
                    "evidence": "no basic or easy questions are permitted",
                },
                "single_correct": {"present": True, "negated": False, "evidence": "with exactly one right answer"},
                "answer_after_each_q": {
                    "present": True,
                    "negated": False,
                    "evidence": "After every question show the answer",
                },
                "very_easy": {"present": False, "negated": False, "evidence": ""},
                "ordered_easy_to_hard": {"present": False, "negated": False, "evidence": ""},
                "hidden_answers": {"present": False, "negated": False, "evidence": ""},
                "multi_interp": {"present": False, "negated": False, "evidence": ""},
                "confusing_language": {"present": False, "negated": False, "evidence": ""},
            },
            "slots": {
                "role": {"score": 1.0, "evidence": "Act as a university professor"},
                "audience": {"score": 1.0, "evidence": "for postgraduate students"},
                "topic": {"score": 1.0, "evidence": "on distributed systems"},
                "count": {"score": 1.0, "evidence": "Prepare 20 MCQs"},
                "n_options": {"score": 1.0, "evidence": "four options A to D"},
                "labeling": {"score": 1.0, "evidence": "options A to D"},
                "marking": {"score": 1.0, "evidence": "with exactly one right answer"},
                "explanation": {"score": 1.0, "evidence": "explain why it is correct"},
                "ordering": {"score": 0.0, "evidence": ""},
            },
        },
    },
    {
        "prompt": "Role: college instructor. Create 10 MCQs on computer security for undergraduate students. Four options per question with strictly one correct choice. Keep answers hidden. Arrange questions properly.",
        "analysis": {
            "atoms": {
                "single_correct": {
                    "present": True,
                    "negated": False,
                    "evidence": "strictly one correct choice",
                },
                "hidden_answers": {"present": True, "negated": False, "evidence": "Keep answers hidden"},
                "very_easy": {"present": False, "negated": False, "evidence": ""},
                "expert_level": {"present": False, "negated": False, "evidence": ""},
                "no_basic": {"present": False, "negated": False, "evidence": ""},
                "ordered_easy_to_hard": {"present": False, "negated": False, "evidence": ""},
                "answer_after_each_q": {"present": False, "negated": False, "evidence": ""},
                "multi_interp": {"present": False, "negated": False, "evidence": ""},
                "confusing_language": {"present": False, "negated": False, "evidence": ""},
            },
            "slots": {
                "role": {"score": 1.0, "evidence": "Role: college instructor"},
                "audience": {"score": 1.0, "evidence": "for undergraduate students"},
                "topic": {"score": 1.0, "evidence": "on computer security"},
                "count": {"score": 1.0, "evidence": "Create 10 MCQs"},
                "n_options": {"score": 1.0, "evidence": "Four options per question"},
                "labeling": {"score": 0.0, "evidence": ""},
                "marking": {"score": 1.0, "evidence": "strictly one correct choice"},
                "explanation": {"score": 0.0, "evidence": ""},
                "ordering": {"score": 0.5, "evidence": "Arrange questions properly"},
            },
        },
    },
]


def is_reader_enabled() -> bool:
    """Check if the LLM reader is enabled via environment variables."""
    for var in ("STUDY_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"):
        val = (os.environ.get(var) or "").strip()
        if val:
            return True
    return False


def normalize_text_for_quote(s: str) -> str:
    """Normalize whitespace and lowercase for strict quote substring matching."""
    return " ".join(s.strip().split()).lower()


def check_quote_strict(quote: str, prompt_text: str) -> bool:
    """Strict quote check: whitespace normalized and lowercased only."""
    if not quote or not str(quote).strip():
        return False
    norm_quote = normalize_text_for_quote(quote)
    norm_prompt = normalize_text_for_quote(prompt_text)
    return norm_quote in norm_prompt


def normalize_quotes_and_dashes(s: str) -> str:
    """Normalize whitespace, lowercase, curly vs straight quotes, and dash variants."""
    s = " ".join(s.strip().split()).lower()
    # Curly quotes to straight quotes
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("‘", "'").replace("’", "'").replace("′", "'").replace("`", "'")
    # Dash variants to standard hyphen
    s = s.replace("—", "-").replace("–", "-").replace("−", "-").replace("‐", "-")
    return s


def check_quote_normalized(quote: str, prompt_text: str) -> bool:
    """Normalized quote check: whitespace, curly quotes, and dash variants normalized."""
    if not quote or not str(quote).strip():
        return False
    norm_quote = normalize_quotes_and_dashes(quote)
    norm_prompt = normalize_quotes_and_dashes(prompt_text)
    return norm_quote in norm_prompt


def is_verbatim_quote(quote: str, prompt_text: str, mode: str = "strict") -> bool:
    """Verbatim quote check.

    mode == 'strict': whitespace normalisation only.
    mode == 'normalized': whitespace, curly quotes, and dash variants normalised.
    """
    if mode == "normalized":
        return check_quote_normalized(quote, prompt_text)
    return check_quote_strict(quote, prompt_text)


def find_quote_span(quote: str, prompt_text: str, mode: str = "strict") -> tuple[int, int] | None:
    """Find character start and end index in original prompt text (case-insensitive, whitespace-flexible)."""
    if not quote or not quote.strip():
        return None
    # Escape quote but allow arbitrary whitespace between tokens
    clean_quote = quote.strip()
    tokens = [re.escape(t) for t in clean_quote.split()]
    pattern = r"\s+".join(tokens)
    match = re.search(pattern, prompt_text, re.IGNORECASE)
    if match:
        return match.start(), match.end()

    if mode == "normalized":
        # Allow curly quotes or dash variants in regex matching
        alt_pattern = pattern
        alt_pattern = alt_pattern.replace('"', '["“”]')
        alt_pattern = alt_pattern.replace("'", "['‘’′`]")
        alt_pattern = alt_pattern.replace("-", "[-—–−‐]")
        match_alt = re.search(alt_pattern, prompt_text, re.IGNORECASE)
        if match_alt:
            return match_alt.start(), match_alt.end()

    return None


def is_negated_phrase(phrase: str) -> bool:
    """Check if an evidence quote contains explicit negation markers."""
    norm = normalize_text_for_quote(phrase)
    neg_pattern = re.compile(
        r"\b(?:not\s+easy|not\s+very\s+easy|no\s+very\s+easy|no\s+easy|no\s+basic|do\s+not|don't|never|not|without|neither)\b",
        re.IGNORECASE,
    )
    return bool(neg_pattern.search(norm))


def get_system_prompt() -> str:
    """Construct full system prompt with definitions, rubrics, and worked examples."""
    lines = [
        "You are an expert educational requirement extractor for multiple-choice question prompts.",
        "Your job is to analyze the prompt and extract requirements for 9 governed atoms and 9 governed slots.",
        "Output MUST be valid JSON adhering strictly to the requested schema.",
        "",
        "RULES FOR EVIDENCE QUOTING (MANDATORY):",
        "1. For every atom that is present, you MUST provide an exact, verbatim quote from the prompt.",
        "2. For every slot scored 0.5 or 1.0, you MUST provide an exact, verbatim quote from the prompt supporting the score.",
        "3. If an atom is absent or a slot is scored 0.0, evidence MUST be an empty string.",
        "4. Any quote that is not a verbatim substring of the user prompt will be rejected by automated code checks.",
        "5. If a requirement is negated (e.g. 'not easy', 'no basic questions', 'do not show answers', 'do not use tricky wording'), set negated: true.",
        "",
        "### GOVERNED ATOMS (9 items):",
    ]
    for name, r in ATOM_RUBRICS.items():
        lines.append(f"- **{name}** ({r.name}): {r.description}")
        lines.append(f"  Positive rule: {r.positive_rule}")
        lines.append(f"  Negative rule: {r.negative_rule}")

    lines.append("\n### GOVERNED SLOTS (9 items, strictly 0.0, 0.5, or 1.0):")
    for slot, sr in SLOT_RUBRICS.items():
        lines.append(f"- **{slot}** ({sr.name}): {sr.description}")
        lines.append(f"  1.0: {sr.score_1_0_rule}")
        lines.append(f"  0.5: {sr.score_0_5_rule}")
        lines.append(f"  0.0: {sr.score_0_rule}")

    lines.append("\n### WORKED EXAMPLES (from development dataset):")
    for i, ex in enumerate(WORKED_EXAMPLES, 1):
        lines.append(f"\nExample {i}:")
        lines.append(f'Prompt: "{ex["prompt"]}"')
        lines.append(f"Output: {json.dumps(ex['analysis'])}")

    lines.append("\nNow analyze the user's prompt and return ONLY the JSON object.")
    return "\n".join(lines)


def get_cache_key(prompt_text: str, model_name: str) -> str:
    """Compute deterministic SHA-256 cache key."""
    raw = f"{prompt_text.strip()}|{model_name}|{RUBRIC_VERSION}|{SCHEMA_VERSION}"
    return sha256(raw.encode("utf-8")).hexdigest()


def call_llm_api(
    prompt_text: str,
    model: str,
    api_key: str,
    api_base: str,
    seed: int = 42,
    temperature: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Execute API call with fixed seed and temperature 0.

    Returns (raw_request, raw_response_json, latency_seconds).
    """
    system_prompt = get_system_prompt()
    payload = {
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ],
    }

    req = urllib.request.Request(
        api_base,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "PromptRepositoryLab/0.1.0",
        },
        method="POST",
    )

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        resp_data = json.loads(resp.read().decode("utf-8"))
    latency = time.time() - t0

    return payload, resp_data, latency


def classify_failure_reason(error_str: str, raw_content: str = "") -> str:
    """Classify why the reader was unavailable or failed: no_key, rate_limit, malformed_output, network_error."""
    err_low = (error_str or "").lower()
    if "429" in err_low or "rate limit" in err_low or "rate_limit" in err_low:
        return "rate_limit"
    if (
        "json" in err_low
        or "missing atoms or slots" in err_low
        or "expecting value" in err_low
        or "invalid character" in err_low
        or "unterminated" in err_low
    ):
        return "malformed_output"
    return "network_error"


def validate_and_filter_reader_output(
    raw_analysis: dict[str, Any],
    prompt_text: str,
    quote_mode: str = "strict",
    prompt_id: str | None = None,
    is_dev: bool = False,
    selected_model: str = "",
) -> dict[str, Any]:
    """Code-enforced validation of LLM claims against the prompt text.

    - Every quote must be a verbatim substring of the prompt.
    - If quote is missing or invalid, claim is marked unsupported and discarded.
    - If negated is True (or quote contains negation), atom is NOT asserted.
    - For dev data only, discarded claims are logged to reader_discards_dev.jsonl.
    """
    raw_atoms = raw_analysis.get("atoms", {})
    raw_slots = raw_analysis.get("slots", {})

    validated_atoms: dict[str, dict[str, Any]] = {}
    accepted_spans: list[dict[str, Any]] = []

    per_atom_diag: dict[str, dict[str, int]] = {}
    per_slot_diag: dict[str, dict[str, int]] = {}

    for atom in ATOM_NAMES:
        item = raw_atoms.get(atom, {})
        present = bool(item.get("present", False))
        negated = bool(item.get("negated", False))
        evidence = str(item.get("evidence", "")).strip()

        strict_quote_valid = check_quote_strict(evidence, prompt_text) if evidence else False
        norm_quote_valid = check_quote_normalized(evidence, prompt_text) if evidence else False
        quote_valid = norm_quote_valid if quote_mode == "normalized" else strict_quote_valid

        # Check for explicit negation in evidence
        if evidence and is_negated_phrase(evidence):
            negated = True

        # Evidence requirement: discard claim if verbatim substring evidence is absent
        unsupported = False
        if present and not quote_valid:
            present = False
            unsupported = True

        # Negation check: atoms marked negated are not asserted
        asserted = present and (not negated) and quote_valid

        span = find_quote_span(evidence, prompt_text, mode=quote_mode) if (asserted and quote_valid) else None
        if asserted and span:
            accepted_spans.append(
                {
                    "item_type": "atom",
                    "item_name": atom,
                    "quote": evidence,
                    "start": span[0],
                    "end": span[1],
                }
            )

        # Track per-atom diagnostics
        raw_claimed = bool(item.get("present", False))
        raw_neg = bool(item.get("negated", False)) or (bool(evidence) and is_negated_phrase(evidence))
        strict_acc = int(raw_claimed and (not raw_neg) and strict_quote_valid)
        norm_acc = int(raw_claimed and (not raw_neg) and norm_quote_valid)

        per_atom_diag[atom] = {
            "claims_asserted": 1 if raw_claimed else 0,
            "claims_discarded_quote": 1 if (raw_claimed and not quote_valid) else 0,
            "claims_discarded_negated": 1 if (raw_claimed and raw_neg) else 0,
            "claims_accepted": 1 if asserted else 0,
            "strict_accepted": strict_acc,
            "norm_accepted": norm_acc,
        }

        # For discarded claims on dev data only: log model quote and prompt text
        if is_dev and raw_claimed and not quote_valid and prompt_text:
            try:
                DISCARDS_DEV_LOG.parent.mkdir(parents=True, exist_ok=True)
                entry = {
                    "prompt_id": prompt_id or "",
                    "item_type": "atom",
                    "item_name": atom,
                    "model": selected_model,
                    "quote": evidence,
                    "prompt_text": prompt_text,
                    "discard_reason": "not_verbatim_quote",
                    "strict_valid": strict_quote_valid,
                    "norm_valid": norm_quote_valid,
                }
                with open(DISCARDS_DEV_LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception:
                pass

        validated_atoms[atom] = {
            "present": present,
            "negated": negated,
            "asserted": asserted,
            "evidence": evidence,
            "quote_valid": quote_valid,
            "strict_quote_valid": strict_quote_valid,
            "norm_quote_valid": norm_quote_valid,
            "unsupported": unsupported,
            "span": span,
        }

    validated_slots: dict[str, dict[str, Any]] = {}
    for slot in SLOT_NAMES:
        item = raw_slots.get(slot, {})
        score_val = item.get("score", 0.0)
        try:
            score = float(score_val)
            if score not in (0.0, 0.5, 1.0):
                score = 0.0
        except (ValueError, TypeError):
            score = 0.0

        evidence = str(item.get("evidence", "")).strip()
        strict_quote_valid = check_quote_strict(evidence, prompt_text) if evidence else False
        norm_quote_valid = check_quote_normalized(evidence, prompt_text) if evidence else False
        quote_valid = norm_quote_valid if quote_mode == "normalized" else strict_quote_valid

        claimed_present = score in (0.5, 1.0)
        unsupported = False
        if claimed_present and not quote_valid:
            score = 0.0
            unsupported = True

        span = find_quote_span(evidence, prompt_text, mode=quote_mode) if (score > 0 and quote_valid) else None
        if score > 0 and span:
            accepted_spans.append(
                {
                    "item_type": "slot",
                    "item_name": slot,
                    "score": score,
                    "quote": evidence,
                    "start": span[0],
                    "end": span[1],
                }
            )

        strict_acc = int(claimed_present and strict_quote_valid)
        norm_acc = int(claimed_present and norm_quote_valid)

        per_slot_diag[slot] = {
            "claims_asserted": 1 if claimed_present else 0,
            "claims_discarded_quote": 1 if (claimed_present and not quote_valid) else 0,
            "claims_accepted": 1 if (claimed_present and quote_valid) else 0,
            "strict_accepted": strict_acc,
            "norm_accepted": norm_acc,
        }

        # For discarded claims on dev data only: log model quote and prompt text
        if is_dev and claimed_present and not quote_valid and prompt_text:
            try:
                DISCARDS_DEV_LOG.parent.mkdir(parents=True, exist_ok=True)
                entry = {
                    "prompt_id": prompt_id or "",
                    "item_type": "slot",
                    "item_name": slot,
                    "model": selected_model,
                    "quote": evidence,
                    "prompt_text": prompt_text,
                    "discard_reason": "not_verbatim_quote",
                    "strict_valid": strict_quote_valid,
                    "norm_valid": norm_quote_valid,
                }
                with open(DISCARDS_DEV_LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception:
                pass

        validated_slots[slot] = {
            "score": score,
            "evidence": evidence,
            "quote_valid": quote_valid,
            "strict_quote_valid": strict_quote_valid,
            "norm_quote_valid": norm_quote_valid,
            "unsupported": unsupported,
            "span": span,
        }

    return {
        "atoms": validated_atoms,
        "slots": validated_slots,
        "accepted_spans": accepted_spans,
        "diagnostics": {
            "per_atom": per_atom_diag,
            "per_slot": per_slot_diag,
            "claims_asserted_total": sum(v["claims_asserted"] for v in per_atom_diag.values())
            + sum(v["claims_asserted"] for v in per_slot_diag.values()),
            "claims_discarded_quote_total": sum(v["claims_discarded_quote"] for v in per_atom_diag.values())
            + sum(v["claims_discarded_quote"] for v in per_slot_diag.values()),
            "claims_discarded_negated_total": sum(v["claims_discarded_negated"] for v in per_atom_diag.values()),
            "claims_accepted_total": sum(v["claims_accepted"] for v in per_atom_diag.values())
            + sum(v["claims_accepted"] for v in per_slot_diag.values()),
            "strict_accepted_total": sum(v["strict_accepted"] for v in per_atom_diag.values())
            + sum(v["strict_accepted"] for v in per_slot_diag.values()),
            "norm_accepted_total": sum(v["norm_accepted"] for v in per_atom_diag.values())
            + sum(v["norm_accepted"] for v in per_slot_diag.values()),
        },
    }


def read_prompt_with_llm(
    prompt_text: str,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    use_cache: bool = True,
    prompt_id: str | None = None,
    is_dev: bool = False,
    quote_mode: str = "strict",
) -> dict[str, Any]:
    """Main LLM reader entry point.

    - Returns disabled status if no key is available.
    - Caches results by deterministic SHA-256.
    - Retries once on malformed JSON.
    - Validates all evidence quotes in code.
    - Records detailed diagnostics.
    """
    empty_diag: dict[str, Any] = {
        "calls_made": 0,
        "calls_returned": 0,
        "malformed_responses": 0,
        "retries": 0,
        "reader_failed": 0,
        "claims_asserted_total": 0,
        "claims_discarded_quote_total": 0,
        "claims_discarded_negated_total": 0,
        "claims_accepted_total": 0,
        "strict_accepted_total": 0,
        "norm_accepted_total": 0,
        "per_atom": {
            a: {
                "claims_asserted": 0,
                "claims_discarded_quote": 0,
                "claims_discarded_negated": 0,
                "claims_accepted": 0,
                "strict_accepted": 0,
                "norm_accepted": 0,
            }
            for a in ATOM_NAMES
        },
        "per_slot": {
            s: {
                "claims_asserted": 0,
                "claims_discarded_quote": 0,
                "claims_accepted": 0,
                "strict_accepted": 0,
                "norm_accepted": 0,
            }
            for s in SLOT_NAMES
        },
    }

    key = (
        (api_key or "").strip()
        or (os.environ.get("STUDY_API_KEY") or "").strip()
        or (os.environ.get("GROQ_API_KEY") or "").strip()
        or (os.environ.get("OPENAI_API_KEY") or "").strip()
    )
    if not key:
        return {
            "enabled": False,
            "status": "disabled_no_api_key",
            "reader_unavailable_reason": "no_key",
            "message": "Reader is off: no API key set. Running on rules only.",
            "atoms": {},
            "slots": {},
            "accepted_spans": [],
            "diagnostics": empty_diag,
        }

    is_groq = bool((os.environ.get("GROQ_API_KEY") or "").strip()) and not os.environ.get("STUDY_API_BASE")
    default_endpoint = (
        "https://api.groq.com/openai/v1/chat/completions" if is_groq else "https://api.openai.com/v1/chat/completions"
    )
    default_model = "llama-3.3-70b-versatile" if is_groq else "gpt-4o-mini"

    selected_model: str = str(model or os.environ.get("STUDY_MODEL") or default_model)
    endpoint: str = str(api_base or os.environ.get("STUDY_API_BASE") or default_endpoint)

    cache_key = get_cache_key(prompt_text, selected_model)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    # Check cache
    if use_cache and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            raw_analysis = cached.get("raw_analysis")
            if not raw_analysis:
                raw_analysis = {
                    "atoms": cached.get("atoms", {}),
                    "slots": cached.get("slots", {}),
                }
            validated = validate_and_filter_reader_output(
                raw_analysis,
                prompt_text,
                quote_mode=quote_mode,
                prompt_id=prompt_id,
                is_dev=is_dev,
                selected_model=selected_model,
            )
            return {
                **cached,
                "cached": True,
                "status": "cached",
                "reader_unavailable_reason": None,
                **validated,
                "diagnostics": {
                    "calls_made": 0,
                    "calls_returned": 0,
                    "malformed_responses": 0,
                    "retries": 0,
                    "reader_failed": 0,
                    **validated["diagnostics"],
                },
            }
        except Exception:
            pass

    # Call API with retry once on malformed JSON
    max_attempts = 2
    raw_content = ""
    parsed_json: dict[str, Any] | None = None
    last_error = ""
    latency_total = 0.0
    token_usage: dict[str, Any] = {}
    attempts_made = 0
    calls_returned = 0
    malformed_count = 0
    retries_count = 0

    for attempt in range(1, max_attempts + 1):
        attempts_made += 1
        if attempt > 1:
            retries_count += 1
        try:
            req_payload, resp_data, lat = call_llm_api(prompt_text, selected_model, key, endpoint)
            latency_total += lat
            calls_returned += 1
            token_usage = resp_data.get("usage", {})
            choice = resp_data.get("choices", [{}])[0]
            raw_content = choice.get("message", {}).get("content", "")

            # Log request and response without credentials
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            log_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{cache_key[:8]}_att{attempt}"
            log_path = LOG_DIR / f"{log_id}.json"
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": selected_model,
                "latency_seconds": lat,
                "attempt": attempt,
                "token_usage": token_usage,
                "request_payload": req_payload,
                "response_body": resp_data,
            }
            log_path.write_text(json.dumps(log_entry, indent=2), encoding="utf-8")

            try:
                parsed_json = json.loads(raw_content)
                if isinstance(parsed_json, dict) and "atoms" in parsed_json and "slots" in parsed_json:
                    break
                else:
                    malformed_count += 1
                    last_error = "Missing atoms or slots in JSON"
                    parsed_json = None
            except Exception as parse_err:
                malformed_count += 1
                last_error = f"Malformed JSON: {parse_err}"
                parsed_json = None
        except Exception as e:
            last_error = str(e)
            parsed_json = None

    if parsed_json is None:
        reason = classify_failure_reason(last_error, raw_content)
        return {
            "enabled": True,
            "status": "reader_failed",
            "reader_unavailable_reason": reason,
            "message": f"LLM reader failed after {max_attempts} attempts: {last_error}. Falling back to rules only.",
            "model": selected_model,
            "latency_seconds": latency_total,
            "atoms": {},
            "slots": {},
            "accepted_spans": [],
            "diagnostics": {
                **empty_diag,
                "calls_made": attempts_made,
                "calls_returned": calls_returned,
                "malformed_responses": malformed_count,
                "retries": retries_count,
                "reader_failed": 1,
            },
        }

    validated = validate_and_filter_reader_output(
        parsed_json,
        prompt_text,
        quote_mode=quote_mode,
        prompt_id=prompt_id,
        is_dev=is_dev,
        selected_model=selected_model,
    )
    result = {
        "enabled": True,
        "status": "success",
        "reader_unavailable_reason": None,
        "model": selected_model,
        "latency_seconds": latency_total,
        "token_usage": token_usage,
        "rubric_version": RUBRIC_VERSION,
        "schema_version": SCHEMA_VERSION,
        "raw_analysis": parsed_json,
        **validated,
        "diagnostics": {
            "calls_made": attempts_made,
            "calls_returned": calls_returned,
            "malformed_responses": malformed_count,
            "retries": retries_count,
            "reader_failed": 0,
            **validated["diagnostics"],
        },
    }

    # Store cache
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result
