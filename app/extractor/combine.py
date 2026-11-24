"""Combine rule-based extraction and LLM reader verdicts with verifiable evidence.

Task 2:
- agree: reader and rules say the same, and reader quote passed verbatim check -> Auto-accept.
- reader_only: reader asserts, rules do not, quote passed check -> needs_review.
- rules_only: rules assert, reader does not -> needs_review.
- disagree_value: both give slot score and they differ -> Take neither -> needs_review.
- unsupported: reader asserted without valid quote -> Ignore claim, log it.
- Human decisions in manual review queue override both machines.
- Energy range [energy_min, energy_max] returned when any atom needs review.
- Improve gate returns verified only when no atom or slot is needs_review or unclassified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from app.auth_db import connect
from app.extractor.llm_reader import (
    ATOM_NAMES,
    SLOT_NAMES,
    read_prompt_with_llm,
)
from app.linter.core import evaluate, extract_atoms, score_slots
from app.linter.lexicon import find_unclassified_clauses, split_sentences

ROOT = Path(__file__).resolve().parent.parent.parent
CUE_LEXICON_PATH = ROOT / "data" / "gold" / "cue_lexicon.json"


def load_cue_lexicon() -> list[str]:
    """Load cue phrases from data/gold/cue_lexicon.json."""
    if not CUE_LEXICON_PATH.exists():
        return []
    try:
        data = json.loads(CUE_LEXICON_PATH.read_text(encoding="utf-8"))
        cues: list[str] = list(data.get("weak_cues", []))
        for fam_items in data.get("families", {}).values():
            cues.extend(fam_items)
        seen = set()
        unique_cues = []
        for c in cues:
            clow = str(c).strip().lower()
            if clow and clow not in seen:
                seen.add(clow)
                unique_cues.append(clow)
        return unique_cues
    except Exception:
        return []


def find_unrecognised_cue_flags(
    prompt: str,
    accepted_atoms: list[str],
    highlight_spans: list[dict[str, Any]],
    human_decisions: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Flag sentences containing requirement-like phrasing where no accepted atom was found."""
    cues = load_cue_lexicon()
    if not cues:
        return [], []

    sentences = split_sentences(prompt)
    flags: list[dict[str, Any]] = []
    open_flag_items: list[dict[str, Any]] = []

    pos = 0
    for sentence in sentences:
        clean_sentence = sentence.strip()
        if not clean_sentence:
            continue

        # Find character span of sentence in prompt
        start_char = prompt.find(clean_sentence, pos)
        if start_char == -1:
            start_char = prompt.find(clean_sentence)
        if start_char == -1:
            start_char = pos
            end_char = pos + len(clean_sentence)
        else:
            end_char = start_char + len(clean_sentence)
            pos = end_char

        # Ignore short header lines like 'Format:' or 'Task:'
        words = clean_sentence.split()
        if len(words) <= 2 and clean_sentence.rstrip().endswith(":"):
            continue

        # Check if an accepted atom was found in this sentence
        has_accepted = False
        for s in highlight_spans:
            if s.get("item_type") == "atom" and (
                s.get("item_name") in accepted_atoms or s.get("item_name") == "clear_unambiguous"
            ):
                s_start, s_end = s.get("start", -1), s.get("end", -1)
                if s_start != -1 and max(s_start, start_char) < min(s_end, end_char):
                    has_accepted = True
                    break

        if not has_accepted:
            # Check rule-based extraction on this specific sentence
            sentence_atoms = set(extract_atoms(clean_sentence))
            if (sentence_atoms & set(accepted_atoms)) or ("clear_unambiguous" in sentence_atoms):
                has_accepted = True

        if has_accepted:
            continue

        # Check for presence of weak or family cues in sentence
        matched_cue = None
        sentence_lower = clean_sentence.lower()
        for cue in cues:
            if " " in cue:
                if cue in sentence_lower:
                    matched_cue = cue
                    break
            else:
                pattern = r"\b" + re.escape(cue) + r"\b"
                if re.search(pattern, sentence_lower):
                    matched_cue = cue
                    break

        if matched_cue:
            message = (
                f"This sentence contains requirement-like phrasing that the system does not recognise: {clean_sentence}"
            )
            flag = {
                "item_type": "cue_flag",
                "item_name": "possible_unrecognised_requirement",
                "sentence": clean_sentence,
                "matched_cue": matched_cue,
                "start": start_char,
                "end": end_char,
                "message": message,
            }
            flags.append(flag)

            # Check human decision
            h_dec = (human_decisions or {}).get(("cue_flag", clean_sentence))
            if h_dec is None:
                open_flag_items.append(
                    {
                        "item_type": "cue_flag",
                        "item_name": "possible_unrecognised_requirement",
                        "verdict": "possible_unrecognised_requirement",
                        "quote": clean_sentence,
                        "rule_opinion": f"Cue detected: {matched_cue}",
                        "reader_opinion": None,
                        "message": message,
                    }
                )

    return flags, open_flag_items


@dataclass
class CombinedExtractionResult:
    prompt: str
    prompt_id: str | None
    reader_enabled: bool
    reader_status: str
    reader_message: str | None
    accepted_atoms: list[str]
    review_atoms: list[str]
    atom_verdicts: dict[str, dict[str, Any]]
    accepted_slots: dict[str, float]
    review_slots: list[str]
    slot_verdicts: dict[str, dict[str, Any]]
    energy: int
    energy_range: list[int]
    energy_min: int
    energy_max: int
    has_review_range: bool
    review_reason: str | None
    completeness: float
    highlight_spans: list[dict[str, Any]]
    open_items: list[dict[str, Any]]
    unclassified_clauses: list[str]
    verified_ready: bool
    energy_lower_bound: int = 0
    energy_is_lower_bound: bool = True
    energy_display: str = ""
    cue_flags: list[dict[str, Any]] | None = None
    reader_unavailable_reason: str | None = None


def get_existing_human_decisions(prompt_id: str | None, prompt_text: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Retrieve any existing human review decisions for this prompt from SQLite."""
    decisions: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        with connect() as db:
            rows = db.execute(
                "SELECT item_type, item_name, status, decision_value, decided_by, decided_at "
                "FROM manual_reviews WHERE (prompt_id=? OR prompt_text=?) AND status IN ('confirmed', 'rejected')",
                (prompt_id or "", prompt_text),
            ).fetchall()
            for r in rows:
                decisions[(r["item_type"], r["item_name"])] = {
                    "status": r["status"],
                    "decision_value": r["decision_value"],
                    "decided_by": r["decided_by"],
                    "decided_at": r["decided_at"],
                }
    except Exception:
        pass
    return decisions


def queue_open_items_in_db(
    prompt_id: str | None,
    prompt_text: str,
    open_items: list[dict[str, Any]],
) -> None:
    """Store open items needing human review in the database queue if not already present."""
    if not open_items:
        return
    try:
        with connect() as db:
            for item in open_items:
                # Check if item is already in queue
                existing = db.execute(
                    "SELECT id, status FROM manual_reviews "
                    "WHERE (prompt_id=? OR prompt_text=?) AND item_type=? AND item_name=?",
                    (prompt_id or "", prompt_text, item["item_type"], item["item_name"]),
                ).fetchone()
                if not existing:
                    db.execute(
                        "INSERT INTO manual_reviews(prompt_id, prompt_text, item_type, item_name, verdict, quote, rule_opinion, reader_opinion, status) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')",
                        (
                            prompt_id or "adhoc",
                            prompt_text,
                            item["item_type"],
                            item["item_name"],
                            item["verdict"],
                            item.get("quote") or "",
                            str(item.get("rule_opinion", "")),
                            str(item.get("reader_opinion", "")),
                        ),
                    )
    except Exception:
        pass


def combine_rules_and_reader(
    prompt: str,
    prompt_id: str | None = None,
    reader_output: dict[str, Any] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    record_reviews: bool = True,
    policy_atoms: tuple[str, ...] | None = None,
    is_dev: bool = False,
    quote_mode: str = "strict",
) -> CombinedExtractionResult:
    """Combine rule-based extraction and LLM reader verdicts with verifiable evidence."""
    # 1. Rule-based extraction
    rule_atoms = extract_atoms(prompt)
    rule_slots = score_slots(prompt)

    # 2. LLM reader output
    if reader_output is None:
        reader_output = read_prompt_with_llm(
            prompt,
            model=model,
            api_key=api_key,
            api_base=api_base,
            prompt_id=prompt_id,
            is_dev=is_dev,
            quote_mode=quote_mode,
        )

    reader_enabled = bool(reader_output.get("enabled", False))
    reader_status = str(reader_output.get("status", "disabled"))
    reader_message = reader_output.get("message")
    reader_unavailable_reason = reader_output.get("reader_unavailable_reason")
    if not reader_enabled and not reader_unavailable_reason:
        reader_unavailable_reason = "no_key"

    # 3. Existing human decisions
    human_decisions = get_existing_human_decisions(prompt_id, prompt)

    atom_verdicts: dict[str, dict[str, Any]] = {}
    accepted_atoms: list[str] = []
    review_atoms: list[str] = []
    open_items: list[dict[str, Any]] = []
    highlight_spans: list[dict[str, Any]] = []

    # Process Atoms
    for atom in ATOM_NAMES:
        rule_has = atom in rule_atoms
        h_dec = human_decisions.get(("atom", atom))

        if h_dec is not None:
            # Prioritize human review decisions
            status = h_dec["status"]
            is_confirmed = status == "confirmed"
            verdict = "human_confirmed" if is_confirmed else "human_rejected"
            atom_verdicts[atom] = {
                "verdict": verdict,
                "accepted": is_confirmed,
                "needs_review": False,
                "quote": "",
                "rule_opinion": rule_has,
                "reader_opinion": None,
                "human_override": True,
                "decided_by": h_dec["decided_by"],
                "decided_at": h_dec["decided_at"],
            }
            if is_confirmed:
                accepted_atoms.append(atom)
            continue

        if not reader_enabled or reader_status == "reader_failed":
            # Running on rules only
            verdict = "rules_only" if rule_has else "absent"
            accepted = rule_has
            atom_verdicts[atom] = {
                "verdict": verdict,
                "accepted": accepted,
                "needs_review": False,
                "quote": "",
                "rule_opinion": rule_has,
                "reader_opinion": None,
                "human_override": False,
            }
            if accepted:
                accepted_atoms.append(atom)
            continue

        # Reader is active and ran
        r_item = reader_output.get("atoms", {}).get(atom, {})
        r_asserted = bool(r_item.get("asserted", False))
        r_unsupported = bool(r_item.get("unsupported", False))
        quote = str(r_item.get("evidence", "")).strip()
        span = r_item.get("span")

        if r_unsupported:
            # Model asserted without valid verbatim quote: discard unsupported claim
            if rule_has:
                verdict = "rules_only"
                accepted = False
                needs_review = True
                review_atoms.append(atom)
                open_items.append(
                    {
                        "item_type": "atom",
                        "item_name": atom,
                        "verdict": verdict,
                        "quote": quote,
                        "rule_opinion": "present",
                        "reader_opinion": "unsupported_quote",
                    }
                )
            else:
                verdict = "unsupported"
                accepted = False
                needs_review = False
        elif r_asserted and rule_has:
            # Agree: both assert and quote passed verbatim check
            verdict = "agree"
            accepted = True
            needs_review = False
            accepted_atoms.append(atom)
            if span:
                highlight_spans.append(
                    {
                        "item_type": "atom",
                        "item_name": atom,
                        "quote": quote,
                        "start": span[0],
                        "end": span[1],
                    }
                )
        elif r_asserted and not rule_has:
            # Reader only: reader asserts with valid quote, rules do not
            verdict = "reader_only"
            accepted = False
            needs_review = True
            review_atoms.append(atom)
            open_items.append(
                {
                    "item_type": "atom",
                    "item_name": atom,
                    "verdict": verdict,
                    "quote": quote,
                    "rule_opinion": "absent",
                    "reader_opinion": "present",
                }
            )
        elif rule_has and not r_asserted:
            # Rules only: rules assert, reader does not
            verdict = "rules_only"
            accepted = False
            needs_review = True
            review_atoms.append(atom)
            open_items.append(
                {
                    "item_type": "atom",
                    "item_name": atom,
                    "verdict": verdict,
                    "quote": quote,
                    "rule_opinion": "present",
                    "reader_opinion": "absent",
                }
            )
        else:
            # Agree absent
            verdict = "agree"
            accepted = False
            needs_review = False

        atom_verdicts[atom] = {
            "verdict": verdict,
            "accepted": accepted,
            "needs_review": needs_review,
            "quote": quote,
            "rule_opinion": rule_has,
            "reader_opinion": r_asserted,
            "span": span,
            "human_override": False,
        }

    # Process Slots
    slot_verdicts: dict[str, dict[str, Any]] = {}
    accepted_slots: dict[str, float] = {}
    review_slots: list[str] = []

    for slot in SLOT_NAMES:
        rule_score = float(rule_slots.get(slot, 0.0))
        h_dec = human_decisions.get(("slot", slot))

        if h_dec is not None:
            status = h_dec["status"]
            val_str = h_dec["decision_value"]
            dec_score = float(val_str) if val_str is not None else (1.0 if status == "confirmed" else 0.0)
            verdict = "human_confirmed" if status == "confirmed" else "human_rejected"
            slot_verdicts[slot] = {
                "verdict": verdict,
                "score": dec_score,
                "needs_review": False,
                "quote": "",
                "rule_score": rule_score,
                "reader_score": None,
                "human_override": True,
                "decided_by": h_dec["decided_by"],
                "decided_at": h_dec["decided_at"],
            }
            accepted_slots[slot] = dec_score
            continue

        if not reader_enabled or reader_status == "reader_failed":
            verdict = "rules_only"
            accepted_slots[slot] = rule_score
            slot_verdicts[slot] = {
                "verdict": verdict,
                "score": rule_score,
                "needs_review": False,
                "quote": "",
                "rule_score": rule_score,
                "reader_score": None,
                "human_override": False,
            }
            continue

        r_item = reader_output.get("slots", {}).get(slot, {})
        r_score = float(r_item.get("score", 0.0))
        r_unsupported = bool(r_item.get("unsupported", False))
        quote = str(r_item.get("evidence", "")).strip()
        span = r_item.get("span")

        if r_unsupported:
            r_score = 0.0

        if rule_score == r_score:
            verdict = "agree"
            score = rule_score
            needs_review = False
            accepted_slots[slot] = score
            if score > 0 and span:
                highlight_spans.append(
                    {
                        "item_type": "slot",
                        "item_name": slot,
                        "score": score,
                        "quote": quote,
                        "start": span[0],
                        "end": span[1],
                    }
                )
        elif r_score > 0 and rule_score == 0.0:
            verdict = "reader_only"
            score = 0.0
            needs_review = True
            review_slots.append(slot)
            accepted_slots[slot] = 0.0
            open_items.append(
                {
                    "item_type": "slot",
                    "item_name": slot,
                    "verdict": verdict,
                    "quote": quote,
                    "rule_opinion": 0.0,
                    "reader_opinion": r_score,
                }
            )
        elif rule_score > 0.0 and r_score == 0.0:
            verdict = "rules_only"
            score = 0.0
            needs_review = True
            review_slots.append(slot)
            accepted_slots[slot] = 0.0
            open_items.append(
                {
                    "item_type": "slot",
                    "item_name": slot,
                    "verdict": verdict,
                    "quote": quote,
                    "rule_opinion": rule_score,
                    "reader_opinion": 0.0,
                }
            )
        else:
            # Differing positive slot scores route to review with neutral score
            verdict = "disagree_value"
            score = 0.0
            needs_review = True
            review_slots.append(slot)
            accepted_slots[slot] = 0.0
            open_items.append(
                {
                    "item_type": "slot",
                    "item_name": slot,
                    "verdict": verdict,
                    "quote": quote,
                    "rule_opinion": rule_score,
                    "reader_opinion": r_score,
                }
            )

        slot_verdicts[slot] = {
            "verdict": verdict,
            "score": score,
            "needs_review": needs_review,
            "quote": quote,
            "rule_score": rule_score,
            "reader_score": r_score,
            "span": span,
            "human_override": False,
        }

    # Energy and Energy Range computation
    if review_atoms:
        e_excl = evaluate(prompt, atoms=accepted_atoms, policy_atoms=policy_atoms).energy
        all_atoms = sorted(set(accepted_atoms) | set(review_atoms))
        e_incl = evaluate(prompt, atoms=all_atoms, policy_atoms=policy_atoms).energy
        energy_min = min(e_excl, e_incl)
        energy_max = max(e_excl, e_incl)
        energy_range = [energy_min, energy_max]
        energy = e_excl
        has_review_range = True
        review_reason = f"Atoms pending human review: {', '.join(review_atoms)}"
    else:
        e = evaluate(prompt, atoms=accepted_atoms, policy_atoms=policy_atoms).energy
        energy_min = energy_max = energy = e
        energy_range = [e, e]
        has_review_range = False
        review_reason = None

    completeness = round(sum(accepted_slots.values()) / float(len(SLOT_NAMES)), 3)

    # Check unclassified clauses
    unclassified_clauses = find_unclassified_clauses(prompt)
    for clause in unclassified_clauses:
        open_items.append(
            {
                "item_type": "clause",
                "item_name": "unclassified_text",
                "verdict": "unclassified",
                "quote": clause,
                "rule_opinion": "unmatched",
                "reader_opinion": None,
            }
        )

    # Unrecognised cue identification
    cue_flags, open_cue_items = find_unrecognised_cue_flags(prompt, accepted_atoms, highlight_spans, human_decisions)
    open_items.extend(open_cue_items)

    # Queue open items in database
    if record_reviews and open_items:
        queue_open_items_in_db(prompt_id, prompt, open_items)

    verified_ready = (
        len(review_atoms) == 0
        and len(review_slots) == 0
        and len(unclassified_clauses) == 0
        and len(open_cue_items) == 0
    )

    energy_lower_bound = energy
    energy_is_lower_bound = True
    if has_review_range:
        energy_display = f"[{energy_min}, {energy_max}] conflicts (at least {energy_min} conflicts found)"
    else:
        if energy == 0:
            energy_display = "0 conflicts found among recognised requirements; unrecognised requirements may add more"
        else:
            energy_display = f"at least {energy} conflicts found"

    return CombinedExtractionResult(
        prompt=prompt,
        prompt_id=prompt_id,
        reader_enabled=reader_enabled,
        reader_status=reader_status,
        reader_message=reader_message,
        accepted_atoms=sorted(accepted_atoms),
        review_atoms=sorted(review_atoms),
        atom_verdicts=atom_verdicts,
        accepted_slots=accepted_slots,
        review_slots=sorted(review_slots),
        slot_verdicts=slot_verdicts,
        energy=energy,
        energy_range=energy_range,
        energy_min=energy_min,
        energy_max=energy_max,
        has_review_range=has_review_range,
        review_reason=review_reason,
        completeness=completeness,
        highlight_spans=highlight_spans,
        open_items=open_items,
        unclassified_clauses=unclassified_clauses,
        verified_ready=verified_ready,
        energy_lower_bound=energy_lower_bound,
        energy_is_lower_bound=energy_is_lower_bound,
        energy_display=energy_display,
        cue_flags=cue_flags,
        reader_unavailable_reason=reader_unavailable_reason,
    )
