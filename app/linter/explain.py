from __future__ import annotations

import re

ATOM_PATTERNS = {
    "very_easy": r"very easy|beginner level|basic level",
    "expert_level": r"expert level|advanced level|expert",
    "no_basic": r"no basic|without basics|skip the basics|no basic questions",
    "hidden_answers": r"hidden answers?|hide answers?|answers? (?:hidden|at the end|in an answer key)",
    "answer_after_each_q": r"answer after each question|answers? after every question|show(?: the)? answers? after each question|show the answer immediately",
    "single_correct": r"exactly one (?:correct )?(?:answer|option)|single (?:correct )?(?:answer|option)|one correct answer",
    "multi_interp": r"multiple interpretations?|multiple correct|more than one correct|ambiguous",
    "clear_unambiguous": r"clear|unambiguous|precise",
    "confusing_language": r"confusing|unclear|vague|ambiguous",
}
SLOT_PATTERNS = {
    "role": r"\b(?:teacher|instructor|professor|faculty)\b",
    "audience": r"\b(?:bachelor|master|undergraduate|graduate|students?)\b",
    "topic": r"\b(?:topic|course|about|on)\s*[:=]?\s*[#\w][^,.;\n]{2,}",
    "count": r"\b\d+\s+(?:questions?|items?)\b",
    "n_options": r"\b(?:\d+\s+(?:options?|choices?)|four options?)\b",
    "labeling": r"\b(?:a\s*(?:,|to|-|through)\s*d|a\)\s*b\)\s*c\)\s*d\))",
    "marking": r"\b(?:correct answer|mark(?:ed|ing)? as correct|answer key)\b",
    "explanation": r"\b(?:explanation|explain why|rationale)\b",
    "ordering": r"\b(?:order(?:ed)?|progress(?:ively)?|difficulty|bloom)\b",
}


def evidence(prompt: str, atoms: list[str], slot_scores: dict[str, float]) -> dict:
    spans = []
    atom_notes = []
    for atom in atoms:
        pattern = ATOM_PATTERNS.get(atom)
        matches = list(re.finditer(pattern, prompt, re.IGNORECASE)) if pattern else []
        for match in matches:
            spans.append({"start": match.start(), "end": match.end(), "atom": atom})
        confidence = "medium" if atom in ("multi_interp", "clear_unambiguous", "confusing_language") else "high"
        atom_notes.append(
            {
                "atom": atom,
                "confidence": confidence,
                "manual_review": confidence != "high",
                "evidence": matches[0].group(0) if matches else None,
            }
        )
    sentences = re.split(r"(?<=[.!?])\s+|\n+", prompt)
    slots = []
    cursor = 0
    for sentence in sentences:
        for slot, pattern in SLOT_PATTERNS.items():
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                slots.append((slot, cursor + match.start(), cursor + match.end(), sentence.strip()))
        cursor += len(sentence) + 1
    by_slot = {slot: (start, end, sentence) for slot, start, end, sentence in slots}
    slot_details = []
    for slot, score in slot_scores.items():
        found = by_slot.get(slot)
        slot_details.append(
            {
                "slot": slot,
                "score": score,
                "evidence": found[2] if found else None,
                "gap": None if found else f"Add a clear {slot.replace('_', ' ')} requirement.",
            }
        )
    spans.sort(key=lambda row: (row["start"], row["end"], row["atom"]))
    return {
        "spans": spans,
        "atoms": atom_notes,
        "slots": slot_details,
        "manual_review": any(row["manual_review"] for row in atom_notes),
    }


def highlight_html(prompt: str, spans: list[dict]) -> str:
    from html import escape

    ordered = sorted(spans, key=lambda row: (row["start"], -row["end"]))
    out = []
    cursor = 0
    for row in ordered:
        start, end = row["start"], row["end"]
        if start < cursor or end > len(prompt):
            continue
        out.append(escape(prompt[cursor:start]))
        out.append(f'<mark data-atom="{escape(row["atom"])}">{escape(prompt[start:end])}</mark>')
        cursor = end
    out.append(escape(prompt[cursor:]))
    return "".join(out)
