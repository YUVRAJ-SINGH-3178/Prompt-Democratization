from __future__ import annotations

import re

from app.linter.lexicon import ATOM_REGISTRY, SLOT_REGISTRY, split_sentences


def evidence(prompt: str, atoms: list[str], slot_scores: dict[str, float]) -> dict:
    spans = []
    atom_notes = []
    for atom in atoms:
        spec = ATOM_REGISTRY.get(atom)
        pattern = spec.trigger_regex if spec else None
        matches = list(re.finditer(pattern, prompt, re.IGNORECASE)) if pattern else []
        for match in matches:
            spans.append({"start": match.start(), "end": match.end(), "atom": atom})
        confidence = spec.confidence if spec else "medium"
        atom_notes.append(
            {
                "atom": atom,
                "confidence": confidence,
                "manual_review": confidence != "high",
                "evidence": matches[0].group(0) if matches else None,
            }
        )
    sentences = split_sentences(prompt)
    slots = []
    for sentence in sentences:
        cursor = prompt.find(sentence)
        if cursor == -1:
            cursor = 0
        for slot, slot_spec in SLOT_REGISTRY.items():
            match = re.search(slot_spec.full_pattern, sentence, re.IGNORECASE)
            if not match and slot_spec.partial_pattern:
                match = re.search(slot_spec.partial_pattern, sentence, re.IGNORECASE)
            if match:
                slots.append((slot, cursor + match.start(), cursor + match.end(), sentence.strip()))
    by_slot = {slot: (start, end, sentence) for slot, start, end, sentence in slots}
    slot_details = []
    for slot, score in slot_scores.items():
        found = by_slot.get(slot)
        slot_details.append(
            {
                "slot": slot,
                "score": score,
                "evidence": found[2] if found else None,
                "gap": None if score == 1.0 else f"Add a clear {slot.replace('_', ' ')} requirement.",
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
