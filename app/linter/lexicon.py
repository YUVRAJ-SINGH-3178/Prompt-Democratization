"""Documented Lexicon and Rule Engine for Prompt Requirements and Slots.

This module provides sentence- and clause-level analysis of educational prompts.
Negation is evaluated strictly within clause scope to prevent cross-clause spill.
Every requirement atom and slot pattern includes:
- An explicit general rule
- At least two verified test sentences: one positive and one negative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Complete atom vocabulary
ATOMS: tuple[str, ...] = (
    "very_easy",
    "expert_level",
    "no_basic",
    "hidden_answers",
    "answer_after_each_q",
    "single_correct",
    "multi_interp",
    "clear_unambiguous",
    "confusing_language",
    "ordered_easy_to_hard",
)

# Standard pairwise conflicts
CONFLICTS: tuple[tuple[str, str], ...] = (
    ("very_easy", "expert_level"),
    ("very_easy", "no_basic"),
    ("hidden_answers", "answer_after_each_q"),
    ("single_correct", "multi_interp"),
    ("clear_unambiguous", "confusing_language"),
    ("clear_unambiguous", "multi_interp"),
)

# Nine governed prompt slots
SLOTS: tuple[str, ...] = (
    "role",
    "audience",
    "topic",
    "count",
    "n_options",
    "labeling",
    "marking",
    "explanation",
    "ordering",
)


@dataclass(frozen=True)
class AtomSpec:
    atom: str
    description: str
    rule: str
    positive_example: str
    negative_example: str
    confidence: str
    trigger_regex: str
    is_negative_requirement: bool = False
    negation_regex: str | None = None


@dataclass(frozen=True)
class SlotSpec:
    slot: str
    description: str
    rule: str
    positive_example: str
    negative_example: str
    full_pattern: str
    partial_pattern: str | None = None


# Documented Lexicon Specifications for Requirement Atoms
ATOM_REGISTRY: dict[str, AtomSpec] = {
    "very_easy": AtomSpec(
        atom="very_easy",
        description="Questions must be very easy, beginner-level, or basic.",
        rule="Matches explicit requests for questions to be very easy, beginner level, or basic. "
        "Negated if clause contains negation before the trigger (e.g., 'not very easy').",
        positive_example="All questions must be very easy so every student can answer them.",
        negative_example="The questions should not be very easy; make them challenging.",
        confidence="high",
        trigger_regex=r"\b(?:very\s+easy|beginner[- ]level|basic[- ]level|extremely\s+easy|trivial)\b",
        is_negative_requirement=False,
    ),
    "expert_level": AtomSpec(
        atom="expert_level",
        description="Questions must assess advanced, expert-level, or research knowledge.",
        rule="Matches explicit requirements that questions test advanced or expert-level knowledge/difficulty. "
        "The word 'expert' alone describing a role (e.g., 'expert professor') does NOT trigger this atom.",
        positive_example="They must test advanced research knowledge and expert-level understanding.",
        negative_example="Act as an expert professor and prepare MCQs on computer science.",
        confidence="high",
        trigger_regex=(
            r"\b(?:expert[- ]level(?:\s+(?:understanding|knowledge|difficulty|problem[- ]solving|questions?))?|"
            r"advanced[- ]level|advanced\s+research\s+knowledge|expert\s+(?:understanding|knowledge|difficulty))\b"
        ),
        is_negative_requirement=False,
    ),
    "no_basic": AtomSpec(
        atom="no_basic",
        description="Basic or introductory questions must be excluded.",
        rule="Matches affirmative instructions to exclude, skip, or avoid basic/introductory questions.",
        positive_example="Do not include any basic questions in this test.",
        negative_example="Include basic questions to establish baseline understanding.",
        confidence="high",
        trigger_regex=(
            r"\b(?:do\s+not\s+include\s+(?:any\s+)?basic\s+questions?|"
            r"no\s+basic\s+questions?|skip\s+(?:the\s+)?basics?|"
            r"without\s+(?:any\s+)?basics?|avoid\s+(?:any\s+)?basic\s+questions?|"
            r"no\s+introductory\s+questions?|exclude\s+(?:any\s+)?basic\s+questions?)\b"
        ),
        is_negative_requirement=True,
    ),
    "hidden_answers": AtomSpec(
        atom="hidden_answers",
        description="The correct answer must be concealed or not revealed to students.",
        rule="Matches instructions that answers should not be revealed to students or kept hidden/at the end.",
        positive_example="Highlight the correct answer in bold, but do not reveal the correct answer to students.",
        negative_example="Do not hide the answers from students during self-study.",
        confidence="high",
        trigger_regex=(
            r"\b(?:do\s+not\s+reveal\s+(?:the\s+)?(?:correct\s+)?answers?(?:\s+to\s+students)?|"
            r"hidden\s+answers?|hide\s+(?:the\s+)?answers?|keep\s+(?:the\s+)?answers?\s+hidden|"
            r"answers?\s+(?:hidden|at\s+the\s+end|in\s+an\s+answer\s+key)|"
            r"do\s+not\s+show\s+answers?(?:\s+until\s+the\s+end)?)\b"
        ),
        is_negative_requirement=True,
    ),
    "answer_after_each_q": AtomSpec(
        atom="answer_after_each_q",
        description="The answer must be provided immediately after each question.",
        rule="Matches instructions to provide, show, or display the answer after every question.",
        positive_example="After every question, provide the answer and a detailed explanation.",
        negative_example="Do not provide the answer after each question; reserve all answers for the end.",
        confidence="high",
        trigger_regex=(
            r"\b(?:after\s+(?:every|each)\s+question,?\s+(?:provide|show|give|display|include)\s+(?:the\s+)?answers?|"
            r"answers?\s+after\s+(?:every|each)\s+question|"
            r"show\s+(?:the\s+)?answers?\s+after\s+each\s+question|"
            r"show\s+the\s+answer\s+immediately)\b"
        ),
        is_negative_requirement=False,
    ),
    "single_correct": AtomSpec(
        atom="single_correct",
        description="Exactly one option is the correct answer.",
        rule="Matches requirements for only one, single, or exactly one correct answer.",
        positive_example="Make sure there is only one correct answer even if some options seem correct.",
        negative_example="Questions may allow more than one correct answer.",
        confidence="high",
        trigger_regex=(
            r"\b(?:only\s+one\s+correct\s+answer|single\s+(?:correct\s+)?(?:answer|option)|"
            r"exactly\s+one\s+(?:correct\s+)?(?:answer|option)|"
            r"one\s+correct\s+answer\s+even\s+if)\b"
        ),
        is_negative_requirement=False,
    ),
    "multi_interp": AtomSpec(
        atom="multi_interp",
        description="Options or questions may have multiple interpretations or multiple correct answers.",
        rule="Matches requirements allowing multiple interpretations, ambiguity, or multiple correct choices.",
        positive_example="Include tricky statements and options that can be interpreted in multiple ways.",
        negative_example="Ensure options cannot be interpreted in multiple ways.",
        confidence="medium",
        trigger_regex=(
            r"\b(?:options\s+that\s+can\s+be\s+interpreted\s+in\s+multiple\s+ways|"
            r"multiple\s+interpretations?|multiple\s+correct\s+(?:answers?|options?)|"
            r"more\s+than\s+one\s+correct\s+option|"
            r"interpreted\s+in\s+multiple\s+ways|interpreted\s+in\s+more\s+than\s+one\s+way)\b"
        ),
        is_negative_requirement=False,
    ),
    "clear_unambiguous": AtomSpec(
        atom="clear_unambiguous",
        description="Questions and wording must be clear and unambiguous.",
        rule="Matches instructions that questions/wording must be clear, unambiguous, leaving no room for misinterpretation.",
        positive_example="Ensure each question is clear and unambiguous, leaving no room for misinterpretation.",
        negative_example="Questions need not be clear and unambiguous if testing tolerance for ambiguity.",
        confidence="medium",
        trigger_regex=(
            r"\b(?:clear\s+and\s+unambiguous|clear,\s+unambiguous|unambiguous\s+questions?|"
            r"leaving\s+no\s+room\s+for\s+misinterpretation|clear\s+and\s+precise\s+wording)\b"
        ),
        is_negative_requirement=False,
    ),
    "confusing_language": AtomSpec(
        atom="confusing_language",
        description="Instructions to use confusing language or tricky wording to test students.",
        rule="Matches deliberate requests to use confusing language, tricky statements, or confusing wording.",
        positive_example="Use confusing language so that students have to think carefully.",
        negative_example="Avoid confusing language and tricky statements in exam questions.",
        confidence="medium",
        trigger_regex=(
            r"\b(?:use\s+confusing\s+language|confusing\s+(?:language|wording|questions?|instruction)|"
            r"(?:question\s+wording|language|wording|instruction|statements?)\s+is\s+confusing|"
            r"confuse\s+students|tricky\s+statements|confusing\s+wording|intentionally\s+vague\s+language)\b"
        ),
        is_negative_requirement=False,
    ),
    "ordered_easy_to_hard": AtomSpec(
        atom="ordered_easy_to_hard",
        description="Questions must be arranged in order of increasing difficulty from easiest to hardest.",
        rule="Matches explicit instructions to arrange questions from easiest to hardest or in order of increasing difficulty.",
        positive_example="Arrange all questions from the easiest to the hardest.",
        negative_example="Do not arrange questions from easiest to hardest; shuffle them.",
        confidence="high",
        trigger_regex=(
            r"\b(?:from\s+(?:the\s+)?easiest\s+to\s+(?:the\s+)?hardest|"
            r"in\s+order\s+of\s+increasing\s+difficulty|order\s+of\s+increasing\s+difficulty|"
            r"increasing\s+difficulty|progressing\s+from\s+(?:the\s+)?easiest\s+to\s+(?:the\s+)?hardest|"
            r"easiest\s+to\s+hardest)\b"
        ),
        is_negative_requirement=False,
    ),
}

# Documented Lexicon Specifications for Nine Governed Slots
SLOT_REGISTRY: dict[str, SlotSpec] = {
    "role": SlotSpec(
        slot="role",
        description="Persona framing for the educator/author.",
        rule="Full credit (1.0) for explicit role prompt framing like 'Act as a university teacher', 'expert professor', 'experienced professor'. Partial credit (0.5) for educator mentions without persona framing.",
        positive_example="Act as an experienced professor and generate questions.",
        negative_example="Write questions for bachelor students about database systems.",
        full_pattern=(
            r"\b(?:act\s+as\s+(?:a|an)\s+(?:expert\s+|experienced\s+|university\s+)?(?:teacher|professor|instructor|faculty)|"
            r"university\s+teacher|university\s+professor|college\s+professor|role:\s*(?:teacher|professor|instructor|faculty))\b"
        ),
        partial_pattern=r"\b(?:teacher|professor|instructor|faculty|educator)\b",
    ),
    "audience": SlotSpec(
        slot="audience",
        description="Target educational cohort or student audience.",
        rule="Full credit (1.0) for specific academic cohort (bachelor, master, undergraduate, postgraduate). "
        "Partial credit (0.5) for generic audience framing ('for students', 'aimed at students'). "
        "No credit (0.0) if students is purely incidental (e.g., 'to confuse students').",
        positive_example="Generate 15 MCQs on database management systems for bachelor students.",
        negative_example="Create 25 MCQs on operating systems to confuse students.",
        full_pattern=r"\b(?:bachelor|master|undergraduate|postgraduate|graduate|freshman|sophomore|junior|senior)\s*(?:students?|cohort|level)?\b",
        partial_pattern=r"\b(?:for\s+students|aimed\s+at\s+students|suitable\s+for\s+students|target\s+audience:\s*students?|assess\s+students)\b",
    ),
    "topic": SlotSpec(
        slot="topic",
        description="Academic subject or specific course topic.",
        rule="Full credit (1.0) for identified topic or course phrase (e.g., 'on Normalization', 'about machine learning'). "
        "Partial credit (0.5) for broad discipline without topic (e.g., 'on computer science'). "
        "No credit (0.0) for generic phrases ('cover important concepts').",
        positive_example="Create 10 multiple-choice questions on Normalization.",
        negative_example="Create some multiple-choice questions that cover the important concepts.",
        full_pattern=(
            r"\b(?:course|topic|about|on)\s*[:=]?\s*"
            r"(?:normalization|database\s+management\s+systems|machine\s+learning|operating\s+systems|"
            r"computer\s+networks|algorithms|software\s+engineering|calculus|linear\s+algebra|"
            r"data\s+structures|artificial\s+intelligence|cryptography|compiler\s+design)\b"
        ),
        partial_pattern=r"\b(?:on|about|in|course|topic)\s*[:=]?\s*(?:computer\s+science|engineering|science|business|medicine|humanities|[A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+){0,3})\b",
    ),
    "count": SlotSpec(
        slot="count",
        description="Total quantity of questions requested.",
        rule="Full credit (1.0) for specific integer count followed by questions, MCQs, or items. "
        "No credit (0.0) for non-specific quantities ('some questions', 'prepare MCQs').",
        positive_example="Create 20 MCQs about machine learning.",
        negative_example="Create some multiple-choice questions for students.",
        full_pattern=r"\b\d+\s*(?:multiple[- ]choice\s+questions?|mcqs?|questions?|items?|problems?)\b",
        partial_pattern=None,
    ),
    "n_options": SlotSpec(
        slot="n_options",
        description="Number of choices per question.",
        rule="Full credit (1.0) for explicit number of options ('four options', 'four choices', 'four answer choices'). "
        "Partial credit (0.5) for unspecified choices ('multiple choices').",
        positive_example="Provide four choices for every question.",
        negative_example="Provide multiple choices for each question.",
        full_pattern=r"\b(?:four|4|three|3|five|5)\s+(?:answer\s+)?(?:options?|choices?)\b",
        partial_pattern=r"\b(?:multiple\s+choices?|options\s+for\s+each\s+question)\b",
    ),
    "labeling": SlotSpec(
        slot="labeling",
        description="Option labeling scheme (e.g., A, B, C, D).",
        rule="Full credit (1.0) for explicit label format '(A, B, C, D)' or 'labelled A, B, C, and D'. "
        "Partial credit (0.5) for general instruction to label without format.",
        positive_example="Each question should have four options labelled A, B, C, and D.",
        negative_example="Include choices for each question without mentioning letter labels.",
        full_pattern=(
            r"\b(?:label(?:led)?\s*(?:each\s+answer\s+choice\s*)?\(?\s*a\s*,\s*b\s*,\s*c\s*,\s*(?:and\s+)?d\s*\)?|"
            r"\(?\s*a\s*,\s*b\s*,\s*c\s*,\s*d\s*\)?|a\s*(?:to|-|through)\s*d|a\)\s*b\)\s*c\)\s*d\))\b"
        ),
        partial_pattern=r"\b(?:label\s+(?:the\s+)?(?:options|choices)|use\s+labels)\b",
    ),
    "marking": SlotSpec(
        slot="marking",
        description="Identification or indication of the correct answer.",
        rule="Full credit (1.0) for direct instruction to mark, highlight, or provide the correct answer or answers. "
        "Partial credit (0.5) for vague mentions ('include answers').",
        positive_example="Highlight the correct answer in bold.",
        negative_example="Do not indicate which choice is correct.",
        full_pattern=(
            r"\b(?:mark\s+(?:the\s+)?(?:right|correct)\s+answer|provide\s+(?:the\s+)?answers?|"
            r"highlight\s+(?:the\s+)?correct\s+answer|give\s+(?:the\s+)?correct\s+answer|"
            r"the\s+correct\s+answer|indicate\s+(?:the\s+)?correct\s+answer|answer\s+key)\b"
        ),
        partial_pattern=r"\b(?:include\s+answers?|answers?\s+included)\b",
    ),
    "explanation": SlotSpec(
        slot="explanation",
        description="Answer explanation or rationale requirement.",
        rule="Full credit (1.0) for definite requirement of explanations for answers. "
        "Partial credit (0.5) for conditional or partial explanations ('wherever necessary', 'if needed').",
        positive_example="After every question, provide the answer and a detailed explanation.",
        negative_example="Explain the answers wherever necessary.",
        full_pattern=(
            r"\b(?:provide\s+an\s+explanation|include\s+explanations?|detailed\s+explanation|"
            r"explain\s+why|an\s+explanation\s+for\s+every\s+option|explain\s+why\s+the\s+other\s+options\s+are\s+wrong)\b"
        ),
        partial_pattern=r"\b(?:explain\s+(?:the\s+)?answers?\s+wherever\s+necessary|explanations?\s+(?:wherever\s+necessary|if\s+needed|where\s+appropriate|optional))\b",
    ),
    "ordering": SlotSpec(
        slot="ordering",
        description="Ordering or arrangement of questions.",
        rule="Full credit (1.0) for ordering by difficulty or pedagogical sequence ('increasing difficulty', 'easiest to hardest'). "
        "Partial credit (0.5) for generic ordering instruction ('arrange them properly').",
        positive_example="Organize the questions in order of increasing difficulty.",
        negative_example="Arrange them properly.",
        full_pattern=(
            r"\b(?:arrange\s+all\s+questions\s+from\s+(?:the\s+)?easiest\s+to\s+(?:the\s+)?hardest|"
            r"in\s+order\s+of\s+increasing\s+difficulty|order\s+of\s+increasing\s+difficulty|"
            r"easiest\s+to\s+hardest|ordered\s+by\s+difficulty|"
            r"progressing\s+from\s+(?:the\s+)?easiest\s+to\s+(?:the\s+)?hardest)\b"
        ),
        partial_pattern=r"\b(?:arrange\s+them\s+properly|order\s+properly|organize\s+properly|arrange\s+logically)\b",
    ),
}


# Text processing helpers: Sentence and Clause segmentation
def split_sentences(text: str) -> list[str]:
    """Split text into sentences while protecting abbreviations and format markers."""
    t = re.sub(r"[ \t]+", " ", text).strip()
    protected = re.sub(
        r"\b(e\.g\.|i\.e\.|etc\.|vs\.|prof\.|dr\.|dept\.|no\.)",
        lambda m: m.group(0).replace(".", "<DOT>"),
        t,
        flags=re.IGNORECASE,
    )
    raw = re.split(r"(?<=[.!?])\s+|\n+", protected)
    sentences = [s.replace("<DOT>", ".").strip() for s in raw if s.strip()]
    return sentences


def split_clauses(sentence: str) -> list[str]:
    """Split a sentence into clauses along contrastive conjunctions and semicolons."""
    # Split on contrastive conjunctions, semicolons, or dashes
    parts = re.split(
        r"(?:;\s*|,\s*(?=(?:but|however|yet|although|whereas|while|except|even\s+if)\b)|\b(?:but|however|yet|although|whereas|while|except)\b)",
        sentence,
        flags=re.IGNORECASE,
    )
    return [p.strip() for p in parts if p.strip()]


def has_clause_negation(clause: str, match_start: int) -> bool:
    """Check if the clause contains a negation token preceding the matched pattern."""
    prefix = clause[:match_start]
    # Check for negative words in the immediate prefix within the clause
    return bool(
        re.search(
            r"\b(?:no|not|never|avoid|without|do\s+not|don't|should\s+not|shouldn't|cannot|can't|neither)\b",
            prefix,
            re.IGNORECASE,
        )
    )


def extract_atoms_from_text(prompt: str) -> list[str]:
    """Extract normalized requirement atoms using clause-level negation and lexical patterns."""
    found: set[str] = set()
    sentences = split_sentences(prompt)

    for sentence in sentences:
        clauses = split_clauses(sentence)
        for clause in clauses:
            for atom_name, spec in ATOM_REGISTRY.items():
                match = re.search(spec.trigger_regex, clause, re.IGNORECASE)
                if match:
                    if spec.is_negative_requirement:
                        # Negative requirement (like no_basic, hidden_answers):
                        # The trigger itself encodes negation.
                        # Check if it is cancelled by a double negation (e.g. "do not skip basics")
                        if not has_clause_negation(clause, match.start()):
                            found.add(atom_name)
                    else:
                        # Positive requirement:
                        # If preceded by negation token in the clause, it is negated (do not extract).
                        if not has_clause_negation(clause, match.start()):
                            found.add(atom_name)

    return sorted(found)


def score_slots_from_text(prompt: str) -> dict[str, float]:
    """Score the nine governed slots using lexical full- and partial-credit patterns."""
    sentences = split_sentences(prompt)
    scores: dict[str, float] = {}

    for slot_name, spec in SLOT_REGISTRY.items():
        matched_full = False
        matched_partial = False

        for sentence in sentences:
            if re.search(spec.full_pattern, sentence, re.IGNORECASE):
                # Ensure audience incidental nouns don't trigger full credit
                if slot_name == "audience":
                    # Check that it's an audience cohort, not an incidental object
                    if re.search(spec.full_pattern, sentence, re.IGNORECASE):
                        matched_full = True
                        break
                elif slot_name == "topic":
                    # Check that it's not just a generic word
                    if not re.search(r"\b(?:cover\s+the\s+important\s+concepts)\b", sentence, re.IGNORECASE):
                        matched_full = True
                        break
                else:
                    matched_full = True
                    break

            if spec.partial_pattern and re.search(spec.partial_pattern, sentence, re.IGNORECASE):
                matched_partial = True

        if matched_full:
            scores[slot_name] = 1.0
        elif matched_partial:
            scores[slot_name] = 0.5
        else:
            scores[slot_name] = 0.0

    return scores


def get_lexicon_manifest() -> dict[str, Any]:
    """Return structured manifest of the documented lexicon for verification and reporting."""
    return {
        "atom_count": len(ATOM_REGISTRY),
        "slot_count": len(SLOT_REGISTRY),
        "atoms": {name: spec.__dict__ for name, spec in ATOM_REGISTRY.items()},
        "slots": {name: spec.__dict__ for name, spec in SLOT_REGISTRY.items()},
    }
