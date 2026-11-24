"""Documented Lexicon and Rule Engine for Prompt Requirements and Slots.

This module provides sentence- and clause-level analysis of educational prompts.
Negation is evaluated strictly within clause scope to prevent cross-clause spill.
Every requirement atom and slot pattern includes:
- An explicit general rule
- Matched phrase extraction with character spans
- Synonym lists and number word recognition
- Detection of unclassified instructional clauses
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.linter.rubrics import SLOT_RUBRICS

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
class MatchedAtom:
    atom: str
    matched_phrase: str
    start: int
    end: int
    confidence: str


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
        rule="Matches explicit requests for questions to be very easy, beginner level, trivial, or simple. "
        "Negated if clause contains negation before the trigger (e.g., 'not very easy').",
        positive_example="All questions must be very easy so every student can answer them.",
        negative_example="The questions should not be very easy; make them challenging.",
        confidence="high",
        trigger_regex=(
            r"\b(?:very\s+easy|extremely\s+easy|beginner[- ]level|basic[- ]level|elementary[- ]level|"
            r"introductory[- ]level|trivial|simple\s+questions?|easy\s+level|easy\s+questions?|"
            r"easy\s+so\s+every\s+student\s+can\s+answer|all\s+questions\s+must\s+be\s+easy|"
            r"questions\s+must\s+be\s+(?:very\s+)?easy|make\s+all\s+questions?\s+(?:very\s+)?easy|"
            r"easiest\s+level|all\s+very\s+easy)\b"
        ),
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
            r"expert\s+level|advanced[- ]level|advanced\s+research\s+knowledge|expert\s+(?:understanding|knowledge|difficulty)|"
            r"deep\s+critical\s+thinking|complex\s+problem[- ]solving|research[- ]level|"
            r"phd[- ]level|graduate[- ]level\s+understanding|master[- ]level\s+understanding)\b"
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
            r"\b(?:do\s+not\s+(?:include|put)\s+(?:any\s+)?basic\s+questions?|"
            r"no\s+basic\s+questions?(?:\s+allowed|\s+in\s+this)?|skip\s+(?:the\s+)?basics?|"
            r"skip\s+(?:the\s+)?basics?\s+completely|skip\s+introductory\s+concepts|without\s+(?:any\s+)?basics?|"
            r"avoid\s+(?:any\s+)?basic\s+questions?|no\s+introductory\s+questions?|"
            r"exclude\s+(?:any\s+)?basic\s+questions?|omit\s+(?:any\s+)?basic\s+questions?|"
            r"no\s+simple\s+questions?|don't\s+include\s+(?:any\s+)?basic\s+questions?)\b"
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
            r"don't\s+reveal\s+(?:the\s+)?(?:correct\s+)?answers?(?:\s+to\s+students)?|hidden\s+answers?|"
            r"hide\s+(?:the\s+)?answers?(?:\s+from\s+students)?|"
            r"keep\s+(?:the\s+)?(?:correct\s+)?answers?\s+hidden(?:\s+from\s+students)?|"
            r"(?:correct\s+)?answers?\s+keep\s+hidden(?:\s+from\s+students)?|"
            r"answers?\s+(?:hidden|at\s+the\s+end|in\s+an\s+answer\s+key)|"
            r"do\s+not\s+show\s+answers?(?:\s+to\s+students)?(?:\s+until\s+(?:the\s+end|after\s+grading))?|"
            r"never\s+show\s+(?:the\s+)?answers\s+to\s+students|withhold\s+(?:the\s+)?answers?(?:\s+from\s+students)?|"
            r"students?\s+must\s+not\s+see\s+(?:the\s+)?answers?|answers?\s+should\s+not\s+be\s+visible|"
            r"do\s+not\s+provide\s+an\s+answer\s+key\s+to\s+students)\b"
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
            r"\b(?:after\s+(?:every|each)\s+question,?\s+(?:provide|show|give|display|include|state|write)\s+(?:the\s+)?answers?|"
            r"answers?\s+after\s+(?:every|each)\s+question|"
            r"show\s+(?:the\s+)?answers?\s+after\s+each\s+question|"
            r"show\s+the\s+answer\s+immediately|"
            r"state\s+the\s+answer\s+after\s+each\s+question|"
            r"display\s+the\s+answer\s+after\s+each\s+question|"
            r"answer\s+and\s+explanation\s+after\s+each\s+question|"
            r"answer\s+after\s+each\s+question|"
            r"give\s+the\s+answer\s+right\s+after\s+the\s+question|"
            r"put\s+the\s+answer\s+below\s+each\s+question|"
            r"immediately\s+follow\s+each\s+question\s+with\s+the\s+answer)\b"
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
            r"\b(?:only\s+one\s+correct\s+answer|single\s+(?:correct\s+)?(?:answer|option|choice)|"
            r"strictly\s+only\s+one\s+correct\s+answer|strictly\s+one\s+(?:correct\s+)?(?:answer|option|choice|right\s+answer)|"
            r"strictly\s+one\s+right\s+answer|exactly\s+one\s+(?:correct\s+)?(?:answer|option|choice)|"
            r"one\s+correct\s+answer\s+even\s+if|just\s+one\s+option\s+is\s+correct|"
            r"one\s+right\s+answer|sole\s+correct\s+answer|only\s+one\s+option\s+can\s+be\s+correct|"
            r"no\s+second\s+right\s+answer|one\s+correct\s+answer\s+for\s+each)\b"
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
            r"multiple\s+interpretations?|multiple\s+correct\s+(?:answers?|options?|choices?)|"
            r"more\s+than\s+one\s+correct\s+option|"
            r"interpreted\s+in\s+multiple\s+ways|interpreted\s+in\s+more\s+than\s+one\s+way|"
            r"ambiguous\s+options?|choices?\s+can\s+be\s+interpreted\s+in\s+different\s+ways|"
            r"several\s+choices\s+seem\s+plausible)\b"
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
            r"leaving\s+no\s+room\s+for\s+misinterpretation|clear\s+and\s+precise\s+wording|"
            r"no\s+ambiguity|clear\s+wording\s+without\s+ambiguity|clear\s+unambiguous|"
            r"strictly\s+defined\s+wording)\b"
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
            r"\b(?:use\s+confusing\s+language|confusing\s+(?:language|wording|questions?|instruction|phrasing)|"
            r"(?:question\s+wording|language|wording|instruction|statements?)\s+is\s+confusing|"
            r"confuse\s+students|tricky\s+statements|tricky\s+language|tricky\s+wording|"
            r"confusing\s+wording|intentionally\s+vague\s+language|deliberately\s+confusing|"
            r"deliberately\s+misleading\s+wording|double\s+negatives)\b"
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
            r"easiest\s+to\s+hardest|easy\s+to\s+hard|order\s+from\s+easy\s+to\s+hard|"
            r"sort\s+from\s+easiest\s+to\s+hardest|ascending\s+order\s+of\s+difficulty|"
            r"ascending\s+difficulty|progress\s+from\s+simplest\s+.*?\s+to\s+most\s+difficult|"
            r"progress\s+from\s+(?:the\s+)?easiest\s+introductory\s+items\s+to\s+(?:the\s+)?hardest|"
            r"ordered\s+by\s+difficulty)\b"
        ),
        is_negative_requirement=False,
    ),
}

# Number words pattern
NUMBER_WORDS = (
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty(?:-five)?|"
    r"twenty\s+five|twenty\s+four|thirty|forty|fifty)"
)

# Documented Lexicon Specifications for Nine Governed Slots
SLOT_REGISTRY: dict[str, SlotSpec] = {
    "role": SlotSpec(
        slot="role",
        description=SLOT_RUBRICS["role"].description,
        rule=SLOT_RUBRICS["role"].score_1_0_rule,
        positive_example=SLOT_RUBRICS["role"].examples["1.0"],
        negative_example=SLOT_RUBRICS["role"].examples["0.0"],
        full_pattern=(
            r"\b(?:act\s+as\s+(?:a|an)?\s*(?:expert|distinguished|experienced|college|university|academic)?\s*(?:teacher|professor|instructor|faculty|lecturer)|"
            r"role:\s*(?:university\s+)?(?:teacher|professor|instructor|faculty|lecturer)|"
            r"persona:\s*(?:university\s+)?(?:teacher|professor|instructor|faculty|lecturer)|"
            r"instructor\s+persona:\s*.*?(?:teacher|professor|instructor|faculty|lecturer)|"
            r"you\s+are\s+to\s+act\s+in\s+the\s+professional\s+capacity\s+of\s+(?:a|an)\s+.*?(?:teacher|professor|instructor|faculty|lecturer)|"
            r"please\s+function\s+as\s+(?:a|an)?\s*(?:expert|experienced)?\s*(?:teacher|professor|instructor|faculty|lecturer)|"
            r"serve\s+as\s+(?:a|an)?\s*(?:university)?\s*(?:teacher|professor|instructor|faculty|lecturer)|"
            r"you\s+are\s+(?:a\s+)?teacher\s+for\s+university|"
            r"you\s+are\s+a\s+teacher\s+designing|"
            r"university\s+teacher|university\s+professor|college\s+professor)\b"
        ),
        partial_pattern=r"\b(?:teacher|professor|instructor|faculty|lecturer|educator)\b",
    ),
    "audience": SlotSpec(
        slot="audience",
        description=SLOT_RUBRICS["audience"].description,
        rule=SLOT_RUBRICS["audience"].score_1_0_rule,
        positive_example=SLOT_RUBRICS["audience"].examples["1.0"],
        negative_example=SLOT_RUBRICS["audience"].examples["0.0"],
        full_pattern=(
            r"\b(?:bachelor|master|undergraduate|postgraduate|graduate|freshman|sophomore|junior|senior|"
            r"undergraduates|postgraduates|graduates)\s*(?:students?|cohort|learners?|class|level)?\b"
        ),
        partial_pattern=(
            r"\b(?:for\s+(?:the\s+)?(?:\w+\s+)?students?|aimed\s+at\s+students?|suitable\s+for\s+students?|"
            r"target\s+audience:\s*students?|audience:\s*students?|assess\s+students?|test\s+students?|"
            r"students\s+taking|from\s+students|to\s+students|for\s+(?:the\s+)?class|"
            r"for\s+(?:the\s+)?learners?|aimed\s+at\s+learners?|help\s+students|"
            r"help\s+them\s+understand|suitable\s+for\s+examination|intended\s+for\s+students)\b"
        ),
    ),
    "topic": SlotSpec(
        slot="topic",
        description=SLOT_RUBRICS["topic"].description,
        rule=SLOT_RUBRICS["topic"].score_1_0_rule,
        positive_example=SLOT_RUBRICS["topic"].examples["1.0"],
        negative_example=SLOT_RUBRICS["topic"].examples["0.0"],
        full_pattern=(
            r"\b(?:course|topic|about|on|covering|focusing\s+on|focusing\s+exclusively\s+on|subject|professor\s+of)\s*[:=]?\s*"
            r"(?:topic\s+is\s+)?(?:normalization|database\s+normalization|database\s+management\s+systems|database\s+systems|machine\s+learning|operating\s+systems|"
            r"computer\s+networks|algorithms|software\s+engineering|calculus|linear\s+algebra|"
            r"data\s+structures|artificial\s+intelligence|cryptography|compiler\s+design|"
            r"cpu\s+scheduling|binary\s+search\s+trees|graph\s+theory|networking|"
            r"distributed\s+transactions|consensus\s+protocols|cryptographic\s+protocols|"
            r"memory\s+management|virtual\s+memory|computational\s+complexity|np[- ]completeness|"
            r"object[- ]oriented\s+programming|recursion|sql\s+indexing|software\s+testing|"
            r"computer\s+architecture|web\s+development|digital\s+circuits|discrete\s+mathematics|"
            r"distributed\s+systems|cloud\s+computing|python\s+programming|computer\s+security|"
            r"software\s+architecture|web\s+security|relational\s+algebra|computer\s+systems|"
            r"sorting\s+algorithms)\b|"
            r"\btopic\s+is\s+[a-z\s]+\b"
        ),
        partial_pattern=r"\b(?:on|about|in|course|topic|subject)\s*[:=]?\s*(?:computer\s+science|engineering|science|business|medicine|humanities|computing)\b",
    ),
    "count": SlotSpec(
        slot="count",
        description=SLOT_RUBRICS["count"].description,
        rule=SLOT_RUBRICS["count"].score_1_0_rule,
        positive_example=SLOT_RUBRICS["count"].examples["1.0"],
        negative_example=SLOT_RUBRICS["count"].examples["0.0"],
        full_pattern=(
            rf"\b{NUMBER_WORDS}\s*(?:multiple[- ]choice\s+questions?|multiple\s+choice|mcqs?|questions?|items?|problems?|test\s+questions?)\b|"
            r"\b(?:count|quantity|items)\s*[:=]\s*\d+\b"
        ),
        partial_pattern=r"\b(?:around|between|at\s+least|approx(?:imately)?)\s*\d+",
    ),
    "n_options": SlotSpec(
        slot="n_options",
        description=SLOT_RUBRICS["n_options"].description,
        rule=SLOT_RUBRICS["n_options"].score_1_0_rule,
        positive_example=SLOT_RUBRICS["n_options"].examples["1.0"],
        negative_example=SLOT_RUBRICS["n_options"].examples["0.0"],
        full_pattern=(
            r"\b(?:four|4|three|3|five|5)\s+(?:clearly\s+distinguished\s+)?(?:answer\s+)?(?:options?|choices?|alternatives?)\b|"
            r"\b(?:options|choices)\s*[:=]\s*(?:four|4|three|3|five|5)\b|"
            r"\b4\s+options\b|\b4\s+choices\b"
        ),
        partial_pattern=r"\b(?:multiple\s+choices?|options\s+for\s+each\s+question|choices\s+for\s+each|choices\s+and\s+answers)\b",
    ),
    "labeling": SlotSpec(
        slot="labeling",
        description=SLOT_RUBRICS["labeling"].description,
        rule=SLOT_RUBRICS["labeling"].score_1_0_rule,
        positive_example=SLOT_RUBRICS["labeling"].examples["1.0"],
        negative_example=SLOT_RUBRICS["labeling"].examples["0.0"],
        full_pattern=(
            r"\b(?:label(?:led)?\s*(?:each\s+answer\s+choice\s*)?\(?\s*a\s*,\s*b\s*,\s*c\s*,\s*(?:and\s+)?d\s*\)?|"
            r"\(?\s*a\s*,\s*b\s*,\s*c\s*,\s*d\s*\)?|a\s*(?:to|-|through)\s*d|a\)\s*b\)\s*c\)\s*d\)|"
            r"\(a-d\)|a\s+b\s+c\s+d\s+letters|letters\s+a(?:-|\s+to\s+)d|labelled\s+a\s+through\s+d|"
            r"four\s+choice\s+a\s+b\s+c\s+d|four\s+choices\s+a\s+b\s+c\s+d)\b"
        ),
        partial_pattern=r"\b(?:label\s+(?:the\s+)?(?:options|choices)|use\s+labels|each\s+option\s+has\s+a\s+clear\s+label)\b",
    ),
    "marking": SlotSpec(
        slot="marking",
        description=SLOT_RUBRICS["marking"].description,
        rule=SLOT_RUBRICS["marking"].score_1_0_rule,
        positive_example=SLOT_RUBRICS["marking"].examples["1.0"],
        negative_example=SLOT_RUBRICS["marking"].examples["0.0"],
        full_pattern=(
            r"\b(?:mark\s+(?:the\s+)?(?:right|correct)\s+(?:choice|answer)|mark\s+the\s+answer|mark\s+correct|mark\s+answer|"
            r"provide\s+(?:the\s+)?answers?|provide\s+(?:four\s+options\s+and\s+)?answers?|"
            r"give\s+(?:the\s+)?answers?|give\s+(?:the\s+)?correct\s+answer|"
            r"highlight\s+(?:the\s+)?(?:correct|right)\s+(?:answer|choice)|supply\s+(?:the\s+)?correct\s+answer|"
            r"the\s+correct\s+answer|correct\s+answer|indicate\s+(?:the\s+)?(?:right|correct)\s+(?:choice|answer)|"
            r"answer\s+key|specify\s+which\s+choice\s+is\s+right|highlight\s+the\s+right\s+choice|"
            r"show\s+(?:the\s+)?answers?|never\s+show\s+(?:the\s+)?answers|answers?\s+after\s+each\s+question|"
            r"only\s+one\s+correct\s+answer|strictly\s+one\s+correct\s+answer|put\s+answers\s+at\s+the\s+end|"
            r"with\s+answers|give\s+answers|option\s+is\s+correct|strictly\s+one\s+right\s+answer|"
            r"(?:only|exactly|strictly)?\s*one\s+correct\s+(?:option|choice|answer)|"
            r"don't\s+reveal\s+answers|do\s+not\s+reveal\s+(?:the\s+)?correct\s+answer|"
            r"answer\s+and\s+explanation|an\s+explanation\s+for\s+the\s+answer)\b"
        ),
        partial_pattern=r"\b(?:include\s+answers?|answers?\s+included|with\s+answers|and\s+answers)\b",
    ),
    "explanation": SlotSpec(
        slot="explanation",
        description=SLOT_RUBRICS["explanation"].description,
        rule=SLOT_RUBRICS["explanation"].score_1_0_rule,
        positive_example=SLOT_RUBRICS["explanation"].examples["1.0"],
        negative_example=SLOT_RUBRICS["explanation"].examples["0.0"],
        full_pattern=(
            r"\b(?:provide\s+(?:an\s+|detailed\s+)?explanations?|include\s+(?:an\s+|detailed\s+)?explanations?|detailed\s+explanations?|"
            r"explain\s+why|explain\s+every\s+answer|explain\s+each\s+answer|explain\s+each\s+option|"
            r"an\s+explanation\s+for\s+every\s+option|explain\s+why\s+the\s+other\s+options\s+are\s+wrong|"
            r"write\s+a\s+full\s+technical\s+explanation|write\s+clear\s+explanation|"
            r"explanation\s+for\s+(?:the\s+)?answers?|give\s+reason|rationale|"
            r"feedback:\s*detailed\s+explanation|provide\s+answers\s+and\s+detailed\s+explanations?|"
            r"provide\s+explanations|and\s+explanations|explain\s+each\s+choice|"
            r"answer\s+and\s+explanation|detailed\s+explanation|"
            r"explain\s+(?:the\s+)?answers(?!\s+wherever|\s+if|\s+where|\s+optional))\b"
        ),
        partial_pattern=(
            r"\b(?:explain\s+(?:the\s+)?answers?\s+wherever\s+necessary|"
            r"explanations?\s+(?:wherever\s+necessary|if\s+needed|where\s+appropriate|optional)|"
            r"provide\s+explanations\s+wherever\s+necessary|"
            r"explain\s+if\s+(?:necessary|needed)|explanations?\s+wherever\s+necessary)\b"
        ),
    ),
    "ordering": SlotSpec(
        slot="ordering",
        description=SLOT_RUBRICS["ordering"].description,
        rule=SLOT_RUBRICS["ordering"].score_1_0_rule,
        positive_example=SLOT_RUBRICS["ordering"].examples["1.0"],
        negative_example=SLOT_RUBRICS["ordering"].examples["0.0"],
        full_pattern=(
            r"\b(?:(?:arrange|organize)\s+all\s+questions\s+from\s+(?:the\s+)?easiest\s+to\s+(?:the\s+)?hardest|"
            r"(?:arrange|organize)\s+(?:from\s+)?(?:the\s+)?easiest\s+to\s+(?:the\s+)?hardest|"
            r"in\s+order\s+of\s+increasing\s+difficulty|order\s+of\s+increasing\s+difficulty|"
            r"increasing\s+difficulty|progressing\s+from\s+(?:the\s+)?easiest\s+to\s+(?:the\s+)?hardest|"
            r"easiest\s+to\s+hardest|easy\s+to\s+hard|order\s+from\s+easy\s+to\s+hard|"
            r"sort\s+from\s+easiest\s+to\s+hardest|ascending\s+order\s+of\s+difficulty|"
            r"ascending\s+difficulty|progress\s+from\s+simplest\s+.*?\s+to\s+most\s+difficult|"
            r"progress\s+from\s+(?:the\s+)?easiest\s+introductory\s+items\s+to\s+(?:the\s+)?hardest|"
            r"ordered\s+by\s+difficulty)\b"
        ),
        partial_pattern=r"\b(?:arrange|order|organize)\s+(?:them\s+)?properly\b|\barrange\s+logically\b",
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
    raw = re.split(r"(?<=[.!?])\s+|\n+|(?<=\n)\s*[-*•]\s*|(?<=\s)[-*•]\s+|(?:\b(?:Rule\s*\d+|\d+\.)\s*)", protected)
    sentences = [s.replace("<DOT>", ".").strip() for s in raw if s.strip()]
    return sentences


def split_clauses(sentence: str) -> list[str]:
    """Split a sentence into clauses along contrastive conjunctions, semicolons, and punctuation."""
    parts = re.split(
        r"(?:;\s*|,\s*(?=(?:but|however|yet|although|whereas|while|except|even\s+if|only|instead|rather)\b)|\b(?:but|however|yet|although|whereas|while|except)\b)",
        sentence,
        flags=re.IGNORECASE,
    )
    return [p.strip() for p in parts if p.strip()]


def has_clause_negation(clause: str, match_start: int) -> bool:
    """Check if the clause contains a negation token preceding the matched pattern."""
    prefix = clause[:match_start]
    return bool(
        re.search(
            r"\b(?:no|not|never|avoid|without|do\s+not|don't|should\s+not|shouldn't|cannot|can't|neither|omit)\b",
            prefix,
            re.IGNORECASE,
        )
    )


def extract_matched_atoms_from_text(prompt: str) -> list[MatchedAtom]:
    """Extract requirement atoms with matched phrase and character spans."""
    matched_atoms: list[MatchedAtom] = []
    seen: set[str] = set()

    sentences = split_sentences(prompt)
    cursor = 0

    for sentence in sentences:
        clauses = split_clauses(sentence)
        for clause in clauses:
            clause_start = prompt.find(clause, cursor)
            if clause_start == -1:
                clause_start = prompt.find(clause)
            if clause_start != -1:
                cursor = clause_start

            for atom_name, spec in ATOM_REGISTRY.items():
                if atom_name in seen:
                    continue
                match = re.search(spec.trigger_regex, clause, re.IGNORECASE)
                if match:
                    # Role check for expert_level
                    if atom_name == "expert_level":
                        prefix = clause[: match.start()].lower()
                        if re.search(r"\b(?:act\s+as\s+(?:an?\s+)?|role:\s*|persona:\s*)$", prefix):
                            continue

                    is_negated = has_clause_negation(clause, match.start())
                    if spec.is_negative_requirement:
                        # Negative requirement (like no_basic, hidden_answers):
                        if not is_negated:
                            span_start = clause_start + match.start() if clause_start != -1 else 0
                            span_end = span_start + (match.end() - match.start())
                            matched_atoms.append(
                                MatchedAtom(
                                    atom=atom_name,
                                    matched_phrase=match.group(0),
                                    start=span_start,
                                    end=span_end,
                                    confidence=spec.confidence,
                                )
                            )
                            seen.add(atom_name)
                    else:
                        # Positive requirement:
                        if not is_negated:
                            span_start = clause_start + match.start() if clause_start != -1 else 0
                            span_end = span_start + (match.end() - match.start())
                            matched_atoms.append(
                                MatchedAtom(
                                    atom=atom_name,
                                    matched_phrase=match.group(0),
                                    start=span_start,
                                    end=span_end,
                                    confidence=spec.confidence,
                                )
                            )
                            seen.add(atom_name)

    return sorted(matched_atoms, key=lambda a: a.atom)


def extract_atoms_from_text(prompt: str) -> list[str]:
    """Extract normalized requirement atoms list."""
    matched = extract_matched_atoms_from_text(prompt)
    return [m.atom for m in matched]


def find_unclassified_clauses(prompt: str) -> list[str]:
    """Identify substantive instructional clauses that were not matched to any atom or slot."""
    unclassified = []
    directive_pattern = re.compile(
        r"\b(?:must|should|ensure|do\s+not|don't|require|requires?|make|create|arrange|provide|include|use|avoid|keep|write|label|mark|give|hide)\b",
        re.IGNORECASE,
    )
    sentences = split_sentences(prompt)
    for s in sentences:
        clauses = split_clauses(s)
        for c in clauses:
            if not directive_pattern.search(c):
                continue
            # Check if this clause matched any atom
            matched_any_atom = any(re.search(spec.trigger_regex, c, re.IGNORECASE) for spec in ATOM_REGISTRY.values())
            if matched_any_atom:
                continue
            # Check if this clause matched any slot full or partial pattern
            matched_any_slot = any(
                re.search(spec.full_pattern, c, re.IGNORECASE)
                or (spec.partial_pattern and re.search(spec.partial_pattern, c, re.IGNORECASE))
                for spec in SLOT_REGISTRY.values()
            )
            if matched_any_slot:
                continue

            # Standard pedagogical instructions (distractors, clarity, exam utility, Bloom range)
            pedagogical_pattern = re.compile(
                r"\b(?:distractors?|plausible|subtly\s+flawed|clear\s+and\s+concise|"
                r"keep\s+.*?explanations?|clear\s+and\s+unambiguous|basic\s+recall|"
                r"critical\s+thinking|apply\s+knowledge|useful\s+for\s+(?:their\s+)?exams?|"
                r"useful\s+for\s+examinations?|important\s+concepts?|important\s+topics?|"
                r"topics?\s+that\s+students\s+should\s+know|neither\s+too\s+easy|"
                r"interesting|challenging|suitab(?:le|ility))\b",
                re.IGNORECASE,
            )
            if pedagogical_pattern.search(c):
                continue
            clean_c = c.strip(" ,.-*•\t\n")
            if len(clean_c.split()) < 3:
                continue
            unclassified.append(clean_c)
    return unclassified


def score_slots_from_text(prompt: str) -> dict[str, float]:
    """Score the nine governed slots using lexical full- and partial-credit patterns."""
    sentences = split_sentences(prompt)
    scores: dict[str, float] = {}

    for slot_name, spec in SLOT_REGISTRY.items():
        matched_full = False
        matched_partial = False

        for sentence in sentences:
            m_full = re.search(spec.full_pattern, sentence, re.IGNORECASE)
            if m_full:
                # Check negation for ordering
                if slot_name == "ordering":
                    if has_clause_negation(sentence, m_full.start()):
                        continue
                if slot_name == "explanation":
                    if spec.partial_pattern and re.search(spec.partial_pattern, sentence, re.IGNORECASE):
                        matched_partial = True
                        continue
                if slot_name == "audience":
                    if not re.search(
                        r"\b(?:confuse\s+students|distract\s+students)\b", sentence, re.IGNORECASE
                    ) or re.search(
                        r"\b(?:for\s+(?:bachelor|master|undergraduate|postgraduate|graduate|students))\b",
                        sentence,
                        re.IGNORECASE,
                    ):
                        matched_full = True
                        break
                elif slot_name == "topic":
                    if not re.search(r"\b(?:cover\s+the\s+important\s+concepts)\b", sentence, re.IGNORECASE):
                        matched_full = True
                        break
                else:
                    matched_full = True
                    break

            if spec.partial_pattern:
                m_part = re.search(spec.partial_pattern, sentence, re.IGNORECASE)
                if m_part:
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
