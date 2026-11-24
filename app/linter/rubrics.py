"""Unified Slot Scoring Rubrics.

Single source of truth for the nine governed prompt slots.
Used by both the extractor logic and the labeling documentation.
Scores for each slot are strictly 0.0, 0.5, or 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlotRubric:
    slot: str
    name: str
    description: str
    score_0_rule: str
    score_0_5_rule: str
    score_1_0_rule: str
    examples: dict[str, str]  # keys: "1.0", "0.5", "0.0"


@dataclass(frozen=True)
class AtomRubric:
    atom: str
    name: str
    description: str
    positive_rule: str
    negative_rule: str
    example: str


RUBRIC_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"

ATOM_RUBRICS: dict[str, AtomRubric] = {
    "very_easy": AtomRubric(
        atom="very_easy",
        name="Very Easy Questions",
        description="Request for very easy, basic, or elementary difficulty questions.",
        positive_rule="Prompt explicitly asks for very easy, simple, elementary, or beginner-level questions.",
        negative_rule="Prompt does not ask for very easy questions or negates it (e.g., 'not easy', 'no easy questions').",
        example="All questions must be very easy so that any beginner student can answer.",
    ),
    "expert_level": AtomRubric(
        atom="expert_level",
        name="Expert Level Questions",
        description="Demand for advanced, expert, or high-difficulty questions.",
        positive_rule="Prompt demands advanced, expert-level, or deep technical questions.",
        negative_rule="Prompt does not demand expert-level questions, or merely mentions 'expert' as the teacher persona.",
        example="Questions must require expert-level knowledge and deep research understanding.",
    ),
    "no_basic": AtomRubric(
        atom="no_basic",
        name="No Basic Questions",
        description="Explicit prohibition against basic, trivial, or easy questions.",
        positive_rule="Prompt explicitly forbids easy, introductory, or basic questions.",
        negative_rule="Prompt allows basic questions or does not prohibit them.",
        example="Do not include any basic or introductory questions.",
    ),
    "ordered_easy_to_hard": AtomRubric(
        atom="ordered_easy_to_hard",
        name="Ordered Easy to Hard",
        description="Mandate to order questions in ascending difficulty from easiest to hardest.",
        positive_rule="Prompt explicitly directs questions to be sorted or arranged from easiest to hardest.",
        negative_rule="Prompt does not require difficulty ordering or specifies random/reverse ordering.",
        example="Organize the questions in order of increasing difficulty, starting with the easiest.",
    ),
    "hidden_answers": AtomRubric(
        atom="hidden_answers",
        name="Hidden Answers",
        description="Directive to hide, withhold, or not reveal the answers.",
        positive_rule="Prompt demands that the answers or answer key remain hidden or not provided.",
        negative_rule="Prompt requests providing answers or does not forbid showing them.",
        example="Do not show the answers or provide an answer key anywhere.",
    ),
    "answer_after_each_q": AtomRubric(
        atom="answer_after_each_q",
        name="Answer After Each Question",
        description="Requirement to display the answer immediately following each individual question.",
        positive_rule="Prompt requires the answer to appear directly after every single question.",
        negative_rule="Prompt does not specify showing answers after each question.",
        example="After each question, show the correct answer and explain why it is correct.",
    ),
    "single_correct": AtomRubric(
        atom="single_correct",
        name="Single Correct Option",
        description="Constraint that exactly one answer option is correct per question.",
        positive_rule="Prompt mandates only one right answer, exactly one correct choice, or strictly one answer.",
        negative_rule="Prompt allows multiple correct answers or does not specify single correctness.",
        example="Each question must have four choices, with strictly one correct answer.",
    ),
    "multi_interp": AtomRubric(
        atom="multi_interp",
        name="Multiple Interpretations",
        description="Directive for questions or options to allow multiple interpretations or ambiguous readings.",
        positive_rule="Prompt asks for options or questions that can be interpreted in multiple ways.",
        negative_rule="Prompt requires clarity or does not ask for multiple interpretations.",
        example="Include options that can be interpreted in multiple ways.",
    ),
    "confusing_language": AtomRubric(
        atom="confusing_language",
        name="Confusing Language",
        description="Demand for tricky phrasing, convoluted wording, or tricky double negatives.",
        positive_rule="Prompt demands tricky, confusing, or misleading wording.",
        negative_rule="Prompt demands clear, simple, or unambiguous wording.",
        example="Use confusing language and double negatives to confuse students.",
    ),
}


SLOT_RUBRICS: dict[str, SlotRubric] = {
    "role": SlotRubric(
        slot="role",
        name="Educator Persona / Role",
        description="Persona framing for the educator or question author.",
        score_0_rule="No persona or educator role mentioned.",
        score_0_5_rule="Educator or teaching role mentioned without persona framing (e.g., 'for the teacher', 'as an educator').",
        score_1_0_rule="Explicit role/persona framing instruction (e.g., 'Act as a university teacher', 'Role: professor', 'experienced instructor').",
        examples={
            "1.0": "Act as an experienced professor of computer science.",
            "0.5": "Write questions that an instructor can use in class.",
            "0.0": "Generate multiple choice questions about sorting algorithms.",
        },
    ),
    "audience": SlotRubric(
        slot="audience",
        name="Target Audience / Cohort",
        description="Target educational cohort or student level.",
        score_0_rule="No audience specified, or students mentioned only incidentally (e.g., 'to confuse students').",
        score_0_5_rule="Generic audience framing without specific cohort (e.g., 'for students', 'aimed at learners', 'suitable for a class').",
        score_1_0_rule="Specific educational cohort named (e.g., 'bachelor students', 'undergraduate', 'postgraduate', 'high school', 'freshman').",
        examples={
            "1.0": "Create questions suitable for second-year bachelor students.",
            "0.5": "Make the questions appropriate for students taking an exam.",
            "0.0": "Generate tricky questions to confuse students.",
        },
    ),
    "topic": SlotRubric(
        slot="topic",
        name="Academic Subject / Topic",
        description="Specific academic topic, course, or subject area.",
        score_0_rule="No topic or discipline specified, or generic phrases ('cover important concepts').",
        score_0_5_rule="Broad academic discipline without a specific topic (e.g., 'on computer science', 'in biology').",
        score_1_0_rule="Specific subject topic or identified course (e.g., 'on Normalization', 'about B-trees', 'operating systems deadlocks').",
        examples={
            "1.0": "Prepare questions on Database Normalization and functional dependencies.",
            "0.5": "Prepare multiple choice questions on computer science.",
            "0.0": "Make sure the questions cover all the important concepts.",
        },
    ),
    "count": SlotRubric(
        slot="count",
        name="Question Quantity",
        description="Total quantity of questions requested.",
        score_0_rule="Non-specific quantity ('some questions', 'a set of MCQs', 'several questions').",
        score_0_5_rule="Approximate or bounded count ('around 10 questions', 'between 5 and 10 questions', 'at least 5 questions').",
        score_1_0_rule="Specific exact count with number word or digits (e.g., '10 questions', '20 MCQs', 'five questions', '15 items').",
        examples={
            "1.0": "Create exactly 15 multiple-choice questions.",
            "0.5": "Create around 10 to 12 questions.",
            "0.0": "Create some multiple-choice questions for the quiz.",
        },
    ),
    "n_options": SlotRubric(
        slot="n_options",
        name="Number of Options per Question",
        description="Number of choices or alternatives for each question.",
        score_0_rule="No mention of choices or options.",
        score_0_5_rule="Generic mention of choices/options without count ('multiple choices', 'options for each question').",
        score_1_0_rule="Explicit number of options specified (e.g., 'four options', '4 choices', 'five alternatives', '3 options').",
        examples={
            "1.0": "Provide four choices for every question.",
            "0.5": "Each question should have multiple choices.",
            "0.0": "Write questions with answer keys.",
        },
    ),
    "labeling": SlotRubric(
        slot="labeling",
        name="Option Labeling Scheme",
        description="Identifier scheme for options (A-D, 1-4, etc.).",
        score_0_rule="No labeling scheme mentioned.",
        score_0_5_rule="Generic instruction to label options without scheme ('label each choice', 'use letters').",
        score_1_0_rule="Explicit labeling scheme specified (e.g., 'labelled A, B, C, D', '(A, B, C, D)', 'options A to D').",
        examples={
            "1.0": "Label each answer choice (A, B, C, D).",
            "0.5": "Make sure each option has a clear label.",
            "0.0": "Provide four choices for each question.",
        },
    ),
    "marking": SlotRubric(
        slot="marking",
        name="Correct Answer Marking",
        description="Specification for identifying or marking the correct answer.",
        score_0_rule="No indication of marking or providing correct answers (or purely negative: 'do not mark answers').",
        score_0_5_rule="Vague mention of answers ('include answers', 'with answers').",
        score_1_0_rule="Explicit instruction to mark, highlight, or supply the correct answer (e.g., 'mark the right answer', 'highlight correct answer in bold', 'provide the correct answer').",
        examples={
            "1.0": "Highlight the correct answer in bold after each question.",
            "0.5": "Include answers with the questions.",
            "0.0": "Write multiple choice questions for a classroom quiz.",
        },
    ),
    "explanation": SlotRubric(
        slot="explanation",
        name="Answer Explanation / Rationale",
        description="Requirement for rationale or explanations for answers.",
        score_0_rule="No explanation mentioned.",
        score_0_5_rule="Conditional or optional explanations ('explain wherever necessary', 'explanations optional', 'if needed').",
        score_1_0_rule="Mandatory explanation required (e.g., 'provide detailed explanations', 'explain why each option is correct or incorrect').",
        examples={
            "1.0": "Provide an explanation for why the chosen answer is correct.",
            "0.5": "Explain the answers wherever necessary.",
            "0.0": "Provide four choices and mark the right answer.",
        },
    ),
    "ordering": SlotRubric(
        slot="ordering",
        name="Question Ordering / Arrangement",
        description="Pedagogical sequence or difficulty progression of questions.",
        score_0_rule="No ordering instruction, or random/shuffled order.",
        score_0_5_rule="Generic ordering instruction without difficulty criterion ('arrange them properly', 'order logically').",
        score_1_0_rule="Specific pedagogical or difficulty ordering (e.g., 'in order of increasing difficulty', 'from easiest to hardest').",
        examples={
            "1.0": "Organize the questions in order of increasing difficulty from easiest to hardest.",
            "0.5": "Arrange the questions properly in a logical sequence.",
            "0.0": "Write 10 questions on software testing.",
        },
    ),
}


def get_slot_rubrics_markdown() -> str:
    """Generate Markdown documentation of the 9 slot rubrics."""
    lines = [
        "# Nine Governed Prompt Slot Rubrics",
        "",
        "Every slot receives a score of `0.0`, `0.5`, or `1.0`.",
        "",
        "| Slot | 1.0 (Full Credit) | 0.5 (Partial Credit) | 0.0 (No Credit) |",
        "|---|---|---|---|",
    ]
    for slot, r in SLOT_RUBRICS.items():
        lines.append(f"| **{r.slot}** ({r.name}) | {r.score_1_0_rule} | {r.score_0_5_rule} | {r.score_0_rule} |")
    lines.append("")
    lines.append("## Examples by Slot")
    lines.append("")
    for slot, r in SLOT_RUBRICS.items():
        lines.append(f"### `{r.slot}`: {r.name}")
        lines.append(f'- **1.0**: "{r.examples["1.0"]}"')
        lines.append(f'- **0.5**: "{r.examples["0.5"]}"')
        lines.append(f'- **0.0**: "{r.examples["0.0"]}"')
        lines.append("")
    return "\n".join(lines)
