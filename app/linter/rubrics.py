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
