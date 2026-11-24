"""Generate 68 diverse dev prompts for requirement extractor development.

Covers terse, verbose, non-native, bullet lists, numbered rules,
negations, paraphrases, separate-sentence conflicts, and zero-atom prompts.
All rows are marked source: authored_by_builder, synthetic: 1.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.linter.rubrics import get_slot_rubrics_markdown

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "gold" / "dev_authored"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEV_PROMPTS = [
    # -------------------------------------------------------------
    # Category 1: Terse Prompts (5 prompts)
    # -------------------------------------------------------------
    {
        "id": "dev-01",
        "voice": "terse",
        "prompt": "Act as professor. 10 MCQs on Normalization for bachelor students. 4 choices (A, B, C, D). Mark correct. Easy to hard.",
        "atoms": ["ordered_easy_to_hard"],
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 1.0,
        },
    },
    {
        "id": "dev-02",
        "voice": "terse",
        "prompt": "Create 5 MCQs on CPU scheduling. Four options each. Give answers.",
        "atoms": [],
        "slots": {
            "role": 0.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-03",
        "voice": "terse",
        "prompt": "Role: university teacher. 20 MCQs on binary search trees. 4 options (A-D). Explain every answer. All very easy.",
        "atoms": ["very_easy"],
        "slots": {
            "role": 1.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 0.0,
            "explanation": 1.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-04",
        "voice": "terse",
        "prompt": "15 MCQs on graph theory for undergraduates. 4 choices. Mark answer. Skip basic questions. Expert level.",
        "atoms": ["no_basic", "expert_level"],
        "slots": {
            "role": 0.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-05",
        "voice": "terse",
        "prompt": "MCQs on networking. Hide answers from students. Put answer key at the end.",
        "atoms": ["hidden_answers"],
        "slots": {
            "role": 0.0,
            "audience": 0.5,
            "topic": 1.0,
            "count": 0.0,
            "n_options": 0.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    # -------------------------------------------------------------
    # Category 2: Verbose Prompts (5 prompts)
    # -------------------------------------------------------------
    {
        "id": "dev-06",
        "voice": "verbose",
        "prompt": (
            "You are to act in the professional capacity of a distinguished university professor lecturing to senior bachelor students. "
            "Your task is to compose an extensive examination consisting of precisely 25 multiple-choice questions focusing exclusively on "
            "distributed transactions and consensus protocols. Each item must present four clearly distinguished alternatives labelled A, B, C, and D. "
            "Highlight the correct answer in bold text immediately below the question stem, and subsequently provide a comprehensive, detailed explanation "
            "clarifying why the chosen alternative is sound while detailing the precise technical misconceptions in the other three alternatives. "
            "Structure the sequence of questions systematically so that they progress from the easiest introductory items to the hardest analytical scenarios."
        ),
        "atoms": ["ordered_easy_to_hard"],
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 1.0,
            "explanation": 1.0,
            "ordering": 1.0,
        },
    },
    {
        "id": "dev-07",
        "voice": "verbose",
        "prompt": (
            "Please function as an experienced instructor who teaches graduate seminar courses in computer science. "
            "Generate twelve multiple-choice questions on cryptographic protocols for postgraduate learners. "
            "Every question requires four options labelled A through D. Make sure that all questions require advanced research knowledge "
            "and expert-level understanding of zero-knowledge proofs. Omit any basic questions from this assessment. "
            "Directly after each question, provide the answer and write a full technical explanation for every choice."
        ),
        "atoms": ["expert_level", "no_basic", "answer_after_each_q"],
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 1.0,
            "explanation": 1.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-08",
        "voice": "verbose",
        "prompt": (
            "Act as a university professor preparing a rigorous quiz on memory management and virtual memory for undergraduate students. "
            "Produce thirty questions with four answer choices labelled A, B, C, and D. Do not reveal the correct answer to students under any circumstances, "
            "as this will be administered as an in-class test. Provide a separate master answer key at the very end of the document. "
            "Ensure that exactly one option is the correct answer for every question without exception."
        ),
        "atoms": ["hidden_answers", "single_correct"],
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-09",
        "voice": "verbose",
        "prompt": (
            "Serve as a university teacher responsible for assessing bachelor cohort knowledge in algorithms and computational complexity. "
            "Develop 10 multiple-choice questions on NP-completeness. Each problem must feature four choices labelled (A, B, C, D). "
            "Construct questions using deliberately confusing language and tricky statements so that students are forced to re-read every premise. "
            "Include options that can be interpreted in multiple ways, testing whether students can detect ambiguity. "
            "Provide the correct answer and a thorough rationale explaining why the distractors fail."
        ),
        "atoms": ["confusing_language", "multi_interp"],
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 1.0,
            "explanation": 1.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-10",
        "voice": "verbose",
        "prompt": (
            "Act as an academic instructor. Prepare a series of multiple-choice questions covering object-oriented programming principles. "
            "Ensure that the questions help students review fundamental concepts. Give four options for each item and specify which choice is right. "
            "Explain the answers wherever necessary to ensure comprehension."
        ),
        "atoms": [],
        "slots": {
            "role": 1.0,
            "audience": 0.5,
            "topic": 1.0,
            "count": 0.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.5,
            "ordering": 0.0,
        },
    },
    # -------------------------------------------------------------
    # Category 3: Non-native English phrasing (8 prompts)
    # -------------------------------------------------------------
    {
        "id": "dev-11",
        "voice": "non_native",
        "prompt": "Please act as university teacher. Make 10 question on database Normalization for bachelor students. 4 choices with A, B, C, D letters. Mark right choice. Make all question very easy so everyone pass.",
        "atoms": ["very_easy"],
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-12",
        "voice": "non_native",
        "prompt": "You are teacher for university. Write 15 MCQ about operating systems. Four options each. After each question show the answer and give reason. Do not put basic question, only expert level.",
        "atoms": ["answer_after_each_q", "no_basic", "expert_level"],
        "slots": {
            "role": 1.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 1.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-13",
        "voice": "non_native",
        "prompt": "Act as professor. Create 20 test questions for computer science students about recursion. Four choice A B C D. Correct answer keep hidden from students. Sort from easiest to hardest.",
        "atoms": ["hidden_answers", "ordered_easy_to_hard"],
        "slots": {
            "role": 1.0,
            "audience": 0.5,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 1.0,
        },
    },
    {
        "id": "dev-14",
        "voice": "non_native",
        "prompt": "Teacher role. 8 MCQ on SQL indexing for undergraduate learners. 4 options. Strictly only one correct answer per question. Write clear explanation for every choice.",
        "atoms": ["single_correct"],
        "slots": {
            "role": 0.5,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 1.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-15",
        "voice": "non_native",
        "prompt": "Act as college instructor. 12 multiple choice on machine learning. Four choices. Use tricky language and statements that can be interpreted in multiple ways to challenge them. Explain why others wrong.",
        "atoms": ["confusing_language", "multi_interp"],
        "slots": {
            "role": 1.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 0.0,
            "explanation": 1.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-16",
        "voice": "non_native",
        "prompt": "Prepare some questions about data structures for students. Provide four options and answers. Explain if necessary.",
        "atoms": [],
        "slots": {
            "role": 0.0,
            "audience": 0.5,
            "topic": 1.0,
            "count": 0.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.5,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-17",
        "voice": "non_native",
        "prompt": "Act as professor. 10 questions on compiler design for bachelor cohort. Must have four choices labelled A, B, C, D. Ensure every question is clear and unambiguous without room for misinterpretation. Order by increasing difficulty.",
        "atoms": ["clear_unambiguous", "ordered_easy_to_hard"],
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 0.0,
            "explanation": 0.0,
            "ordering": 1.0,
        },
    },
    {
        "id": "dev-18",
        "voice": "non_native",
        "prompt": "Make 20 MCQs on software engineering. Give 4 choices. Don't reveal answers to students until test is finished.",
        "atoms": ["hidden_answers"],
        "slots": {
            "role": 0.0,
            "audience": 0.5,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    # -------------------------------------------------------------
    # Category 4: Bullet lists (8 prompts)
    # -------------------------------------------------------------
    {
        "id": "dev-19",
        "voice": "bullet_list",
        "prompt": (
            "Task instructions:\n"
            "- Role: Act as a university professor\n"
            "- Audience: Bachelor students in computing\n"
            "- Topic: Relational algebra\n"
            "- Count: 10 multiple-choice questions\n"
            "- Options: 4 choices labelled (A, B, C, D)\n"
            "- Answer key: Highlight the correct answer in bold\n"
            "- Explanations: Include an explanation for every option\n"
            "- Sequence: Progressing from the easiest to the hardest"
        ),
        "atoms": ["ordered_easy_to_hard"],
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 1.0,
            "explanation": 1.0,
            "ordering": 1.0,
        },
    },
    {
        "id": "dev-20",
        "voice": "bullet_list",
        "prompt": (
            "* Instructor persona: Experienced teacher\n"
            "* Course topic: Computer networks and socket programming\n"
            "* Quantity: 15 questions\n"
            "* Choices: Four options each\n"
            "* Difficulty requirement: All questions must be very easy\n"
            "* Explanation: Provide detailed explanations"
        ),
        "atoms": ["very_easy"],
        "slots": {
            "role": 1.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 0.0,
            "explanation": 1.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-21",
        "voice": "bullet_list",
        "prompt": (
            "- Act as university lecturer for undergraduate class\n"
            "- 20 MCQs on operating systems kernels\n"
            "- 4 options labelled A to D\n"
            "- Skip the basics completely; no basic questions allowed\n"
            "- Must require advanced research knowledge and expert level understanding\n"
            "- Answer and explanation after each question"
        ),
        "atoms": ["no_basic", "expert_level", "answer_after_each_q"],
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 1.0,
            "explanation": 1.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-22",
        "voice": "bullet_list",
        "prompt": (
            "* Topic: Cryptography and hashing algorithms\n"
            "* Target: Postgraduate students\n"
            "* Format: 10 MCQs with four choices\n"
            "* Answers: Keep answers hidden from students; put answer key at the end\n"
            "* Feedback: Detailed explanation for each correct answer"
        ),
        "atoms": ["hidden_answers"],
        "slots": {
            "role": 0.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 1.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-23",
        "voice": "bullet_list",
        "prompt": (
            "- Role: College professor\n"
            "- Subject: Web security\n"
            "- Items: 12 multiple choice questions\n"
            "- Options: Four choices labelled A, B, C, D\n"
            "- Correctness: Exactly one correct option per question\n"
            "- Clarity: Ensure each question is clear and unambiguous"
        ),
        "atoms": ["single_correct", "clear_unambiguous"],
        "slots": {
            "role": 1.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-24",
        "voice": "bullet_list",
        "prompt": (
            "* Act as professor of software engineering\n"
            "* 15 MCQs for master students\n"
            "* Four options each\n"
            "* Use tricky wording and confusing language to test attention\n"
            "* Include options that can be interpreted in multiple ways\n"
            "* Mark correct choice and explain why distractors fail"
        ),
        "atoms": ["confusing_language", "multi_interp"],
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 1.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-25",
        "voice": "bullet_list",
        "prompt": (
            "- Create multiple choice questions for students\n"
            "- Subject: Computer architecture\n"
            "- Four choices for each\n"
            "- Mark the right answer\n"
            "- Arrange properly"
        ),
        "atoms": [],
        "slots": {
            "role": 0.0,
            "audience": 0.5,
            "topic": 1.0,
            "count": 0.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.5,
        },
    },
    {
        "id": "dev-26",
        "voice": "bullet_list",
        "prompt": (
            "* Act as a university teacher\n"
            "* Topic: Calculus and integration techniques\n"
            "* Count: 8 questions for freshman students\n"
            "* Four options labelled A, B, C, D\n"
            "* Sort in ascending order of difficulty from easiest to hardest"
        ),
        "atoms": ["ordered_easy_to_hard"],
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 0.0,
            "explanation": 0.0,
            "ordering": 1.0,
        },
    },
    # -------------------------------------------------------------
    # Category 5: Numbered rules (8 prompts)
    # -------------------------------------------------------------
    {
        "id": "dev-27",
        "voice": "numbered_rules",
        "prompt": (
            "Rule 1: Act as an experienced professor in computer science.\n"
            "Rule 2: Create 10 MCQs on database Normalization for bachelor students.\n"
            "Rule 3: Give four options for each question labelled A, B, C, and D.\n"
            "Rule 4: Mark the correct answer in bold.\n"
            "Rule 5: After every question, provide the answer and a detailed explanation.\n"
            "Rule 6: Organize from the easiest to the hardest."
        ),
        "atoms": ["answer_after_each_q", "ordered_easy_to_hard"],
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 1.0,
            "explanation": 1.0,
            "ordering": 1.0,
        },
    },
    {
        "id": "dev-28",
        "voice": "numbered_rules",
        "prompt": (
            "1. Persona: university teacher.\n"
            "2. Subject: Artificial intelligence heuristics.\n"
            "3. Quantity: 20 MCQs for graduate students.\n"
            "4. Options: 4 choices per item.\n"
            "5. Requirement: All questions must be very easy so every student can answer.\n"
            "6. Provide explanations wherever necessary."
        ),
        "atoms": ["very_easy"],
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 0.0,
            "explanation": 0.5,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-29",
        "voice": "numbered_rules",
        "prompt": (
            "1. Act as university instructor.\n"
            "2. Prepare 15 MCQs on operating systems.\n"
            "3. Provide four choices for each item.\n"
            "4. Avoid basic questions; do not include any basic questions.\n"
            "5. Test expert-level understanding and complex problem-solving.\n"
            "6. State the answer after each question."
        ),
        "atoms": ["no_basic", "expert_level", "answer_after_each_q"],
        "slots": {
            "role": 1.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-30",
        "voice": "numbered_rules",
        "prompt": (
            "1. Role: professor.\n"
            "2. Target: bachelor students.\n"
            "3. Topic: Data structures and binary heaps.\n"
            "4. 10 MCQs with four options labelled (A, B, C, D).\n"
            "5. Do not reveal the correct answer to students.\n"
            "6. Put answers at the end in an answer key."
        ),
        "atoms": ["hidden_answers"],
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-31",
        "voice": "numbered_rules",
        "prompt": (
            "Rule 1: Prepare 25 MCQs on computer networks.\n"
            "Rule 2: Give four options for every question.\n"
            "Rule 3: Ensure there is only one correct answer per item.\n"
            "Rule 4: Write clear, unambiguous questions leaving no room for misinterpretation.\n"
            "Rule 5: Highlight the right choice."
        ),
        "atoms": ["single_correct", "clear_unambiguous"],
        "slots": {
            "role": 0.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-32",
        "voice": "numbered_rules",
        "prompt": (
            "1. Act as university teacher.\n"
            "2. Create 15 questions on cloud computing.\n"
            "3. Give four choices labelled A-D.\n"
            "4. Use confusing language and tricky statements to test critical reading.\n"
            "5. Allow options that can be interpreted in multiple ways.\n"
            "6. Supply the correct answer and explanation."
        ),
        "atoms": ["confusing_language", "multi_interp"],
        "slots": {
            "role": 1.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 1.0,
            "explanation": 1.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-33",
        "voice": "numbered_rules",
        "prompt": (
            "1. Act as expert professor.\n"
            "2. Make 10 MCQs on algorithms.\n"
            "3. Provide four choices for each.\n"
            "4. Mark right answer.\n"
            "5. Arrange them properly."
        ),
        "atoms": [],
        "slots": {
            "role": 1.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.5,
        },
    },
    {
        "id": "dev-34",
        "voice": "numbered_rules",
        "prompt": (
            "Rule 1: Teacher persona for sophomore cohort.\n"
            "Rule 2: Topic is linear algebra and matrix rank.\n"
            "Rule 3: Generate 12 MCQs with 4 choices labelled A, B, C, D.\n"
            "Rule 4: Sequence questions in order of increasing difficulty from easiest to hardest."
        ),
        "atoms": ["ordered_easy_to_hard"],
        "slots": {
            "role": 0.5,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 0.0,
            "explanation": 0.0,
            "ordering": 1.0,
        },
    },
    # -------------------------------------------------------------
    # Category 6: Negations ("do not show answers", etc.) (8 prompts)
    # -------------------------------------------------------------
    {
        "id": "dev-35",
        "voice": "negations",
        "prompt": "Act as professor. 10 MCQs on database systems. Do not reveal the correct answer to students until after grading. Four choices each.",
        "atoms": ["hidden_answers"],
        "slots": {
            "role": 1.0,
            "audience": 0.5,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-36",
        "voice": "negations",
        "prompt": "Create 15 MCQs on machine learning. Do not include any basic questions; skip introductory concepts. 4 options labelled A to D.",
        "atoms": ["no_basic"],
        "slots": {
            "role": 0.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 0.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-37",
        "voice": "negations",
        "prompt": "Act as a university teacher for bachelor students. Write 20 MCQs on operating systems. The questions should not be very easy; make them challenging. 4 options.",
        "atoms": [],  # "not very easy" negates very_easy
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 0.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-38",
        "voice": "negations",
        "prompt": "10 questions on Python programming. Avoid confusing language and tricky statements. Questions must be clear and unambiguous. 4 choices.",
        "atoms": ["clear_unambiguous"],  # "Avoid confusing language" negates confusing_language
        "slots": {
            "role": 0.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 0.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-39",
        "voice": "negations",
        "prompt": "Role: university professor. 12 MCQs on cryptography. Never show the answers to students on the question sheet. Keep answers hidden. 4 options (A-D).",
        "atoms": ["hidden_answers"],
        "slots": {
            "role": 1.0,
            "audience": 0.5,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-40",
        "voice": "negations",
        "prompt": "15 MCQs on computer security. Do not provide an answer key at the end; instead give the answer after each question. 4 choices.",
        "atoms": ["answer_after_each_q"],
        "slots": {
            "role": 0.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-41",
        "voice": "negations",
        "prompt": "Act as instructor. 8 questions on software architecture. Ensure options cannot be interpreted in multiple ways; make sure there is only one correct answer.",
        "atoms": ["single_correct"],  # "cannot be interpreted in multiple ways" negates multi_interp
        "slots": {
            "role": 1.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 0.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-42",
        "voice": "negations",
        "prompt": "Create 10 questions on compiler design. Do not arrange from easiest to hardest; randomize the order. 4 options each.",
        "atoms": [],  # "Do not arrange from easiest to hardest" negates ordered_easy_to_hard
        "slots": {
            "role": 0.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 0.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    # -------------------------------------------------------------
    # Category 7: Paraphrases (8 prompts)
    # -------------------------------------------------------------
    {
        "id": "dev-43",
        "voice": "paraphrases",
        "prompt": "Act as professor. 10 MCQs on database Normalization for bachelor students. Four choices (A, B, C, D). Exactly one option is correct. Explain each answer.",
        "atoms": ["single_correct"],
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 1.0,
            "explanation": 1.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-44",
        "voice": "paraphrases",
        "prompt": "Role: teacher. 15 questions on sorting algorithms. Four choices. Just one option is correct per question, with no second right answer.",
        "atoms": ["single_correct"],
        "slots": {
            "role": 1.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-45",
        "voice": "paraphrases",
        "prompt": "Create 12 MCQs on operating systems. 4 options. Make sure every item has strictly one right answer even when distractors look plausible.",
        "atoms": ["single_correct"],
        "slots": {
            "role": 0.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-46",
        "voice": "paraphrases",
        "prompt": "Act as university teacher. 20 MCQs on machine learning. Four choices. Arrange questions in order of increasing difficulty from easiest to hardest.",
        "atoms": ["ordered_easy_to_hard"],
        "slots": {
            "role": 1.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 0.0,
            "explanation": 0.0,
            "ordering": 1.0,
        },
    },
    {
        "id": "dev-47",
        "voice": "paraphrases",
        "prompt": "Prepare 10 MCQs on data structures for undergraduate students. Four options labelled A to D. Progress from simplest recall items to most difficult critical thinking problems.",
        "atoms": ["ordered_easy_to_hard"],
        "slots": {
            "role": 0.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 0.0,
            "explanation": 0.0,
            "ordering": 1.0,
        },
    },
    {
        "id": "dev-48",
        "voice": "paraphrases",
        "prompt": "Act as professor. 15 MCQs on computer networks. 4 choices. All questions must test advanced research knowledge and expert level understanding.",
        "atoms": ["expert_level"],
        "slots": {
            "role": 1.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 0.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-49",
        "voice": "paraphrases",
        "prompt": "Act as instructor. 8 MCQs on digital circuits for bachelor learners. Four choices (A-D). Questions must be beginner level and very easy for everyone.",
        "atoms": ["very_easy"],
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 0.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-50",
        "voice": "paraphrases",
        "prompt": "10 questions on artificial intelligence. 4 choices. Withhold the answers from students and give answers at the end of the test in an answer key.",
        "atoms": ["hidden_answers"],
        "slots": {
            "role": 0.0,
            "audience": 0.5,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    # -------------------------------------------------------------
    # Category 8: Conflicts hidden across separate sentences (8 prompts)
    # -------------------------------------------------------------
    {
        "id": "dev-51",
        "voice": "separate_sentence_conflicts",
        "prompt": (
            "Act as a university teacher. Create 20 MCQs about machine learning for bachelor students. "
            "All questions must be very easy so every student can answer them. "
            "However, every question must also test advanced research knowledge and expert-level understanding. "
            "Provide four options and mark the answer."
        ),
        "atoms": ["very_easy", "expert_level"],
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-52",
        "voice": "separate_sentence_conflicts",
        "prompt": (
            "Act as a professor. Generate 15 MCQs on computer systems. "
            "Make all questions very easy. "
            "Do not include any basic questions in this examination. "
            "Give four choices for each question."
        ),
        "atoms": ["very_easy", "no_basic"],
        "slots": {
            "role": 1.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 0.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-53",
        "voice": "separate_sentence_conflicts",
        "prompt": (
            "Act as an experienced instructor. Create 10 MCQs on database management systems. "
            "Do not reveal the correct answer to students. "
            "After every question, provide the answer and a detailed explanation. "
            "Each question should have four options labelled A, B, C, D."
        ),
        "atoms": ["hidden_answers", "answer_after_each_q"],
        "slots": {
            "role": 1.0,
            "audience": 0.5,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 1.0,
            "explanation": 1.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-54",
        "voice": "separate_sentence_conflicts",
        "prompt": (
            "Act as a university professor. Prepare 25 MCQs on operating systems for undergraduates. "
            "Make sure there is only one correct answer for each item. "
            "Include options that can be interpreted in multiple ways so that several choices seem plausible. "
            "Every question must have four options."
        ),
        "atoms": ["single_correct", "multi_interp"],
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-55",
        "voice": "separate_sentence_conflicts",
        "prompt": (
            "You are a teacher designing an exam on algorithms. "
            "Ensure each question is clear and unambiguous, leaving no room for misinterpretation. "
            "Use confusing language and tricky statements to test critical thinking. "
            "Provide four options labelled A, B, C, D for all 10 questions."
        ),
        "atoms": ["clear_unambiguous", "confusing_language"],
        "slots": {
            "role": 1.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 0.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-56",
        "voice": "separate_sentence_conflicts",
        "prompt": (
            "Role: university instructor. Topic: software testing. Count: 12 MCQs for bachelor cohort. "
            "Every question must be clear and unambiguous with strictly defined wording. "
            "Write options that can be interpreted in multiple ways so that interpretation matters. "
            "Provide four choices and explanations."
        ),
        "atoms": ["clear_unambiguous", "multi_interp"],
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 0.0,
            "explanation": 1.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-57",
        "voice": "separate_sentence_conflicts",
        "prompt": (
            "Act as professor. Create 15 MCQs on discrete mathematics. "
            "All questions must be extremely easy and trivial for beginner students. "
            "At the same time, ensure no basic questions are included. "
            "Four choices per question."
        ),
        "atoms": ["very_easy", "no_basic"],
        "slots": {
            "role": 1.0,
            "audience": 0.5,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 0.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-58",
        "voice": "separate_sentence_conflicts",
        "prompt": (
            "Act as university teacher. 20 MCQs on distributed systems for postgraduate students. "
            "Keep the correct answer hidden from students; do not show answers on the exam. "
            "Display the answer after each question with a complete rationale. "
            "Provide four choices labelled A through D."
        ),
        "atoms": ["hidden_answers", "answer_after_each_q"],
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 1.0,
            "explanation": 1.0,
            "ordering": 0.0,
        },
    },
    # -------------------------------------------------------------
    # Category 9: Prompts with No Atoms (Zero Atoms) (10 prompts)
    # -------------------------------------------------------------
    {
        "id": "dev-59",
        "voice": "zero_atoms",
        "prompt": "Act as a university teacher and create some multiple-choice questions for students. Make the questions useful for their exams. Give four options for each question and provide the answers. Explain the answers wherever necessary.",
        "atoms": [],
        "slots": {
            "role": 1.0,
            "audience": 0.5,
            "topic": 0.0,
            "count": 0.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.5,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-60",
        "voice": "zero_atoms",
        "prompt": "Act as an expert professor and prepare MCQs on computer science. Include all the important topics that students should know. Provide four choices for every question and mark the right answer. Arrange them properly.",
        "atoms": [],
        "slots": {
            "role": 1.0,
            "audience": 0.5,
            "topic": 0.5,
            "count": 0.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.5,
        },
    },
    {
        "id": "dev-61",
        "voice": "zero_atoms",
        "prompt": "Create 10 multiple-choice questions on compiler design for bachelor students. Provide four options labelled A, B, C, D for each question and mark the correct answer.",
        "atoms": [],
        "slots": {
            "role": 0.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-62",
        "voice": "zero_atoms",
        "prompt": "Act as an instructor. Write 5 questions on software testing. Four choices each. Provide answers and detailed explanations.",
        "atoms": [],
        "slots": {
            "role": 1.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 1.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-63",
        "voice": "zero_atoms",
        "prompt": "Generate 15 MCQs on computer networks for undergraduate learners. Each question must have four options (A, B, C, D) and an explanation for the answer.",
        "atoms": [],
        "slots": {
            "role": 0.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 1.0,
            "explanation": 1.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-64",
        "voice": "zero_atoms",
        "prompt": "Prepare a quiz on data structures. 10 questions with 4 options each. Include the correct answer.",
        "atoms": [],
        "slots": {
            "role": 0.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-65",
        "voice": "zero_atoms",
        "prompt": "Role: university professor. Create 8 MCQs on linear algebra for engineering students. Four choices labelled A to D.",
        "atoms": [],
        "slots": {
            "role": 1.0,
            "audience": 0.5,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 0.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-66",
        "voice": "zero_atoms",
        "prompt": "Act as a teacher. Write 20 MCQs about web development. Give four choices and indicate the right choice. Explain if needed.",
        "atoms": [],
        "slots": {
            "role": 1.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.5,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-67",
        "voice": "zero_atoms",
        "prompt": "Create 12 questions on operating systems. 4 choices. Highlight the right choice in bold.",
        "atoms": [],
        "slots": {
            "role": 0.0,
            "audience": 0.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 0.0,
            "marking": 1.0,
            "explanation": 0.0,
            "ordering": 0.0,
        },
    },
    {
        "id": "dev-68",
        "voice": "zero_atoms",
        "prompt": "Act as an experienced professor. Generate 10 MCQs on database systems for bachelor students. Four options labelled A, B, C, D. Mark the right answer. Provide explanations.",
        "atoms": [],
        "slots": {
            "role": 1.0,
            "audience": 1.0,
            "topic": 1.0,
            "count": 1.0,
            "n_options": 1.0,
            "labeling": 1.0,
            "marking": 1.0,
            "explanation": 1.0,
            "ordering": 0.0,
        },
    },
]


def main() -> None:
    # 1. Write dev_prompts.json
    json_path = OUT_DIR / "dev_prompts.json"
    data_to_write = []
    for item in DEV_PROMPTS:
        row = {
            "prompt_id": item["id"],
            "voice": item["voice"],
            "prompt_text": item["prompt"],
            "source": "authored_by_builder",
            "synthetic": 1,
            "atoms": sorted(item["atoms"]),
            "slots": item["slots"],
        }
        data_to_write.append(row)

    json_path.write_text(json.dumps(data_to_write, indent=2), encoding="utf-8")
    print(f"Wrote {len(data_to_write)} prompts to {json_path}")

    # 2. Write dev_prompts.csv
    csv_path = OUT_DIR / "dev_prompts.csv"
    atom_names = [
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
    ]
    slot_names = ["role", "audience", "topic", "count", "n_options", "labeling", "marking", "explanation", "ordering"]
    fieldnames = ["prompt_id", "voice", "source", "synthetic", "prompt_text"] + atom_names + slot_names
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in DEV_PROMPTS:
            row_dict = {
                "prompt_id": item["id"],
                "voice": item["voice"],
                "source": "authored_by_builder",
                "synthetic": 1,
                "prompt_text": item["prompt"],
            }
            for a in atom_names:
                row_dict[a] = 1 if a in item["atoms"] else 0
            for s in slot_names:
                row_dict[s] = item["slots"].get(s, 0.0)
            writer.writerow(row_dict)
    print(f"Wrote {len(DEV_PROMPTS)} rows to {csv_path}")

    # 3. Write README.md
    rubrics_md = get_slot_rubrics_markdown()
    readme_lines = [
        "# Development Dataset: Authored Prompts",
        "",
        "> **PROVENANCE LABEL**: `source: authored_by_builder`, `synthetic: 1`",
        "> This development dataset was authored by the builder for tuning the rule-based requirement extractor.",
        "> It is strictly separate from the frozen test set and any real human data.",
        "",
        f"Total prompts in dev set: {len(DEV_PROMPTS)}",
        "",
        "## Voice Distribution",
        "",
        "- Terse: 5 prompts",
        "- Verbose: 5 prompts",
        "- Non-native English phrasing: 8 prompts",
        "- Bullet lists: 8 prompts",
        "- Numbered rules: 8 prompts",
        "- Negations (e.g., 'do not show answers', 'avoid basic'): 8 prompts",
        "- Paraphrases (e.g., 'strictly one right answer', 'simplest to hardest'): 8 prompts",
        "- Hidden conflicts across separate sentences: 8 prompts",
        "- Zero atoms (standard prompts with no special atoms): 10 prompts",
        "",
        "## Atom Labelling Rules",
        "",
        "1. `very_easy`: Questions must be very easy, beginner-level, or elementary.",
        "2. `expert_level`: Questions must assess advanced research knowledge or expert-level understanding.",
        "3. `no_basic`: Instruction explicitly excluding, skipping, or avoiding basic/introductory questions.",
        "4. `hidden_answers`: The correct answer must be concealed or withheld from students.",
        "5. `answer_after_each_q`: The answer must be given immediately after each question.",
        "6. `single_correct`: Requirement that exactly one option is the correct answer.",
        "7. `multi_interp`: Requirement allowing multiple interpretations or ambiguity in choices.",
        "8. `clear_unambiguous`: Requirement that questions and wording must be clear and unambiguous.",
        "9. `confusing_language`: Deliberate instruction to use confusing language or tricky statements.",
        "10. `ordered_easy_to_hard`: Ordering questions from easiest to hardest or by increasing difficulty.",
        "",
        "## Conflict Pairs",
        "",
        "- `very_easy` conflicts with `expert_level`",
        "- `very_easy` conflicts with `no_basic`",
        "- `hidden_answers` conflicts with `answer_after_each_q`",
        "- `single_correct` conflicts with `multi_interp`",
        "- Policy conflicts: `clear_unambiguous` conflicts with `confusing_language` and `multi_interp`",
        "",
        "---",
        "",
        rubrics_md,
    ]
    readme_path = OUT_DIR / "README.md"
    readme_path.write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"Wrote labeling rules to {readme_path}")


if __name__ == "__main__":
    main()
