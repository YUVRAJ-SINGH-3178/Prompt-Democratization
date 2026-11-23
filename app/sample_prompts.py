"""Explicitly synthetic SAMPLE prompts; source originals were not provided."""

SAMPLES = [
    {
        "id": "sample-template-complete",
        "name": "SAMPLE — Complete template example",
        "reference_name": "Template",
        "text": (
            "Role: university teacher. For bachelor students, create 10 multiple-choice questions "
            "on Normalization. Each question must have four answer choices, with only one correct answer. "
            "Label each answer choice (A, B, C, D), and mark the correct answer. Provide an explanation "
            "for every question. Organize the questions in order of increasing difficulty."
        ),
    },
    {
        "id": "sample-faculty-1-style",
        "name": "SAMPLE — Underspecified prompt (Faculty-1-style; not source text)",
        "reference_name": "Faculty-1",
        "text": (
            "Act as a university teacher and create some multiple-choice questions for students. "
            "Give four options for each question and provide the answers. "
            "Explain the answers wherever necessary."
        ),
    },
    {
        "id": "sample-faculty-2-style",
        "name": "SAMPLE — Minimal prompt (Faculty-2-style; not source text)",
        "reference_name": "Faculty-2",
        "text": (
            "Act as an expert professor and prepare MCQs on computer science. "
            "Provide four choices for every question and mark the right answer. "
            "Include explanations for the answers. Arrange them properly."
        ),
    },
    {
        "id": "sample-faculty-3-style",
        "name": "SAMPLE — Contradictory difficulty (Faculty-3-style; not source text)",
        "reference_name": "Faculty-3",
        "text": (
            "Act as a university teacher for bachelor students. Create 20 MCQs about machine learning. "
            "Give four options and mark the correct answer. Provide an explanation for each option. "
            "Arrange all questions from the easiest to the hardest. Make all questions very easy, "
            "but also test expert-level understanding. Do not include any basic questions."
        ),
    },
    {
        "id": "sample-faculty-4-style",
        "name": "SAMPLE — Conflicting answer placement (Faculty-4-style; not source text)",
        "reference_name": "Faculty-4",
        "text": (
            "Act as an experienced professor and generate 15 MCQs on database management systems for bachelor students. "
            "Each question should have four options labelled A, B, C, and D. Highlight the correct answer, "
            "but do not reveal the correct answer to students. After every question, provide the answer and a detailed explanation."
        ),
    },
    {
        "id": "sample-faculty-5-style",
        "name": "SAMPLE — Ambiguity requested (Faculty-5-style; not source text)",
        "reference_name": "Faculty-5",
        "text": (
            "Act as a university professor and create 25 MCQs on operating systems. "
            "Every question must have four options. Make sure there is only one correct answer. "
            "Use confusing language and include options that can be interpreted in multiple ways. "
            "Give the correct answer and explain why the other options are wrong."
        ),
    },
]
