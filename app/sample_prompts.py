"""Explicitly synthetic SAMPLE prompts; source originals were not provided."""

SAMPLES = [
    {
        "id": "sample-template-complete",
        "name": "SAMPLE — Complete template example",
        "reference_name": "Template",
        "text": "Role: instructor. For bachelor students in Machine Learning, write 5 questions about gradient descent with four options and 4 choices A to D. Mark the correct answer, include an explanation, and order them by difficulty.",
    },
    {
        "id": "sample-faculty-1-style",
        "name": "SAMPLE — Underspecified prompt (Faculty-1-style; not source text)",
        "reference_name": "Faculty-1",
        "text": "An instructor needs a quiz for bachelor students about literature. Write 5 questions.",
    },
    {
        "id": "sample-faculty-2-style",
        "name": "SAMPLE — Minimal prompt (Faculty-2-style; not source text)",
        "reference_name": "Faculty-2",
        "text": "An instructor will assess bachelor students about accounting with 5 questions, labelled A-D.",
    },
    {
        "id": "sample-faculty-3-style",
        "name": "SAMPLE — Contradictory difficulty (Faculty-3-style; not source text)",
        "reference_name": "Faculty-3",
        "text": "Instructor: for master students, write 5 questions about control systems with four options and 4 choices A to D, mark the correct answer and include an explanation. Make the questions very easy and expert-level.",
    },
    {
        "id": "sample-faculty-4-style",
        "name": "SAMPLE — Conflicting answer placement (Faculty-4-style; not source text)",
        "reference_name": "Faculty-4",
        "text": "Instructor: for bachelor students, write 5 questions about anatomy with four options and 4 choices A to D, mark the correct answer and include an explanation. Keep answers hidden until the end, but show the answer after each question.",
    },
    {
        "id": "sample-faculty-5-style",
        "name": "SAMPLE — Ambiguity requested (Faculty-5-style; not source text)",
        "reference_name": "Faculty-5",
        "text": "Instructor: for master students, write 5 questions about marketing with four options and 4 choices A to D. Use confusing language and allow multiple interpretations in the questions.",
    },
]
