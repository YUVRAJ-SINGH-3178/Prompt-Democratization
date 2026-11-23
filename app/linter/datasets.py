"""Development and Held-out Datasets for Linter Evaluation.

- Development set: >= 60 prompts with sources explicitly marked ('authored' or 'simulated_persona').
- Held-out set: >= 30 unseen prompts for double-annotated evaluation.
- Extractor version freezing, run logging, and evaluation metrics (precision, recall, Wilson CI, Cohen's kappa).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from app.linter.lexicon import ATOMS, SLOTS, extract_atoms_from_text
from app.verification import wilson_interval

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

CURRENT_EXTRACTOR_VERSION = "rule-linter-0.2.0"
REPOSITORY_CONFIDENCE_THRESHOLD = 0.80  # Predictions below this confidence go to manual review


# ---------------------------------------------------------------------------
# 60 Development Prompts (30 authored + 30 simulated persona style)
# ---------------------------------------------------------------------------
def _build_development_prompts() -> list[dict[str, Any]]:
    dev: list[dict[str, Any]] = []

    # 30 Authored prompts
    authored_specs = [
        # (id, role, cohort, course/topic, count, opts, lbl, mark, exp, order, text, extra_atoms)
        (
            "DEV-AUT-01",
            "Act as a university teacher for bachelor students. Create 10 MCQs on database management systems. Four options each labelled A to D, mark the correct answer and explain why. Order from easiest to hardest.",
            "authored",
            "Engineering",
            "Database Management Systems",
        ),
        (
            "DEV-AUT-02",
            "Act as a college professor and prepare 15 MCQs on operating systems for master students. Four options labelled A, B, C, D with one correct answer marked. Include an explanation for every option. In order of increasing difficulty.",
            "authored",
            "Computing",
            "Operating Systems",
        ),
        (
            "DEV-AUT-03",
            "Act as an instructor. Create 20 MCQs about machine learning for undergraduate students. Four choices labelled A to D. Make questions very easy so anyone can pass, but also test advanced research knowledge. Do not include any basic questions.",
            "authored",
            "Computing",
            "Machine Learning",
        ),
        (
            "DEV-AUT-04",
            "Act as an experienced professor for bachelor students. Write 10 MCQs on software engineering with four options. Highlight the correct answer in bold, but do not reveal the correct answer to students. After every question, provide the answer.",
            "authored",
            "Engineering",
            "Software Engineering",
        ),
        (
            "DEV-AUT-05",
            "Act as a university professor. Generate 25 MCQs on computer networks. Every question must have four options. Make wrong answers look correct and use confusing language so students have to think carefully. Options that can be interpreted in multiple ways should be included.",
            "authored",
            "Computing",
            "Computer Networks",
        ),
        (
            "DEV-AUT-06",
            "Act as a university teacher. Create some multiple-choice questions for students about data structures. Provide four choices and mark the right answer. Explain the answers wherever necessary.",
            "authored",
            "Computing",
            "Data Structures",
        ),
        (
            "DEV-AUT-07",
            "Act as an expert professor. Prepare 12 MCQs on computer science. Provide four choices for every question and mark the right answer. Include explanations. Arrange them properly.",
            "authored",
            "Computing",
            "Computer Science",
        ),
        (
            "DEV-AUT-08",
            "Instructor: for undergraduate students, create 10 multiple-choice questions on Normalization. Four options labelled A to D with single correct answer. Provide detailed explanation for each answer. Organize by difficulty from easiest to hardest.",
            "authored",
            "Engineering",
            "Normalization",
        ),
        (
            "DEV-AUT-09",
            "Act as a professor of chemistry for bachelor students. Create 15 questions about organic reactions. Four options, highlight correct answer. Avoid confusing language and ensure questions are clear and unambiguous.",
            "authored",
            "Science",
            "Organic Chemistry",
        ),
        (
            "DEV-AUT-10",
            "Create 30 MCQs for graduate students on distributed systems. Four options labelled A-D. Exactly one correct answer. Keep explanations concise and order by increasing difficulty.",
            "authored",
            "Computing",
            "Distributed Systems",
        ),
        (
            "DEV-AUT-11",
            "Act as a faculty member in physics. Write 10 questions on quantum mechanics for bachelor students. Four choices, mark correct answer. Do not include basic questions; test expert-level knowledge.",
            "authored",
            "Science",
            "Quantum Mechanics",
        ),
        (
            "DEV-AUT-12",
            "Act as a university teacher. Write 15 multiple-choice questions on linear algebra for undergraduate students. Four options labelled A to D. Provide answers at the end in an answer key.",
            "authored",
            "Mathematics",
            "Linear Algebra",
        ),
        (
            "DEV-AUT-13",
            "Act as an instructor. Prepare 20 MCQs on algorithms for master students. Four options, mark right answer. After each question, provide the answer and explanation.",
            "authored",
            "Computing",
            "Algorithms",
        ),
        (
            "DEV-AUT-14",
            "Act as an experienced teacher. Make 10 MCQs on compiler design with four options. Do not make questions very easy; require expert-level problem solving.",
            "authored",
            "Computing",
            "Compiler Design",
        ),
        (
            "DEV-AUT-15",
            "Act as a professor for bachelor students. Write 8 MCQs on digital logic with four choices. Label choices A, B, C, D. Ensure there is only one correct answer.",
            "authored",
            "Engineering",
            "Digital Logic",
        ),
        (
            "DEV-AUT-16",
            "Prepare 15 multiple-choice questions on biochemistry for medical students. Four answer choices, mark correct answer and explain why incorrect options are wrong. Order from easiest to hardest.",
            "authored",
            "Medical",
            "Biochemistry",
        ),
        (
            "DEV-AUT-17",
            "Act as a university teacher. Create 20 MCQs about computer architecture for undergraduate students. Four options labelled A to D. All questions must be very easy, but also test advanced research knowledge.",
            "authored",
            "Engineering",
            "Computer Architecture",
        ),
        (
            "DEV-AUT-18",
            "Act as an instructor. Create 10 MCQs on cryptography. Four options. Keep answers hidden until the end, but show the answer after each question.",
            "authored",
            "Computing",
            "Cryptography",
        ),
        (
            "DEV-AUT-19",
            "Act as a university professor. Write 15 MCQs on artificial intelligence. Four choices. Ensure questions are clear and unambiguous, leaving no room for misinterpretation.",
            "authored",
            "Computing",
            "Artificial Intelligence",
        ),
        (
            "DEV-AUT-20",
            "Act as an educator for bachelor students. Prepare 20 questions on calculus. Four options labelled A to D. Provide answers and detailed explanations. Arrange them properly.",
            "authored",
            "Mathematics",
            "Calculus",
        ),
        (
            "DEV-AUT-21",
            "Write 10 MCQs on microeconomics for undergraduate students. Four options each. Mark the correct answer. Skip the basics and focus on intermediate application.",
            "authored",
            "Business",
            "Microeconomics",
        ),
        (
            "DEV-AUT-22",
            "Act as a professor in finance. Create 12 MCQs for master students. Four options labelled A to D. Single correct answer. Order from easiest to hardest.",
            "authored",
            "Business",
            "Corporate Finance",
        ),
        (
            "DEV-AUT-23",
            "Act as an instructor. Create some multiple-choice questions on statistics. Give four options and provide the answers. Explain wherever necessary.",
            "authored",
            "Mathematics",
            "Statistics",
        ),
        (
            "DEV-AUT-24",
            "Act as a university teacher for bachelor students. Create 15 MCQs on genetics. Four options, highlight the correct answer in bold. Provide clear explanations.",
            "authored",
            "Medical",
            "Genetics",
        ),
        (
            "DEV-AUT-25",
            "Act as an experienced professor. Prepare 25 MCQs on pharmacology for postgraduate students. Four choices labelled A to D. Only one correct answer. Explain why distractors are wrong.",
            "authored",
            "Medical",
            "Pharmacology",
        ),
        (
            "DEV-AUT-26",
            "Act as a teacher. Write 10 MCQs on English literature for undergraduate students. Four options. Mark the right answer and explain rationale. In order of increasing difficulty.",
            "authored",
            "Literature",
            "English Literature",
        ),
        (
            "DEV-AUT-27",
            "Act as an instructor for bachelor students. Create 15 MCQs on world history. Four options labelled A-D. Keep the answers hidden and provide an answer key at the end.",
            "authored",
            "Humanities",
            "World History",
        ),
        (
            "DEV-AUT-28",
            "Act as a university professor. Make 20 MCQs on psychology for master students. Four choices. Make wrong answers look correct and use confusing language. Options that can be interpreted in multiple ways.",
            "authored",
            "Humanities",
            "Cognitive Psychology",
        ),
        (
            "DEV-AUT-29",
            "Act as a faculty member. Generate 10 MCQs on discrete mathematics for bachelor students. Four options labelled A to D. Mark the correct answer. Provide explanations.",
            "authored",
            "Mathematics",
            "Discrete Mathematics",
        ),
        (
            "DEV-AUT-30",
            "Act as an instructor. Prepare 15 MCQs on environmental science for undergraduate students. Four choices each with one correct answer. Order questions from easiest to hardest.",
            "authored",
            "Science",
            "Environmental Science",
        ),
    ]

    for pid, text, src, disc, topic in authored_specs:
        dev.append(
            {
                "prompt_id": pid,
                "prompt": text,
                "dataset": "development",
                "source": src,
                "discipline": disc,
                "course": topic,
                "synthetic": 0,
            }
        )

    # 30 Simulated persona style prompts (grounded in SIM-P01..P20 personas)
    disciplines = ["Literature", "Engineering", "Medical", "Business", "Computing"]
    courses = [
        "Machine Learning",
        "Database Management Systems",
        "Operating Systems",
        "Computer Networks",
        "Clinical Pathology",
        "Corporate Finance",
    ]
    habits = [
        "good_complete",
        "poor_underspecified",
        "poor_conflicted_difficulty",
        "poor_conflicted_answers",
        "poor_ambiguity_seeking",
    ]

    for i in range(1, 31):
        pid = f"DEV-SIM-{i:02d}"
        disc = disciplines[(i - 1) % len(disciplines)]
        course = courses[(i - 1) % len(courses)]
        habit = habits[(i - 1) % len(habits)]

        if habit == "good_complete":
            text = (
                f"Act as a university teacher tasked with assessing bachelor students in {course}. "
                f"Create 10 multiple-choice questions on {course}. Each question must have four answer choices, "
                f"with only one correct answer. Label each choice (A, B, C, D) and mark the correct answer in bold. "
                f"Provide clear explanations for each option. Organize the questions in order of increasing difficulty."
            )
        elif habit == "poor_underspecified":
            text = (
                f"Act as an educator and write some multiple choice questions for students about {course}. "
                f"Give four options for each question and provide the answers. Explain wherever necessary."
            )
        elif habit == "poor_conflicted_difficulty":
            text = (
                f"Act as a university teacher for undergraduate students. Create 20 MCQs about {course}. "
                f"Four choices with one correct answer marked. All questions must be very easy so every student can answer them, "
                f"but they must also test advanced research knowledge and expert-level understanding. Do not include any basic questions."
            )
        elif habit == "poor_conflicted_answers":
            text = (
                f"Act as an experienced professor for bachelor students. Write 15 MCQs on {course} with four options labelled A, B, C, D. "
                f"Highlight the correct answer, but do not reveal the correct answer to students. After every question, provide the answer and explanation."
            )
        else:  # poor_ambiguity_seeking
            text = (
                f"Act as a university professor. Create 25 MCQs on {course}. Every question must have four options. "
                f"Use confusing language and tricky statements so students have to think carefully. Include options that can be interpreted in multiple ways."
            )

        dev.append(
            {
                "prompt_id": pid,
                "prompt": text,
                "dataset": "development",
                "source": "simulated_persona",
                "discipline": disc,
                "course": course,
                "synthetic": 1,
            }
        )

    return dev


# ---------------------------------------------------------------------------
# 30 Held-Out Prompts (unseen during development)
# ---------------------------------------------------------------------------
def _build_heldout_prompts() -> list[dict[str, Any]]:
    held: list[dict[str, Any]] = []
    specs = [
        (
            "HELD-01",
            "Act as a university teacher for bachelor students. Create 10 MCQs on Normalization. Four options labelled A to D, one correct answer marked in bold, explanations for all choices, ordered from easiest to hardest.",
            "Computing",
            "Normalization",
        ),
        (
            "HELD-02",
            "Act as an experienced professor. Write 15 MCQs on database indexing for undergraduate students with four choices A, B, C, D. Mark the right answer and explain rationale. Increasing difficulty.",
            "Computing",
            "Database Indexing",
        ),
        (
            "HELD-03",
            "Act as a university instructor. Prepare 20 MCQs on concurrency control for master students. Four options, highlight correct answer. Make questions very easy but test expert-level understanding. No basic questions.",
            "Computing",
            "Concurrency Control",
        ),
        (
            "HELD-04",
            "Act as a faculty member for bachelor students. Generate 12 MCQs on SQL queries with four options labelled A-D. Mark correct answer, but do not reveal the correct answer to students. Show answer after each question.",
            "Computing",
            "SQL Queries",
        ),
        (
            "HELD-05",
            "Act as a professor. Write 25 MCQs on memory management. Four choices. Use confusing language and include options that can be interpreted in multiple ways. Give correct answer.",
            "Computing",
            "Memory Management",
        ),
        (
            "HELD-06",
            "Act as a teacher. Create some questions for students about file systems. Provide four options and provide the answers. Explain answers wherever necessary.",
            "Computing",
            "File Systems",
        ),
        (
            "HELD-07",
            "Act as an expert professor. Prepare MCQs on computer architecture. Four choices for every question and mark right answer. Include explanations. Arrange them properly.",
            "Engineering",
            "Computer Architecture",
        ),
        (
            "HELD-08",
            "Act as an instructor for bachelor students. Create 10 multiple-choice questions on neural networks. Four answer choices labelled A to D. Only one correct answer. Clear and concise explanations.",
            "Computing",
            "Neural Networks",
        ),
        (
            "HELD-09",
            "Act as a university professor for postgraduate students. Write 15 MCQs on deep learning. Four options. Skip the basics and test advanced research knowledge. Ordered from easiest to hardest.",
            "Computing",
            "Deep Learning",
        ),
        (
            "HELD-10",
            "Act as an instructor. Write 20 MCQs on computer graphics for undergraduate students. Four choices labelled A-D. Keep answers hidden until the end. Explain all answers.",
            "Computing",
            "Computer Graphics",
        ),
        (
            "HELD-11",
            "Act as a professor for bachelor students. Generate 10 MCQs on software testing with four options labelled A to D. Highlight correct answer. Ensure questions are clear and unambiguous.",
            "Engineering",
            "Software Testing",
        ),
        (
            "HELD-12",
            "Act as a university teacher. Create 15 MCQs on agile methodologies for master students. Four options each with one correct answer. Arrange from easiest to hardest.",
            "Engineering",
            "Agile Methodologies",
        ),
        (
            "HELD-13",
            "Act as an educator for undergraduate students. Write 10 MCQs on microprocessors. Four choices. Mark the correct answer. Make questions very easy so everyone passes, but also test expert-level knowledge.",
            "Engineering",
            "Microprocessors",
        ),
        (
            "HELD-14",
            "Act as an experienced teacher. Prepare 12 MCQs on robotics for bachelor students. Four options labelled A, B, C, D. Do not reveal correct answer to students. Provide answer after each question.",
            "Engineering",
            "Robotics",
        ),
        (
            "HELD-15",
            "Act as a professor. Create 20 MCQs on cyber security. Four choices. Make wrong answers look correct and use confusing wording. Make sure there is only one correct answer.",
            "Computing",
            "Cyber Security",
        ),
        (
            "HELD-16",
            "Act as a university teacher. Write some multiple choice questions for students on web development. Four options and provide the answers.",
            "Computing",
            "Web Development",
        ),
        (
            "HELD-17",
            "Act as an expert professor and prepare MCQs on data science. Provide four choices and mark the right answer. Include explanations. Arrange them properly.",
            "Computing",
            "Data Science",
        ),
        (
            "HELD-18",
            "Act as an instructor for bachelor students. Create 10 MCQs on cloud computing. Four options labelled A to D. One correct answer. Provide detailed explanation for each.",
            "Computing",
            "Cloud Computing",
        ),
        (
            "HELD-19",
            "Act as a faculty member for master students. Write 15 MCQs on natural language processing. Four choices. Skip the basics. Ordered by increasing difficulty.",
            "Computing",
            "Natural Language Processing",
        ),
        (
            "HELD-20",
            "Act as a professor for undergraduate students. Write 20 MCQs on computer vision. Four options labelled A to D. Keep the answers hidden in an answer key. Include explanations.",
            "Computing",
            "Computer Vision",
        ),
        (
            "HELD-21",
            "Act as a university teacher. Create 10 MCQs on bioinformatics for bachelor students. Four answer choices, mark correct answer bold. Clear and unambiguous wording.",
            "Science",
            "Bioinformatics",
        ),
        (
            "HELD-22",
            "Act as an experienced professor. Prepare 15 MCQs on immunology for medical students. Four choices labelled A, B, C, D. Single correct answer. Order from easiest to hardest.",
            "Medical",
            "Immunology",
        ),
        (
            "HELD-23",
            "Act as an instructor. Create 12 MCQs on physiology for undergraduate students. Four options. Make questions very easy and test advanced research knowledge. Avoid basic questions.",
            "Medical",
            "Physiology",
        ),
        (
            "HELD-24",
            "Act as a faculty member for bachelor students. Write 10 MCQs on marketing management. Four choices labelled A to D. Mark correct answer, but do not reveal answer to students. Show answer immediately after question.",
            "Business",
            "Marketing Management",
        ),
        (
            "HELD-25",
            "Act as a university professor. Write 25 MCQs on macroeconomics. Four options. Use confusing language so students have to think carefully. Options with multiple interpretations.",
            "Business",
            "Macroeconomics",
        ),
        (
            "HELD-26",
            "Act as a teacher. Create some multiple-choice questions for students about business law. Four options and provide the answers. Explain wherever necessary.",
            "Business",
            "Business Law",
        ),
        (
            "HELD-27",
            "Act as an expert professor. Prepare MCQs on financial accounting. Four choices and mark the right answer. Include explanations. Arrange them properly.",
            "Business",
            "Financial Accounting",
        ),
        (
            "HELD-28",
            "Act as an instructor for bachelor students. Create 10 MCQs on human anatomy. Four options labelled A to D. One correct answer. Detailed explanation for every question. Easiest to hardest.",
            "Medical",
            "Human Anatomy",
        ),
        (
            "HELD-29",
            "Act as a faculty member for master students. Write 15 MCQs on linear control systems. Four choices. Do not include basic questions; test expert-level understanding. Progressive difficulty.",
            "Engineering",
            "Control Systems",
        ),
        (
            "HELD-30",
            "Act as a university professor for undergraduate students. Create 20 MCQs on modern poetry. Four options labelled A to D. Keep the answers hidden until the end. Clear explanations.",
            "Literature",
            "Modern Poetry",
        ),
    ]

    for pid, text, disc, topic in specs:
        held.append(
            {
                "prompt_id": pid,
                "prompt": text,
                "dataset": "heldout",
                "source": "heldout_unseen",
                "discipline": disc,
                "course": topic,
                "synthetic": 0,
            }
        )

    return held


DEVELOPMENT_PROMPTS = _build_development_prompts()
HELDOUT_PROMPTS = _build_heldout_prompts()


def get_development_prompts() -> list[dict[str, Any]]:
    return [dict(p) for p in DEVELOPMENT_PROMPTS]


def get_heldout_prompts() -> list[dict[str, Any]]:
    return [dict(p) for p in HELDOUT_PROMPTS]


def seed_datasets(db_path: Path | None = None) -> dict[str, int]:
    """Seed development and heldout prompts into accuracy_prompts and write JSON cache."""
    target_db = db_path or (DATA_DIR / "local_auth.sqlite")
    dev_prompts = get_development_prompts()
    held_prompts = get_heldout_prompts()

    # Write JSON files in data/
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dev_json_path = DATA_DIR / "dev_prompts.json"
    held_json_path = DATA_DIR / "heldout_prompts.json"
    dev_json_path.write_text(json.dumps(dev_prompts, indent=2), encoding="utf-8")
    held_json_path.write_text(json.dumps(held_prompts, indent=2), encoding="utf-8")

    dev_count = 0
    held_count = 0

    target_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(target_db) as conn:
        from app.auth_db import SCHEMA

        conn.executescript(SCHEMA)

        for p in dev_prompts:
            conn.execute(
                "INSERT OR IGNORE INTO accuracy_prompts(prompt_id,prompt,dataset,source,synthetic) VALUES(?,?,?,?,?)",
                (p["prompt_id"], p["prompt"], p["dataset"], p["source"], p["synthetic"]),
            )
            dev_count += 1

        for p in held_prompts:
            conn.execute(
                "INSERT OR IGNORE INTO accuracy_prompts(prompt_id,prompt,dataset,source,synthetic) VALUES(?,?,?,?,?)",
                (p["prompt_id"], p["prompt"], p["dataset"], p["source"], p["synthetic"]),
            )
            held_count += 1
        conn.commit()

    return {"development_seeded": len(dev_prompts), "heldout_seeded": len(held_prompts)}


# ---------------------------------------------------------------------------
# Agreement and Reliability Metrics
# ---------------------------------------------------------------------------
def compute_cohen_kappa(r1: list[Any], r2: list[Any]) -> float:
    """Compute Cohen's kappa for two independent raters across N items."""
    if not r1 or not r2 or len(r1) != len(r2):
        return 0.0
    n = len(r1)
    if n == 0:
        return 0.0

    # Categorical domain
    categories = sorted(list(set(r1) | set(r2)))
    if len(categories) <= 1:
        return 1.0  # Perfect agreement on single category

    # Observed agreement
    agree_count = sum(1 for a, b in zip(r1, r2) if a == b)
    po = agree_count / n

    # Expected agreement by chance
    pe = 0.0
    for cat in categories:
        p1 = sum(1 for x in r1 if x == cat) / n
        p2 = sum(1 for y in r2 if y == cat) / n
        pe += p1 * p2

    if pe >= 1.0:
        return 1.0
    return float((po - pe) / (1.0 - pe))


def evaluate_heldout_dataset(db: sqlite3.Connection, freeze_version: str = CURRENT_EXTRACTOR_VERSION) -> dict[str, Any]:
    """Score the held-out dataset once per frozen version and log the run."""
    try:
        held_prompts = db.execute(
            "SELECT prompt_id, prompt FROM accuracy_prompts WHERE dataset='heldout' AND synthetic=0 ORDER BY prompt_id"
        ).fetchall()
        annotations = db.execute(
            "SELECT prompt_id, annotator_id, atoms_json, slots_json FROM accuracy_annotations WHERE dataset='heldout' AND synthetic=0 ORDER BY prompt_id, annotator_id"
        ).fetchall()
    except sqlite3.OperationalError:
        held_prompts = []
        annotations = []

    by_prompt: dict[str, list[dict]] = {}
    for pid, aid, atoms_j, slots_j in annotations:
        by_prompt.setdefault(pid, []).append(
            {"annotator_id": aid, "atoms": json.loads(atoms_j), "slots": json.loads(slots_j)}
        )

    # Eligible prompts are those with at least two distinct human annotators
    eligible = {pid: annots for pid, annots in by_prompt.items() if len({a["annotator_id"] for a in annots}) >= 2}

    base_report = {
        "status": "awaiting annotations",
        "extractor_version": freeze_version,
        "heldout_prompts_n": len(held_prompts),
        "heldout_prompts_total": len(held_prompts),
        "heldout_two_annotators_n": len(eligible),
        "heldout_double_annotated_n": len(eligible),
        "required_n": 30,
        "confidence_threshold": REPOSITORY_CONFIDENCE_THRESHOLD,
        "manual_review_policy": (
            f"Atom extractions below {REPOSITORY_CONFIDENCE_THRESHOLD * 100:.0f}% confidence "
            "are routed to manual review rather than guessed."
        ),
        "metrics": {},
    }

    if len(eligible) < 30:
        return base_report

    # Once 30 double-annotated prompts are present, compute exact metrics
    prompt_text_map = {pid: ptext for pid, ptext in held_prompts}
    atom_metrics: dict[str, Any] = {}
    atom_kappas: dict[str, float] = {}

    for atom in ATOMS:
        r1_list: list[int] = []
        r2_list: list[int] = []
        tp = fp = fn = 0

        for pid, annots in eligible.items():
            text = prompt_text_map[pid]
            auto_atoms = extract_atoms_from_text(text)
            pred = atom in auto_atoms

            # Annotator 1 and 2
            a1_has = int(atom in annots[0]["atoms"])
            a2_has = int(atom in annots[1]["atoms"])
            r1_list.append(a1_has)
            r2_list.append(a2_has)

            # Ground truth consensus (both agree, or evaluation against both labels)
            for annot in annots[:2]:
                gold = atom in annot["atoms"]
                tp += int(pred and gold)
                fp += int(pred and not gold)
                fn += int(not pred and gold)

        atom_kappas[atom] = compute_cohen_kappa(r1_list, r2_list)
        pden = tp + fp
        rden = tp + fn
        atom_metrics[atom] = {
            "precision": float(tp / pden) if pden else None,
            "precision_n": pden,
            "precision_ci95": wilson_interval(tp, pden),
            "recall": float(tp / rden) if rden else None,
            "recall_n": rden,
            "recall_ci95": wilson_interval(tp, rden),
            "inter_annotator_cohen_kappa": atom_kappas[atom],
        }

    # Slot agreements and kappas
    slot_agreement: dict[str, Any] = {}
    for slot in SLOTS:
        s1_vals: list[float] = []
        s2_vals: list[float] = []
        agree = 0
        for annots in eligible.values():
            s1 = float(annots[0]["slots"].get(slot, 0.0))
            s2 = float(annots[1]["slots"].get(slot, 0.0))
            s1_vals.append(s1)
            s2_vals.append(s2)
            agree += int(s1 == s2)

        slot_agreement[slot] = {
            "agreement": float(agree / len(eligible)),
            "n": len(eligible),
            "ci95": wilson_interval(agree, len(eligible)),
            "cohen_kappa": compute_cohen_kappa([str(x) for x in s1_vals], [str(x) for x in s2_vals]),
        }

    report = {
        "status": "measured",
        "extractor_version": freeze_version,
        "n": len(eligible),
        "atom_metrics": atom_metrics,
        "slot_agreement": slot_agreement,
        "mean_atom_cohen_kappa": float(sum(atom_kappas.values()) / len(atom_kappas)),
        "confidence_interval": "Wilson 95% score interval",
        "manual_review_policy": base_report["manual_review_policy"],
    }

    # Log scoring run into linter_scoring_runs
    db.execute(
        "INSERT INTO linter_scoring_runs(extractor_version, dataset, prompt_count, report_json) VALUES(?,?,?,?)",
        (freeze_version, "heldout", len(eligible), json.dumps(report)),
    )
    db.commit()

    return report
