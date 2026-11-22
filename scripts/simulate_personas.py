"""Run the isolated 20-persona pilot through an OpenAI-compatible chat API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.simulated_database import audit, connect  # noqa: E402
from app.study import check_response  # noqa: E402
from app.linter.core import ATOMS  # noqa: E402

SEED = 731_204
TEMPERATURE = 0.2
CONDITIONS = (
    "free_form_faculty",
    "filled_template",
    "filled_template_linter_repair",
    "lagrange_routed_template",
    "gaussian_process_routed_template",
)
SLOTS = ("role", "audience", "topic", "count", "n_options", "labeling", "marking", "explanation", "ordering")
SLOT_DEFINITIONS = {
    "role": "A teacher or instructor role is named.",
    "audience": "Learner level or audience is identified.",
    "topic": "Course topic or subject is specific.",
    "count": "Number of questions is specified.",
    "n_options": "Number of choices per question is specified.",
    "labeling": "Choice labels or format are specified.",
    "marking": "Correct answer or answer key handling is specified.",
    "explanation": "Explanation or rationale is requested.",
    "ordering": "Ordering, progression, or difficulty is specified.",
}
TEMPLATES = [
    {
        "template_id": "SIM-T01",
        "name": "Complete assessment brief",
        "task_type": "quiz",
        "prompt": "As a course instructor for #course, create #count four-option A-D MCQs for #audience on #topic. Make one answer correct, mark it, explain the rationale, and order questions from recall to application.",
    },
    {
        "template_id": "SIM-T02",
        "name": "Scenario-based quiz",
        "task_type": "quiz",
        "prompt": "For #course students, write #count A-D MCQs about #topic using short application scenarios. Mark exactly one correct option and include a concise explanation for each.",
    },
    {
        "template_id": "SIM-T03",
        "name": "Concise recall check",
        "task_type": "quiz",
        "prompt": "Write #count recall MCQs on #topic for #audience in #course. Give four options labelled A-D, identify the answer, and add a brief rationale.",
    },
    {
        "template_id": "SIM-T04",
        "name": "Progressive difficulty quiz",
        "task_type": "quiz",
        "prompt": "Create #count four-option MCQs for #audience in #course about #topic. Progress from recall through application, mark one answer per question, and provide a short rationale.",
    },
    {
        "template_id": "SIM-T05",
        "name": "Case interpretation assessment",
        "task_type": "quiz",
        "prompt": "Write #count case-based MCQs for #course students about #topic. Use A-D labels, one defensible correct answer, plausible distractors, and a brief explanation.",
    },
]


def load_personas() -> list[dict]:
    path = ROOT / "data" / "simulated" / "personas.yaml"
    if not path.exists():
        raise FileNotFoundError("Run scripts/generate_personas.py first")
    return json.loads(path.read_text(encoding="utf-8"))["personas"]


def load_local_env() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def setup() -> None:
    with connect() as db:
        for p in load_personas():
            db.execute(
                "INSERT OR IGNORE INTO personas(persona_id,card_json,synthetic,source) VALUES(?,?,1,'sim_persona')",
                (p["id"], json.dumps(p, ensure_ascii=False)),
            )
        for t in TEMPLATES:
            db.execute(
                "INSERT OR IGNORE INTO templates(template_id,name,task_type,prompt,synthetic,source,persona_id) VALUES(?,?,?, ?,1,'sim_persona','SIM-P01')",
                (t["template_id"], t["name"], t["task_type"], t["prompt"]),
            )


def parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def call_model(persona_id: str, task: str, user_task: str, model: str, endpoint: str, key: str, call_seed: int) -> Any:
    request_obj = {
        "model": model,
        "temperature": TEMPERATURE,
        "seed": call_seed,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "You are a simulated university instructor for software testing. This is fictional simulation, not real faculty data. Follow the user's task and return only valid JSON.",
            },
            {"role": "user", "content": user_task},
        ],
    }
    logged_request = {
        "url": endpoint,
        "method": "POST",
        "headers": {"Content-Type": "application/json", "Authorization": "Bearer [REDACTED]"},
        "body": request_obj,
    }
    raw_text = None
    status = "ok"
    try:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(request_obj).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            raw_text = response.read().decode("utf-8")
        envelope = json.loads(raw_text)
        answer = envelope["choices"][0]["message"]["content"]
        payload = parse_json(answer)
        response_json, response_text = json.dumps(envelope, ensure_ascii=False), answer
    except urllib.error.HTTPError as exc:
        raw_text = exc.read().decode("utf-8", errors="replace")
        response_json, response_text = None, raw_text
        _log_call(persona_id, task, call_seed, logged_request, response_json, response_text, "error")
        raise RuntimeError(f"{task} failed for {persona_id}: HTTP {exc.code} {response_text[:500]}") from exc
    except Exception as exc:
        status = "error"
        response_json, response_text = None, raw_text or f"{type(exc).__name__}: {exc}"
        _log_call(persona_id, task, call_seed, logged_request, response_json, response_text, status)
        raise RuntimeError(f"{task} failed for {persona_id}: {response_text[:500]}") from exc
    _log_call(persona_id, task, call_seed, logged_request, response_json, response_text, status)
    return payload


def _log_call(
    persona_id: str,
    task: str,
    call_seed: int,
    request_obj: dict,
    response_json: str | None,
    response_text: str,
    status: str,
) -> None:
    with connect() as db:
        db.execute(
            "INSERT INTO model_calls(persona_id,task,seed,temperature,request_json,response_json,response_text,status,synthetic,source) VALUES(?,?,?,?,?,?,?,?,1,'sim_persona')",
            (
                persona_id,
                task,
                call_seed,
                TEMPERATURE,
                json.dumps(request_obj, ensure_ascii=False),
                response_json,
                response_text,
                status,
            ),
        )


def persona_prompt(card: dict) -> str:
    return json.dumps(card, ensure_ascii=False, sort_keys=True)


def persist_prompt(persona_id: str, idx: int, prompt: str) -> str:
    pid = f"{persona_id}-P{idx:02d}"
    from app.linter.core import evaluate

    lint = evaluate(prompt)
    result = {
        "atoms": lint.atoms,
        "completeness": lint.completeness,
        "energy": lint.energy,
        "repair": lint.repair,
        "slot_scores": lint.slot_scores,
    }
    with connect() as db:
        db.execute(
            "INSERT OR REPLACE INTO prompts(prompt_id,persona_id,prompt,linter_json,synthetic,source) VALUES(?,?,?,?,1,'sim_persona')",
            (pid, persona_id, prompt, json.dumps(result)),
        )
    return pid


def routed_template(card: dict, method: str) -> dict:
    """Choose a study template from T2 data while holding this persona out."""
    import numpy as np
    from scipy.interpolate import BarycentricInterpolator
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

    with connect() as db:
        all_cards = [json.loads(r[0]) for r in db.execute("SELECT card_json FROM personas ORDER BY persona_id")]
        ratings = {
            (pid, tid): float(score)
            for pid, tid, score in db.execute(
                "SELECT persona_id,template_id,AVG(rating) FROM ratings GROUP BY persona_id,template_id"
            )
        }
    templates = [t for t in TEMPLATES if any((p["id"], t["template_id"]) in ratings for p in all_cards)]
    training = [
        p for p in all_cards if p["id"] != card["id"] and all((p["id"], t["template_id"]) in ratings for t in templates)
    ]
    if not templates:
        return TEMPLATES[0]
    if method == "lagrange" and len(training) >= 2:
        estimates = {}
        for template in templates:
            grouped: dict[int, list[float]] = {}
            for persona in training:
                grouped.setdefault(persona["years_teaching"], []).append(
                    ratings[(persona["id"], template["template_id"])]
                )
            x = np.array(sorted(grouped), dtype=float)
            y = np.array([np.mean(grouped[int(v)]) for v in x], dtype=float)
            estimates[template["template_id"]] = (
                float(BarycentricInterpolator(x, y)(card["years_teaching"])) if len(x) > 1 else float(np.mean(y))
            )
    elif method == "gaussian_process" and len(training) >= 4:
        X, y = [], []
        for persona in training:
            for i, template in enumerate(templates):
                X.append(
                    [
                        persona["years_teaching"] / 40,
                        i / max(1, len(templates) - 1),
                        float(persona["discipline"] == card["discipline"]),
                        float(persona["assessment_style"] == card["assessment_style"]),
                    ]
                )
                y.append(ratings[(persona["id"], template["template_id"])])
        model = GaussianProcessRegressor(
            kernel=ConstantKernel(1) * RBF() + WhiteKernel(0.3),
            normalize_y=True,
            random_state=SEED,
            n_restarts_optimizer=0,
        ).fit(np.asarray(X), np.asarray(y))
        q = np.array(
            [[card["years_teaching"] / 40, i / max(1, len(templates) - 1), 1.0, 1.0] for i in range(len(templates))]
        )
        estimates = {t["template_id"]: float(score) for t, score in zip(templates, model.predict(q))}
    else:
        estimates = (
            {
                t["template_id"]: float(np.mean([ratings[(p["id"], t["template_id"])] for p in training]))
                for t in templates
            }
            if training
            else {t["template_id"]: 0.0 for t in templates}
        )
    return max(templates, key=lambda t: (estimates.get(t["template_id"], -float("inf")), t["template_id"]))


def filled_template(template: dict, card: dict) -> str:
    return (
        template["prompt"]
        .replace("#course", card["course"])
        .replace("#count", "5")
        .replace("#audience", "undergraduate students")
        .replace("#topic", card["course"])
    )


def condition_task(card: dict, condition: str, default_template: dict, own_prompts: dict[str, list[str]]) -> str:
    from app.linter.core import evaluate

    if condition == "free_form_faculty":
        prompt = own_prompts[card["id"]][0]
        return f"PERSONA CARD:\n{persona_prompt(card)}\n\nTASK: Generate exactly five MCQs for this course following this free-form prompt you wrote earlier:\n{prompt}"
    if condition == "filled_template":
        template = default_template
        prompt = filled_template(template, card)
        return f"PERSONA CARD:\n{persona_prompt(card)}\n\nTASK: Generate exactly five MCQs using this filled template (condition=filled_template):\n{prompt}"
    if condition == "filled_template_linter_repair":
        template = default_template
        prompt = filled_template(template, card)
        lint = evaluate(prompt)
        repaired = prompt
        if lint.repair:
            repaired += " Apply these minimum repairs: " + ", ".join(lint.repair)
        return f"PERSONA CARD:\n{persona_prompt(card)}\n\nTASK: Generate exactly five MCQs from this template after applying rule-linter minimum repairs. Linter energy={lint.energy}; repairs={json.dumps(lint.repair)}; completeness={lint.completeness:.3f}.\n{repaired}"
    method = "lagrange" if condition == "lagrange_routed_template" else "gaussian_process"
    template = routed_template(card, method)
    prompt = filled_template(template, card)
    return f"PERSONA CARD:\n{persona_prompt(card)}\n\nTASK: Generate exactly five MCQs using the {method} routed synthetic template {template['template_id']} selected from other simulated raters' T2 data (this persona was held out).\n{prompt}"


def run() -> dict:
    key = os.environ.get("STUDY_API_KEY") or os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("STUDY_MODEL")
    endpoint = os.environ.get("STUDY_API_BASE", "https://api.openai.com/v1/chat/completions")
    if not key or not model:
        raise RuntimeError(
            "Missing model configuration. Set STUDY_API_KEY, STUDY_MODEL, and optional STUDY_API_BASE in .env; do not store credentials in this report."
        )
    setup()
    personas = load_personas()
    call_number = 0

    def invoke(pid: str, task: str, prompt: str) -> Any:
        nonlocal call_number
        call_number += 1
        return call_model(pid, task, prompt, model, endpoint, key, SEED + call_number)

    # T1: each author receives only their own card and one task.
    all_prompts = []
    own_prompts: dict[str, list[str]] = {}
    for card in personas:
        request = f'PERSONA CARD:\n{persona_prompt(card)}\n\nTASK: Write exactly three distinct free-form prompts that ask an LLM to generate MCQs for your own course. Follow your stated prompt-writing habit. Do not use a template. Return JSON {{"prompts":[string,string,string]}}.'
        result = invoke(card["id"], "T1_write_prompts", request)
        own_prompts[card["id"]] = []
        for i, prompt in enumerate(result["prompts"], 1):
            pid = persist_prompt(card["id"], i, str(prompt))
            all_prompts.append({"prompt_id": pid, "text": str(prompt)})
            own_prompts[card["id"]].append(str(prompt))

    # T2: each repetition is a separate call; personas do not receive previous ratings.
    for card in personas:
        for repetition in range(1, 4):
            task = {
                "persona_card": card,
                "templates": TEMPLATES,
                "task": f"Rate every template for this course from 1 to 5 for suitability. Give one-line reason per rating. This is repetition {repetition}; do not try to reproduce a previous response.",
                "return_schema": {"ratings": [{"template_id": "SIM-T01", "rating": 1, "reason": "one line"}]},
            }
            result = invoke(card["id"], "T2_rate_templates", json.dumps(task, ensure_ascii=False))
            expected_ids = {t["template_id"] for t in TEMPLATES}
            returned_ids = [row.get("template_id") for row in result["ratings"]]
            if len(returned_ids) != len(set(returned_ids)) or set(returned_ids) != expected_ids:
                raise ValueError(f"T2 must rate every template exactly once for {card['id']}")
            with connect() as db:
                for row in result["ratings"]:
                    db.execute(
                        "INSERT OR REPLACE INTO ratings(persona_id,template_id,repetition,rating,reason,synthetic,source) VALUES(?,?,?,?,?,1,'sim_persona')",
                        (card["id"], row["template_id"], repetition, row["rating"], row["reason"]),
                    )

    # Use a seeded, author-blind common subset for T3. Content is the stimulus only;
    # no author identity, linter result, or other ratings are sent.
    import random

    rng = random.Random(SEED)
    sampled = rng.sample(all_prompts, 30)
    common = [{"prompt_id": f"SIM-I{idx:02d}", "text": row["text"]} for idx, row in enumerate(sampled, 1)]
    blind_to_source = {f"SIM-I{idx:02d}": row["prompt_id"] for idx, row in enumerate(sampled, 1)}
    for card in personas:
        task = {
            "persona_card": card,
            "task": "Independently label each prompt vague, contradictory, both, or fine. List detected atoms from the supplied linter vocabulary and score every defined slot 0, 0.5, or 1. Return JSON annotations array with prompt_id,label,atoms,slots. Do not infer factual correctness. No linter outputs are provided.",
            "items": common,
            "slot_definitions": SLOT_DEFINITIONS,
            "allowed_atoms": list(ATOMS),
        }
        result = invoke(card["id"], "T3_blind_annotations", json.dumps(task, ensure_ascii=False))
        expected_items = {item["prompt_id"] for item in common}
        returned_items = [item.get("prompt_id") for item in result["annotations"]]
        if len(returned_items) != len(set(returned_items)) or set(returned_items) != expected_items:
            raise ValueError(f"T3 must annotate every common item exactly once for {card['id']}")
        with connect() as db:
            for item in result["annotations"]:
                if item["label"] not in ("vague", "contradictory", "both", "fine"):
                    raise ValueError(f"Invalid T3 label from {card['id']}")
                slots = item["slots"]
                if set(slots) != set(SLOTS) or any(v not in (0, 0.5, 1) for v in slots.values()):
                    raise ValueError(f"Invalid slot scores from {card['id']}")
                if any(atom not in ATOMS for atom in item["atoms"]):
                    raise ValueError(f"Invalid linter atom from {card['id']}")
                db.execute(
                    "INSERT OR REPLACE INTO annotations(persona_id,prompt_id,label,atoms_json,slots_json,synthetic,source) VALUES(?,?,?,?,?,1,'sim_persona')",
                    (
                        card["id"],
                        blind_to_source[item["prompt_id"]],
                        item["label"],
                        json.dumps(item["atoms"]),
                        json.dumps(slots),
                    ),
                )

    # T4: generation and blinded scoring use separate requests and different personas.
    for i, card in enumerate(personas):
        for cidx, condition in enumerate(CONDITIONS):
            template = TEMPLATES[(i + cidx) % len(TEMPLATES)]
            task_prompt = condition_task(card, condition, template, own_prompts)
            generated = invoke(
                card["id"],
                f"T4_generate_{condition}",
                f"{task_prompt}\n\nReturn JSON with questions (stem, options A-D with exactly one correct=true, explanation, bloom_level).",
            )
            questions = generated.get("questions", [])
            checks = check_response(questions, 5)
            scorer = personas[(i + 1) % len(personas)]
            score_task = {
                "persona_card": scorer,
                "task": "Blindly score each question's distractor quality 1-5, Bloom level (remember, understand, apply, analyze, evaluate, create), and one-line reason. You cannot verify factual answer correctness; set correctness_judgement exactly to unverified for every question.",
                "questions": questions,
            }
            scores = invoke(scorer["id"], f"T4_score_{condition}", json.dumps(score_task, ensure_ascii=False))
            if len(scores.get("scores", [])) != len(questions):
                raise ValueError(f"T4 scorer returned a different question count for {scorer['id']}")
            blooms = {"remember", "understand", "apply", "analyze", "evaluate", "create"}
            if any(
                not 1 <= int(score["distractor_quality"]) <= 5 or score["bloom_level"] not in blooms
                for score in scores["scores"]
            ):
                raise ValueError(f"T4 scorer returned an invalid score for {scorer['id']}")
            run_id = f"{card['id']}-{condition}"
            with connect() as db:
                # Store full generation and review in a synthetic run row.
                db.execute(
                    "INSERT OR REPLACE INTO study_runs(run_id,persona_id,scorer_persona_id,condition_name,request_json,response_json,checks_json,synthetic,source) VALUES(?,?,?,?,?,?,?,1,'sim_persona')",
                    (
                        run_id,
                        card["id"],
                        scorer["id"],
                        condition,
                        json.dumps({"prompt": task_prompt}),
                        json.dumps(generated, ensure_ascii=False),
                        json.dumps(checks),
                    ),
                )
                for qi, score in enumerate(scores["scores"], 1):
                    db.execute(
                        "INSERT INTO question_scores(run_id,persona_id,question_index,distractor_quality,bloom_level,correctness_judgement,reason,synthetic,source) VALUES(?,?,?,?,?,?,?,1,'sim_persona')",
                        (
                            run_id,
                            scorer["id"],
                            qi,
                            score["distractor_quality"],
                            score["bloom_level"],
                            "unverified",
                            score.get("reason", ""),
                        ),
                    )
    from app.simulated_analysis import analyze

    with connect() as db:
        summary = {
            "status": "complete",
            "label": "SIMULATED",
            "synthetic": True,
            "source": "sim_persona",
            "persona_ids": [p["id"] for p in personas],
            "model": model,
            "seed": SEED,
            "temperature": TEMPERATURE,
            "calls": call_number,
            "audit": audit(db),
            "analysis": analyze(db),
            "source_fixture_status": "Original five faculty prompt texts and source template are unavailable; these items were not invented.",
        }
    out = ROOT / "data" / "simulated"
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(summary["analysis"], indent=2) + "\n", encoding="utf-8")
    (out / "run_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true", help="Print planned calls without contacting a model")
    args = parser.parse_args()
    setup()
    if args.plan_only:
        result = {
            "status": "plan_only",
            "personas": len(load_personas()),
            "templates": len(TEMPLATES),
            "expected_model_calls": 20 + 60 + 20 + 100 + 100,
            "seed": SEED,
            "temperature": TEMPERATURE,
            "model_configured": bool(
                (os.environ.get("STUDY_API_KEY") or os.environ.get("OPENAI_API_KEY")) and os.environ.get("STUDY_MODEL")
            ),
            "database": "data/simulated/personas.sqlite",
            "limitations": [
                "Original five faculty prompts and source template are unavailable.",
                "Thirty T1 prompts will be sampled anonymously for common-item annotation.",
                "The original repository has zero templates; five clearly synthetic study templates are used.",
                "T3/T4 show anonymous prompt or question text from earlier calls as blinded stimuli; author IDs and judgments are withheld, but a strict ban on seeing any other persona's raw text conflicts with those evaluation tasks.",
            ],
        }
        print(json.dumps(result, indent=2))
        return
    try:
        print(json.dumps(run(), indent=2))
    except RuntimeError as exc:
        from app.simulated_analysis import analyze

        with connect() as db:
            report = {
                "status": "not_run_missing_model_configuration",
                "label": "SIMULATED",
                "synthetic": True,
                "source": "sim_persona",
                "persona_ids": [p["id"] for p in load_personas()],
                "reason": str(exc),
                "seed": SEED,
                "temperature": TEMPERATURE,
                "personas": len(load_personas()),
                "calls_completed": 0,
                "audit": audit(db),
                "analysis": analyze(db),
                "source_fixture_status": "Original five faculty prompt texts and source template are unavailable; these items were not invented.",
            }
        out = ROOT / "data" / "simulated"
        out.mkdir(parents=True, exist_ok=True)
        (out / "analysis.json").write_text(json.dumps(report["analysis"], indent=2) + "\n", encoding="utf-8")
        (out / "run_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
