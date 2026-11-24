from __future__ import annotations

import app.env  # noqa: F401
import json
import os
import urllib.request
from typing import Any

CONDITIONS = (
    "free_form_faculty",
    "filled_template",
    "filled_template_linter_repair",
    "lagrange_routed_template",
    "gaussian_process_routed_template",
)


def build_study_prompts(inputs: dict[str, str]) -> dict[str, str]:
    mapping = {
        "free_form_faculty": "faculty_prompt",
        "filled_template": "filled_template",
        "filled_template_linter_repair": "repaired_template",
        "lagrange_routed_template": "lagrange_template",
        "gaussian_process_routed_template": "gp_template",
    }
    missing = [key for key in mapping.values() if not inputs.get(key)]
    if missing:
        raise ValueError("Missing prompt inputs: " + ", ".join(missing))
    return {condition: inputs[field] for condition, field in mapping.items()}


def check_response(payload: Any, question_count: int) -> dict:
    items = payload if isinstance(payload, list) else payload.get("questions", []) if isinstance(payload, dict) else []
    problems = []
    if len(items) != question_count:
        problems.append({"check": "question_count", "expected": question_count, "actual": len(items)})
    for i, item in enumerate(items):
        choices = item.get("options", item.get("choices", [])) if isinstance(item, dict) else []
        labels = [x.get("label") for x in choices if isinstance(x, dict)]
        if len(choices) != 4 or labels != ["A", "B", "C", "D"]:
            problems.append({"check": "four_options_A_to_D", "question": i + 1})
        correct = [x for x in choices if isinstance(x, dict) and x.get("correct") is True]
        if isinstance(item, dict) and item.get("correct_answer") is not None and not correct:
            correct = [item]
        if len(correct) != 1:
            problems.append({"check": "exactly_one_correct_answer", "question": i + 1, "marked": len(correct)})
        if not isinstance(item, dict) or not str(item.get("explanation", "")).strip():
            problems.append({"check": "explanation_present", "question": i + 1})
    return {"passed": not problems, "question_count": len(items), "problems": problems}


def request_completion(prompt: str, model: str, seed: int = 42, temperature: float = 0.2) -> tuple[dict, dict]:
    key = (
        (os.environ.get("STUDY_API_KEY") or "").strip()
        or (os.environ.get("GROQ_API_KEY") or "").strip()
        or (os.environ.get("OPENAI_API_KEY") or "").strip()
    )
    if not key:
        raise RuntimeError("Set GROQ_API_KEY, STUDY_API_KEY, or OPENAI_API_KEY to run the LLM harness")
    is_groq = bool((os.environ.get("GROQ_API_KEY") or "").strip()) and not os.environ.get("STUDY_API_BASE")
    default_endpoint = (
        "https://api.groq.com/openai/v1/chat/completions" if is_groq else "https://api.openai.com/v1/chat/completions"
    )
    endpoint = os.environ.get("STUDY_API_BASE") or os.environ.get("GROQ_API_BASE") or default_endpoint
    request = {
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "Return a JSON object with a questions array. Each question has stem, options with labels A-D and exactly one correct true, and explanation.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(request).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "PromptRepositoryLab/0.1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        raw = json.loads(response.read())
    content = raw["choices"][0]["message"]["content"]
    return request, json.loads(content)


def execute_condition(
    prompt: str, model: str, question_count: int = 5, seed: int = 42, temperature: float = 0.2
) -> dict:
    request, response = request_completion(prompt, model, seed, temperature)
    content = response.get("questions", response)
    return {
        "model": model,
        "request": request,
        "response": response,
        "automatic_checks": check_response(content, question_count),
    }
