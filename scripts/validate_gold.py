"""Checks data/gold/inbox without modifying it. Prints counts only; invents nothing."""

import csv
import glob
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "gold", "inbox")
ATOMS = "very_easy expert_level no_basic ordered_easy_to_hard hidden_answers answer_after_each_q single_correct multi_interp confusing_language".split()
SLOTS = "role audience topic count n_options labeling marking explanation ordering".split()
problems = []
with open(os.path.join(ROOT, "prompts.csv"), newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
ids = [r["prompt_id"] for r in rows]
if len(set(ids)) != len(ids):
    problems.append("duplicate prompt_id")
for r in rows:
    if r["consent"].strip().lower() != "yes":
        problems.append(f"{r['prompt_id']}: consent is not yes")
    if not r["prompt_text"].strip():
        problems.append(f"{r['prompt_id']}: empty prompt_text")
print("prompts:", len(rows))
files = [p for p in glob.glob(os.path.join(ROOT, "annotations_*.csv")) if not p.endswith("TEMPLATE.csv")]
ann = {}
for p in files:
    with open(p, newline="", encoding="utf-8") as f:
        rr = list(csv.DictReader(f))
    name = os.path.basename(p)[12:-4]
    ann[name] = {r["prompt_id"]: r for r in rr}
    for r in rr:
        if r["prompt_id"] not in ids:
            problems.append(f"{name}: unknown prompt_id {r['prompt_id']}")
        for a in ATOMS:
            if r.get(a) not in ("0", "1"):
                problems.append(f"{name}/{r['prompt_id']}: {a} must be 0 or 1")
        for s in SLOTS:
            if r.get(s) not in ("0", "0.5", "1"):
                problems.append(f"{name}/{r['prompt_id']}: {s} must be 0, 0.5 or 1")
    print(f"annotator {name}: {len(rr)} prompts")
if len(ann) >= 2:
    both = set.intersection(*[set(v) for v in ann.values()])
    print("prompts annotated by all annotators:", len(both))
print("problems:", len(problems))
for p in problems[:50]:
    print(" -", p)
sys.exit(1 if problems else 0)
