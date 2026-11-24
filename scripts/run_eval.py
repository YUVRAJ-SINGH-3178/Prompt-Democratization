#!/usr/bin/env python
"""Evaluation CLI script (scripts/run_eval.py).

Provides command-line subcommands for:
- dev-eval: Evaluates the development dataset (Split 1 + Split 2).
- frozen-eval: Evaluates a frozen test set (default: v3) with hash check and single-run gate.
- real-eval: Evaluates against real faculty prompt data (when available).
- model-sweep: Compares multiple LLM models across the development set only.

Works natively across Windows (PowerShell) and Linux without requiring Make.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.env  # noqa: F401, E402
from app.extractor.eval import run_dev_eval, run_frozen_eval, run_model_sweep  # noqa: E402
from app.real_eval import run_real_evaluation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run_eval",
        description="Unified Evaluation CLI for Prompt Repository Lab",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # dev-eval
    sub.add_parser("dev-eval", help="Run evaluation across development dataset")

    # frozen-eval
    froz = sub.add_parser("frozen-eval", help="Run evaluation across frozen test set")
    froz.add_argument(
        "--set",
        dest="set_name",
        default="v3",
        help="Frozen set version to evaluate (e.g. v3, v2). Default: v3",
    )
    froz.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force execution even if evaluation report already exists",
    )

    # real-eval
    sub.add_parser("real-eval", help="Run evaluation across consented real faculty prompts")

    # model-sweep
    sweep = sub.add_parser("model-sweep", help="Compare multiple LLM reader models on development dataset")
    sweep.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of models to evaluate (e.g. 'openai/gpt-oss-20b,openai/gpt-oss-120b,qwen/qwen3.8-27b')",
    )

    args = parser.parse_args()

    if args.subcommand == "dev-eval":
        res = run_dev_eval()
        print(f"dev-eval finished. Report: data/eval/dev_evaluation_report.md (n={res['provenance']['total_prompts']})")
        return 0

    elif args.subcommand == "frozen-eval":
        res = run_frozen_eval(force=args.force, set_name=args.set_name)
        if res.get("status") == "folder_absent":
            return 0
        print(f"frozen-eval ({args.set_name}) finished. Report: data/eval/frozen_{args.set_name}_report.md")
        return 0

    elif args.subcommand == "real-eval":
        res = run_real_evaluation()
        print(f"real-eval status: {res.get('status')}")
        return 0

    elif args.subcommand == "model-sweep":
        model_list = [m.strip() for m in args.models.split(",") if m.strip()]
        res = run_model_sweep(model_list)
        return 0

    else:
        print(f"Unknown subcommand: {args.subcommand}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
