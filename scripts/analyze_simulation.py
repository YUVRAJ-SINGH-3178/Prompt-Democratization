"""Refresh the descriptive pilot report and enforce synthetic provenance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.simulated_analysis import analyze  # noqa: E402
from app.simulated_database import audit, connect  # noqa: E402


def main() -> None:
    with connect() as db:
        checked = audit(db)
        report = analyze(db)
    output = {
        "label": "SIMULATED",
        "synthetic": True,
        "source": "sim_persona",
        "persona_ids": report["persona_ids"],
        "audit": checked,
        "analysis": report,
    }
    target = ROOT / "data" / "simulated" / "analysis.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if checked["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
