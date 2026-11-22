from __future__ import annotations

import json
from pathlib import Path


def reference_fixtures() -> list[dict]:
    path = Path(__file__).with_name("reference_annotations.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [{**row, "source": data["source"], "annotator_count": data["annotator_count"]} for row in data["fixtures"]]
