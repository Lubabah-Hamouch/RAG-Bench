"""Compare the judge's claim labels with a human's. Fill `human_label` in claims_to_label.csv first.

Usage: python evaluation/validate_judge.py [path/to/claims_to_label.csv]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ragbench.metrics.validation import (
    agreement,
    coverage_agreement,
    load_coverage_labelled,
    load_labelled,
)

default = Path(__file__).parent / "results" / "claims_to_label.csv"
path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
rows = load_labelled(path)
coverage_rows = load_coverage_labelled(path)
if not rows and coverage_rows:
    print(json.dumps({"coverage": coverage_agreement(coverage_rows)}, indent=2))
if not rows:
    sys.exit(
        f"No human labels found in {path}. Fill the human_label column "
        "(supported / unsupported / contradicted; covered / not covered for coverage rows) and rerun."
    )
print(json.dumps(agreement(rows), indent=2))
if coverage_rows:
    print(json.dumps({"coverage": coverage_agreement(coverage_rows)}, indent=2))
