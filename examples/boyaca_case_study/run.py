"""The Boyaca case study, reproduced from the published deposit.

Reads the engineered CHIRPS field and the released prediction arrays of the monthly
precipitation benchmark over the Colombian Andes (Zenodo concept DOI
10.5281/zenodo.21576207, or a checkout of the case-study repository), runs every
component through a pre-registered declaration, and checks the headline numbers against
the values printed in the published provenance tables.

    python examples/boyaca_case_study/run.py --thesis-root /path/to/deposit --out report/

The root is also read from ``ANCHORGATE_THESIS_ROOT``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from anchorgate.cli import main as cli_main

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "declaration.yaml"

# quantity, where to read it in report.json, published value, tolerance
CHECKS = [
    (
        "climatology pooled R2, purged block",
        ("C1", "values", "label", "climatology", "pooled_r2"),
        0.763,
        0.002,
    ),
    (
        "ConvLSTM seed-mean MSESS",
        ("C1", "by_configuration", "model", "ConvLSTM-Bidirectional", "msess_mean"),
        -0.543,
        0.002,
    ),
    (
        "GNN-TAT seed-mean MSESS",
        ("C1", "by_configuration", "model", "GNN-TAT-GAT", "msess_mean"),
        -1.038,
        0.002,
    ),
    (
        "shuffled folds, overlap fraction",
        ("C0", "values", "assignment", "shuffled", "fraction"),
        394 / 396,
        1e-6,
    ),
    (
        "contiguous folds, overlap fraction",
        ("C0", "values", "assignment", "contiguous", "fraction"),
        347 / 396,
        1e-6,
    ),
    (
        "purged holdout, overlap fraction",
        ("C0", "values", "assignment", "purged", "fraction"),
        0.0,
        1e-9,
    ),
    (
        "fusion minus raw best, purged",
        ("C3", "summary", "quantity", "total_gain", "mean"),
        -0.041,
        0.001,
    ),
    ("fusion in-sample gap", ("C3", "summary", "quantity", "in_sample_gap", "mean"), 0.084, 0.001),
]


def write_declaration(root: Path, out: Path) -> Path:
    text = TEMPLATE.read_text(encoding="utf-8").replace("{thesis_root}", root.as_posix())
    out.mkdir(parents=True, exist_ok=True)
    path = out / "declaration.yaml"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def lookup(report: dict, spec: tuple) -> float:
    component, table, key_col, key, col = spec
    res = next(r for r in report["results"] if r["component"].startswith(component))
    rows = res["values"] if table == "values" else res["tables"][table]
    row = next(r for r in rows if r[key_col] == key)
    return float(row[col])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--thesis-root", default=os.environ.get("ANCHORGATE_THESIS_ROOT"))
    ap.add_argument("--out", default="report")
    args = ap.parse_args()
    if not args.thesis_root:
        ap.error("--thesis-root or ANCHORGATE_THESIS_ROOT is required")
    root = Path(args.thesis_root).resolve()
    out = Path(args.out).resolve()
    decl = write_declaration(root, out)
    code = cli_main(["run", str(decl), "--out", str(out), "--conformal"])
    import json

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    rows = []
    for name, spec, published, tol in CHECKS:
        got = lookup(report, spec)
        rows.append(
            {
                "quantity": name,
                "package": got,
                "published": published,
                "agrees": abs(got - published) <= tol,
            }
        )
    table = pd.DataFrame(rows)
    print("\nAgainst the published provenance tables:")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if not table["agrees"].all():
        print("\nsome numbers disagree with the published values")
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
