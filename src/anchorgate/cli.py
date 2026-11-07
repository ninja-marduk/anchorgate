"""Command line interface: ``anchorgate validate | audit | anchor``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .anchor import anchor
from .data import Benchmark, benchmark_from_declaration
from .declaration import Declaration
from .folds import (
    FoldAssignment,
    audit,
    contiguous_kfold,
    ladder,
    purged_holdout,
    released_scalar_shuffle,
    shuffled_kfold,
)
from .report import write_report
from .result import Result


def _load(decl_path: str) -> tuple[Declaration, Benchmark]:
    decl = Declaration.load(decl_path)
    root = Path(decl_path).resolve().parent
    bench = benchmark_from_declaration(decl.data, root=root)
    return decl, bench


def _assignments(decl: Declaration, bench: Benchmark) -> tuple[list[FoldAssignment], list[str]]:
    """Build the fold assignments the declaration names, or the standard ladder."""
    if not decl.folds:
        block: list[int] | None = None
        if "purged" in decl.scoring:
            block = decl.score_origins(bench.n_origin, bench.horizon).tolist()
        return ladder(bench, score=block), ["purged"]
    out: list[FoldAssignment] = []
    oos: list[str] = []
    for spec in decl.folds:
        kind = str(spec.get("kind"))
        name = str(spec.get("name", kind))
        n = bench.n_origin
        if kind == "shuffled":
            out.append(
                shuffled_kfold(
                    n, k=int(spec.get("k", 5)), seed=int(spec.get("seed", 42)), name=name
                )
            )
        elif kind == "contiguous":
            out.append(contiguous_kfold(n, k=int(spec.get("k", 5)), name=name))
        elif kind == "purged":
            raw = spec.get("score")
            if raw is None:
                score_list = decl.score_origins(n, bench.horizon).tolist()
            elif isinstance(raw, dict):
                score_list = list(range(int(raw["start"]), int(raw["stop"])))
            else:
                score_list = [int(s) for s in raw]
            out.append(purged_holdout(n, bench.lead, score_list, name=name))
        elif kind == "released":
            out.append(released_scalar_shuffle(n, name=name))
        else:
            raise ValueError(f"unknown fold kind {kind!r}")
        if spec.get("out_of_sample"):
            oos.append(name)
    return out, oos


def cmd_validate(args: argparse.Namespace) -> int:
    decl, bench = _load(args.declaration)
    notes = bench.validate()
    f = bench.reference
    print(f"{decl.describe()}")
    print(
        f"{len(bench.forecasts)} forecasts on {f.n_origin} origins x {f.n_lead} leads"
        f" x {f.n_cell} cells; contract OK"
    )
    for n in notes:
        print(f"note: {n}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    decl, bench = _load(args.declaration)
    bench.validate()
    assignments, oos = _assignments(decl, bench)
    res = audit(bench, assignments, out_of_sample=oos, declaration_hash=decl.sha256)
    _finish([res], bench, decl, args.out)
    return 0 if res.verdict.value != "FAIL" else 1


def cmd_anchor(args: argparse.Namespace) -> int:
    decl, bench = _load(args.declaration)
    res = anchor(bench, decl, mask=args.mask, categorical=args.categorical)
    _finish([res], bench, decl, args.out)
    return 0


def _finish(results: list[Result], bench: Benchmark, decl: Declaration, out: str | None) -> None:
    for r in results:
        print(r.summary())
        for n in r.notes:
            print(f"  {n}")
    if out:
        md, js = write_report(out, results, bench, decl)
        print(f"report: {md} and {js}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="anchorgate", description="An evaluation protocol with a gate."
    )
    ap.add_argument("--version", action="version", version=f"anchorgate {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="check the data contract")
    p.add_argument("declaration")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("audit", help="component 0: target-period overlap of fold assignments")
    p.add_argument("declaration")
    p.add_argument("--out", default=None, help="directory for report.md and report.json")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("anchor", help="component 1: anchor to references and gate")
    p.add_argument("declaration")
    p.add_argument("--out", default=None)
    p.add_argument("--mask", default=None, help="'domain' to score inside the observations' mask")
    p.add_argument("--categorical", action="store_true", help="add tercile categorical scores")
    p.set_defaults(func=cmd_anchor)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
