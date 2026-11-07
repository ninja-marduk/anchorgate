"""anchorgate: an evaluation protocol with a gate for spatiotemporal predictions.

Saved predictions of any model are scored against untrained references on identical
targets, fold assignments are audited for target-period overlap, seed variance is
resolved before any architectural claim, and every report carries the hash of a
declaration written before scoring.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .anchor import anchor, gate_table, quick_anchor
from .data import (
    Benchmark,
    Forecast,
    Observations,
    benchmark_from_declaration,
    forecasts_from_directory,
    from_arrays,
    infer_origin_time,
)
from .declaration import Declaration, Gate
from .folds import (
    FoldAssignment,
    audit,
    contiguous_kfold,
    effective_units,
    ladder,
    overlap,
    purged_holdout,
    released_scalar_shuffle,
    scalar_shuffle_overlap,
    shuffled_kfold,
)
from .references import build_references, climatology_table, training_terciles
from .report import render_markdown, to_json, write_report
from .result import Result, Verdict

__all__ = [
    "Benchmark",
    "Declaration",
    "FoldAssignment",
    "Forecast",
    "Gate",
    "Observations",
    "Result",
    "Verdict",
    "__version__",
    "anchor",
    "audit",
    "benchmark_from_declaration",
    "build_references",
    "climatology_table",
    "contiguous_kfold",
    "effective_units",
    "forecasts_from_directory",
    "from_arrays",
    "gate_table",
    "infer_origin_time",
    "ladder",
    "overlap",
    "purged_holdout",
    "quick_anchor",
    "released_scalar_shuffle",
    "render_markdown",
    "scalar_shuffle_overlap",
    "shuffled_kfold",
    "to_json",
    "training_terciles",
    "write_report",
]
