"""anchorgate: an evaluation protocol with a gate for spatiotemporal predictions.

Saved predictions of any model are scored against untrained references on identical
targets, fold assignments are audited for target-period overlap, seed variance is
resolved before any architectural claim, and every report carries the hash of a
declaration written before scoring.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .anchor import anchor, gate_table, quick_anchor
from .anomaly import anomaly_scores, anomaly_skill, conformal_by_origin, pattern_acc, temporal_acc
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
from .factorial import (
    cliffs_delta,
    factorial,
    friedman_nemenyi,
    group_contrasts,
    holm,
    inflation,
    lead_scores,
    minimum_detectable_effect,
    nemenyi_cd,
    paired_contrasts,
    permutation_test,
    rcbd_anova,
    responses,
)
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
from .fusion import decompose_fusion, fusion_ladder, resolve_assignment
from .references import build_references, climatology_table, training_terciles
from .report import render_markdown, to_json, write_report
from .result import Result, Verdict
from .spatial import (
    collapse_diagnostic,
    correlation_leakage,
    graph_audit,
    skill_by_strata,
    spatial_diagnostics,
    spatial_from_declaration,
    strata_bands,
    strata_from_declaration,
    winner_map,
)

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
    "anomaly_scores",
    "anomaly_skill",
    "audit",
    "benchmark_from_declaration",
    "build_references",
    "cliffs_delta",
    "climatology_table",
    "collapse_diagnostic",
    "conformal_by_origin",
    "contiguous_kfold",
    "correlation_leakage",
    "decompose_fusion",
    "effective_units",
    "factorial",
    "forecasts_from_directory",
    "friedman_nemenyi",
    "from_arrays",
    "fusion_ladder",
    "gate_table",
    "graph_audit",
    "group_contrasts",
    "holm",
    "infer_origin_time",
    "inflation",
    "ladder",
    "lead_scores",
    "minimum_detectable_effect",
    "nemenyi_cd",
    "overlap",
    "paired_contrasts",
    "pattern_acc",
    "permutation_test",
    "purged_holdout",
    "quick_anchor",
    "rcbd_anova",
    "released_scalar_shuffle",
    "render_markdown",
    "resolve_assignment",
    "responses",
    "scalar_shuffle_overlap",
    "shuffled_kfold",
    "skill_by_strata",
    "spatial_diagnostics",
    "spatial_from_declaration",
    "strata_bands",
    "strata_from_declaration",
    "temporal_acc",
    "to_json",
    "training_terciles",
    "winner_map",
    "write_report",
]
