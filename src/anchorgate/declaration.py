"""The declaration: what will be scored, how, and what will be concluded, written first.

The file is hashed when loaded and the hash is carried into every result, so a report
whose declaration changed after scoring is visibly a different report.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_REFERENCES = ("climatology", "persistence", "seasonal_naive")


@dataclass
class Gate:
    metric: str = "msess"
    threshold: float = 0.0
    min_seeds: int = 2
    reference: str = "climatology"


@dataclass
class Declaration:
    name: str
    data: dict[str, Any] = field(default_factory=dict)
    scoring: dict[str, Any] = field(default_factory=lambda: {"origins": "all"})
    references: tuple[str, ...] = DEFAULT_REFERENCES
    gate: Gate = field(default_factory=Gate)
    seeds_expected: int = 3
    contrasts: list[list[str]] = field(default_factory=list)
    strata: dict[str, Any] = field(default_factory=dict)
    conclusions: dict[str, str] = field(default_factory=dict)
    folds: list[dict[str, Any]] = field(default_factory=list)
    raw: bytes = b""
    path: Path | None = None
    preregistered: bool = False

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()

    @property
    def short_hash(self) -> str:
        return self.sha256[:12]

    # ----------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str | Path) -> Declaration:
        """Load a YAML declaration; the file's bytes are what gets hashed."""
        p = Path(path)
        raw = p.read_bytes()
        doc = yaml.safe_load(raw) or {}
        if not isinstance(doc, dict):
            raise ValueError(f"{p}: the declaration must be a mapping")
        d = cls.from_dict(doc)
        d.raw = raw
        d.path = p
        d.preregistered = True
        return d

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> Declaration:
        """Build from a mapping; hashed on its canonical YAML form, flagged as not
        pre-registered because nothing proves it was written before scoring."""
        gate_doc = dict(doc.get("gate") or {})
        gate = Gate(
            metric=str(gate_doc.get("metric", "msess")),
            threshold=float(gate_doc.get("threshold", 0.0)),
            min_seeds=int(gate_doc.get("min_seeds", 2)),
            reference=str(gate_doc.get("reference", "climatology")),
        )
        refs = doc.get("references") or list(DEFAULT_REFERENCES)
        d = cls(
            name=str(doc.get("name", "unnamed")),
            data=dict(doc.get("data") or {}),
            scoring=dict(doc.get("scoring") or {"origins": "all"}),
            references=tuple(str(r) for r in refs),
            gate=gate,
            seeds_expected=int(doc.get("seeds_expected", 3)),
            contrasts=[list(map(str, c)) for c in (doc.get("contrasts") or [])],
            strata=dict(doc.get("strata") or {}),
            conclusions={str(k): str(v) for k, v in (doc.get("conclusions") or {}).items()},
            folds=[dict(f) for f in (doc.get("folds") or [])],
        )
        d.raw = yaml.safe_dump(doc, sort_keys=True).encode("utf-8")
        return d

    # ---------------------------------------------------------------- scoring
    def score_origins(self, n_origin: int, horizon: int) -> np.ndarray:
        """Resolve the ``scoring`` section to origin indices.

        Accepted forms::

            scoring: {origins: all}
            scoring: {origins: [21, 22, 23]}
            scoring: {origins: {start: 21, stop: 33}}      # half-open, like Python
            scoring: {purged: {start: 21, stop: 33}}       # same, and C0 builds the fit set
        """
        spec = self.scoring
        if "purged" in spec:
            block = spec["purged"]
            return np.arange(int(block["start"]), int(block["stop"]))
        origins = spec.get("origins", "all")
        if isinstance(origins, str):
            if origins != "all":
                raise ValueError(f"unknown scoring.origins value {origins!r}")
            return np.arange(n_origin)
        if isinstance(origins, dict):
            return np.arange(int(origins["start"]), int(origins.get("stop", n_origin)))
        return np.asarray(sorted(int(o) for o in origins), dtype=int)

    def describe(self) -> str:
        where = str(self.path) if self.path else "(in memory, not pre-registered)"
        return f"declaration '{self.name}' {self.short_hash} from {where}"
