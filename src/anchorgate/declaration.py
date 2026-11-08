"""The declaration: the design of the evaluation, as a file.

It names the data, the scored origins, the references, the gate, the seeds expected per
configuration, the contrast family, the fold assignments, the strata and what will be
concluded on PASS and on FAIL. Every component reads its choices from it, records the
choices it actually used in its ``Result`` and lists any deviation, so a report can be
read against the design it claims to follow. Two hashes identify it: the file hash (its
bytes) and the design hash (its sections other than ``data``, in canonical form), so the
same design on another machine or another dataset carries the same design hash.
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
    level: float = 0.9
    block: int = 3
    n_boot: int = 1000


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
    factorial: dict[str, Any] = field(default_factory=dict)
    fusion: dict[str, Any] = field(default_factory=dict)
    raw: bytes = b""
    path: Path | None = None
    source: str = "memory"

    @property
    def sha256(self) -> str:
        """Hash of the loaded bytes (or of the canonical form of an in-memory mapping)."""
        return hashlib.sha256(self.raw).hexdigest()

    @property
    def design_hash(self) -> str:
        """Hash of the design alone: every section except ``data``, in canonical YAML."""
        doc = {
            "name": self.name,
            "scoring": self.scoring,
            "references": list(self.references),
            "gate": vars(self.gate),
            "seeds_expected": self.seeds_expected,
            "contrasts": self.contrasts,
            "strata": self.strata,
            "conclusions": self.conclusions,
            "folds": self.folds,
            "factorial": self.factorial,
            "fusion": self.fusion,
        }
        return hashlib.sha256(yaml.safe_dump(doc, sort_keys=True).encode("utf-8")).hexdigest()

    @property
    def short_design_hash(self) -> str:
        return self.design_hash[:12]

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
        d.source = "file"
        return d

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> Declaration:
        """Build from a mapping; hashed on its canonical YAML form and marked as coming
        from memory rather than from a file."""
        gate_doc = dict(doc.get("gate") or {})
        gate = Gate(
            metric=str(gate_doc.get("metric", "msess")),
            threshold=float(gate_doc.get("threshold", 0.0)),
            min_seeds=int(gate_doc.get("min_seeds", 2)),
            reference=str(gate_doc.get("reference", "climatology")),
            level=float(gate_doc.get("level", 0.9)),
            block=int(gate_doc.get("block", 3)),
            n_boot=int(gate_doc.get("n_boot", 1000)),
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
            factorial=dict(doc.get("factorial") or {}),
            fusion=dict(doc.get("fusion") or {}),
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
        where = str(self.path) if self.path else "(in memory)"
        return (
            f"declared design '{self.name}' {self.short_design_hash}"
            f" (file {self.short_hash}) from {where}"
        )

    def conclusion_for(self, verdict: str) -> str | None:
        """The declared conclusion for a PASS or FAIL verdict, if one was written."""
        return self.conclusions.get(verdict.lower())


def deviations(declaration: Declaration | None, overridden: dict[str, bool]) -> list[str]:
    """Names of the declared choices a caller overrode at call time, as sentences."""
    if declaration is None:
        return []
    return [
        f"{name} was set at call time, not taken from the declaration"
        for name, flag in overridden.items()
        if flag
    ]
