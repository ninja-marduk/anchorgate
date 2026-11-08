"""The object every component returns."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class Verdict(str, Enum):
    """What a component concluded under the rule it applied."""

    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT = "INSUFFICIENT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class Result:
    """Numbers, the rule applied to them, and the verdict the rule produced.

    ``values`` is the main tidy table; ``tables`` holds secondary tables (by lead, by
    stratum, categorical scores). ``inputs`` maps a label to the SHA-256 of the array it
    was computed from, so a report can be traced to the exact data it scored.
    """

    component: str
    title: str
    values: pd.DataFrame
    verdict: Verdict
    rule: str
    inputs: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    declaration_hash: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    deviations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "component": self.component,
            "title": self.title,
            "verdict": self.verdict.value,
            "rule": self.rule,
            "inputs": dict(self.inputs),
            "notes": list(self.notes),
            "declaration_hash": self.declaration_hash,
            "parameters": _jsonable(self.parameters),
            "deviations": list(self.deviations),
            "values": _frame_records(self.values),
            "tables": {k: _frame_records(v) for k, v in self.tables.items()},
        }

    def summary(self) -> str:
        return f"{self.component}: {self.verdict.value} ({self.rule})"


def _jsonable(value: Any) -> Any:
    """Plain-Python form of parameters (arrays, numpy scalars, dataclasses)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return {k: _jsonable(getattr(value, k)) for k in value.__dataclass_fields__}
    return value


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in frame.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for k, v in rec.items():
            if isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, float) and np.isnan(v):
                v = None
            clean[str(k)] = v
        out.append(clean)
    return out


def array_hash(arr: np.ndarray) -> str:
    """SHA-256 of an array's bytes, with its shape and dtype folded in."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()
