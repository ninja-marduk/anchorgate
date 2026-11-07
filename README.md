# anchorgate

An evaluation protocol with a gate for spatiotemporal predictions.

`anchorgate` takes the saved predictions of any model of a gridded or station field
(monthly precipitation, temperature, streamflow, anything scored at several lead
times) and answers, with a verdict and the rule that produced it:

- **Does the model carry information an untrained reference does not?** Every forecast
  and every reference (per-cell climatology, persistence, seasonal naive) is scored on
  identical targets; the gate passes only when the mean over seeds of the skill score
  against the reference is positive (component 1).
- **Is the holdout actually out of sample?** Consecutive forecast origins share target
  periods, so folds built from whole origins, even at random, keep most held-out
  targets inside the training fold. The audit measures that fraction for any fold
  assignment and builds a purged holdout whose fraction is zero (component 0).
- **How many independent units does the record hold?** Overlapping origins are not
  independent observations; the audit counts distinct target periods and disjoint
  windows before any test writes an N.

Components 2 to 5 (seed-resolved factorial, calibration-versus-combination
decomposition of ensemble gain, deseasonalised anomaly skill, spatial diagnostics) are
under development; see `CHANGELOG.md`.

Every report carries the SHA-256 of a **declaration** written before scoring: which
origins are scored, which references, which gate, how many seeds, and what will be
concluded on PASS and on FAIL.

## Install

```
pip install anchorgate            # once released; until then:
pip install -e ".[dev]"           # from a checkout
```

Python 3.10 or later; `numpy`, `pandas`, `scipy`, `xarray`, `pyyaml`. NetCDF input
needs `netCDF4`.

## Ten lines

```python
import numpy as np
from anchorgate import Observations, Forecast, Benchmark, anchor, audit, ladder

obs = Observations(series=precip, time=months, cutoff=414)          # (time, y, x); training is time < cutoff
runs = [Forecast(pred=p, target=y, origin_time=origins, lead=np.arange(1, 13),
                 model="GNN-TAT", features="BASIC", seed=s) for s, p in zip((42, 123, 456), preds)]
bench = Benchmark(runs, obs)
bench.validate()                                                     # identical targets, no training leakage

print(anchor(bench, score_origins=range(21, 33)).summary())          # C1: the gate on a scored block
print(audit(bench, ladder(bench, score=range(21, 33)), out_of_sample=["purged"]).summary())  # C0
```

Or from the command line with a declaration file:

```
anchorgate validate declaration.yaml
anchorgate audit    declaration.yaml --out report/
anchorgate anchor   declaration.yaml --out report/ --categorical
```

```yaml
# declaration.yaml
name: boyaca-monthly-precipitation
data:
  observations: {format: field, path: dataset.nc, variable: total_precipitation, cutoff: "2016-07"}
  forecasts:
    - {format: directory, pattern: "output/GNN_TAT/SEED*/predictions.npy", model: GNN-TAT, features: BASIC}
    - {format: netcdf, path: convlstm_seed42.nc}
scoring: {purged: {start: 21, stop: 33}}
references: [climatology, persistence, seasonal_naive]
gate: {metric: msess, reference: climatology, threshold: 0.0, min_seeds: 2}
folds:
  - {kind: shuffled, k: 5, seed: 42}
  - {kind: contiguous, k: 5}
  - {kind: purged, name: purged, out_of_sample: true}
conclusions:
  pass: "the configuration extracts information beyond the per-cell climatology"
  fail: "no configuration exceeds the per-cell climatology on the purged block"
```

## Data contract

| object | holds | checked on `validate()` |
|---|---|---|
| `Observations` | `(time, cells)` field, monthly `time`, `cutoff` index | consecutive periods, cutoff inside the record |
| `Forecast` | `pred`, `target` as `(origin, lead, cells)`, `origin_time`, `lead`, labels `model`, `features`, `seed` | shapes, increasing leads and origins |
| `Benchmark` | forecasts plus observations | identical targets across forecasts; targets equal the observations at `origin + lead`; no target inside the training period; duplicate arrays and mixed seed labels reported |

Loaders: in-memory arrays, long tables (`origin_time, lead, cell, pred, target`), NetCDF
written by the package, and the `SEED<n>/.../predictions.npy` directory layout with
origins recovered from the observations.

## What it refuses to be

Not a metrics library: `xskillscore`, `scores`, `climpred` and `hydroeval` already
provide deterministic and probabilistic metrics on xarray objects. `anchorgate`
implements the few scores its rules need (pooled R2, per-cell NSE, MSESS, RMSE, MAE,
bias, tercile categorical scores) and spends its code on what those libraries do not
do: the fold audit, the seed-blocked design, the gate and the declaration.

Not a training framework: inputs are saved predictions.

## Reproducibility

The test suite includes synthetic cases with known answers (a perfect forecast scores
MSESS 1 and passes; a climatology forecast scores 0 and fails; a fold assignment with a
planted overlap reports it) and a regression suite against a published monthly
precipitation benchmark over the Colombian Andes, enabled with
`ANCHORGATE_THESIS_ROOT=<checkout of the case-study repository>`.

## Licence

MIT. See `CITATION.cff` for how to cite.
