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
- **Is the difference between two configurations larger than the seed?** Runs crossed
  by model and feature set are analysed as a blocked design with the seed as block:
  blocked ANOVA, a within-seed permutation test, every contrast as a paired difference
  with Holm correction over the declared family, the inflation that quoting the best
  seed adds, and the minimum detectable effect at the number of seeds, so that an
  unresolved ranking is never read as equivalence (component 2).
- **Does an ensemble add anything that recalibrating its best member does not?** The
  gain is split out of fold into a calibration term and a combination term, the best
  base learner is chosen on the fit origins, and the advantage of choosing it on the
  scored origins is printed as its own number (component 3).

- **Is there skill once the seasonal cycle is removed?** Forecast and target are
  deseasonalised against the training climatology, which then scores exactly zero. The
  anomaly correlation is computed per cell over time and per origin over space, never
  pooled over both (the pooled number is printed as the inflated comparison), with
  optional conformal intervals whose exchangeable unit is the origin (component 4).
- **Where on the map does a model win?** The per-cell winner among configurations and
  references, skill inside user-named strata, a collapse diagnostic that tells a
  forecast of each cell's mean from a forecast that varies in time, and an optional audit
  of a graph model's adjacency, including whether its correlation term saw the evaluation
  period (component 5).

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
from anchorgate import (Observations, Forecast, Benchmark, anchor, audit, ladder, factorial,
                        decompose_fusion, anomaly_skill, spatial_diagnostics, strata_bands)

bands = strata_bands(elevation, edges=[1500, 2800], labels=["low", "mid", "high"])

obs = Observations(series=precip, time=months, cutoff=414)          # (time, y, x); training is time < cutoff
runs = [Forecast(pred=p, target=y, origin_time=origins, lead=np.arange(1, 13),
                 model="GNN-TAT", features="BASIC", seed=s) for s, p in zip((42, 123, 456), preds)]
bench = Benchmark(runs, obs)
bench.validate()                                                     # identical targets, no training leakage

print(anchor(bench, score_origins=range(21, 33)).summary())          # C1: the gate on a scored block
print(audit(bench, ladder(bench, score=range(21, 33)), out_of_sample=["purged"]).summary())  # C0
print(factorial(bench, score_origins=range(21, 33)).summary())       # C2: seed-blocked contrasts
print(decompose_fusion(bench, assignment="purged", score_origins=range(21, 33)).summary())  # C3
print(anomaly_skill(bench, score_origins=range(21, 33)).summary())   # C4: skill on anomalies
print(spatial_diagnostics(bench, strata={"elevation": bands}).summary())  # C5: who wins where
```

Or from the command line with a declaration file:

```
anchorgate validate  declaration.yaml
anchorgate audit     declaration.yaml --out report/
anchorgate anchor    declaration.yaml --out report/ --categorical
anchorgate factorial declaration.yaml --out report/
anchorgate fusion    declaration.yaml --out report/ --assignment purged
anchorgate anomaly   declaration.yaml --out report/ --conformal
anchorgate spatial   declaration.yaml --out report/ --leakage
anchorgate run       declaration.yaml --out report/        # every component the declaration supports
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
contrasts: [["GNN-TAT[BASIC]", "ConvLSTM[BASIC]"]]       # the C2 family; all pairs when omitted
factorial: {metric: pooled_r2, aggregate: lead_mean, n_perm: 20000}
fusion:
  base: [{model: ConvLSTM, features: BASIC}, {model: GNN-TAT, features: BASIC}]
  assignment: purged
strata:
  elevation: {variable: elevation, edges: [1500, 2800], labels: [low, mid, high]}
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
do: the fold audit, the seed-blocked design, the fusion decomposition, the anomaly and
collapse diagnostics, the gate and the declaration.

Not a training framework: inputs are saved predictions.

## Examples

- `examples/bring_your_own_model.py`: a per-cell ridge regression on lagged anomalies with
  three bootstrap seeds, on a synthetic field; every component in about half a minute.
- `examples/eight_regimes.py`: eight CHIRPS regions read remotely, references only; the
  strength of the climatology and the leakage of ordinary folds outside one case study
  (`pip install "anchorgate[remote]"`).
- `examples/enso_ridge.py`: a per-cell ridge on lagged Nino 3.4 (and a damped-persistence
  baseline) on the same deposit and purged block as the case study; the one configuration
  on this benchmark that clears the gate, with its skill horizon.
- `examples/boyaca_case_study/`: the published monthly precipitation benchmark over the
  Colombian Andes, run from its deposit through a declared design and checked
  against the published provenance numbers.

Documentation: `mkdocs serve` after `pip install -e ".[docs]"`; the rules page states
every conclusion rule in one sentence.

## Reproducibility

The test suite includes synthetic cases with known answers (a perfect forecast scores
MSESS 1 and passes; a climatology forecast scores 0 and fails; a fold assignment with a
planted overlap reports it; an ensemble whose gain is pure calibration decomposes to a
combination term of zero; a planted factor effect survives the blocked tests and a null
design does not) and a regression suite against a published monthly precipitation
benchmark over the Colombian Andes, enabled with
`ANCHORGATE_THESIS_ROOT=<checkout of the case-study repository>`.

## Licence

MIT. See `CITATION.cff` for how to cite.
