# Tutorial: bring your own model

The whole protocol on a model you can train in seconds. The full script is
`examples/bring_your_own_model.py`; this page walks through it.

## 1. Observations

An `Observations` object holds the field as `(time, cells)`, its monthly time axis and
the training cutoff (the index of the first month that is not training). Everything a
reference learns is learned on `time[:cutoff]`.

```python
from anchorgate.synthetic import synthetic_observations
obs = synthetic_observations(seed=0, n_years=40, shape=(6, 8))   # or Observations.from_field(da, cutoff_time="2016-07")
```

## 2. Your model's forecasts

A `Forecast` is `pred` and `target` as `(origin, lead, cells)`, the origin times, the
leads and three labels: `model`, `features`, `seed`. The example fits a per-cell ridge
regression on lagged values; the "seeds" are bootstrap resamples of the training
origins.

```python
runs = [ridge_forecast(obs, feats, seed) for feats in ("LAGS", "LAG0") for seed in (1, 2, 3)]
bench = Benchmark(runs, obs)
bench.validate()   # identical targets, targets equal observations at origin + lead, no training month
```

## 3. Choose the scored block before looking

The last two horizons of origins form the scored block; the purged fit set is derived
from it by C0 with an embargo of `horizon - 1` origins.

```python
n = bench.n_origin
block = list(range(n - 2 * bench.horizon, n))
```

## 4. Run the components

```python
from anchorgate import anchor, audit, ladder, factorial, decompose_fusion, anomaly_skill, spatial_diagnostics

results = [
    audit(bench, ladder(bench, score=block), out_of_sample=["purged"]),
    anchor(bench, score_origins=block),
    factorial(bench, score_origins=block),
    decompose_fusion(bench, assignment="purged", score_origins=block),
    anomaly_skill(bench, score_origins=block, conformal=True),
    spatial_diagnostics(bench, strata={"wet_dry": wet_dry}, score_origins=block),
]
for r in results:
    print(r.summary())
```

## 5. Read the verdicts

- C0 PASS: the purged holdout has zero target-period overlap; the shuffled and
  contiguous assignments are shown with their overlap so a reader sees what a naive
  cross-validation would have claimed.
- C1: the ridge on lags usually passes the gate on a synthetic field whose anomalies are
  autocorrelated, and fails on one whose anomalies are white noise. Both are correct.
- C2: with three bootstrap seeds, the contrast between the two feature sets is judged
  against the dispersion of its paired difference; the minimum detectable effect says
  what "not significant" means here.
- C3: the gain of combining the two feature sets over recalibrating the better one.
- C4: skill once the seasonal cycle is removed, per cell over time.
- C5: which cells each configuration wins, and whether any forecast has collapsed to
  the cell mean.

## 6. Write the report

```python
from anchorgate import write_report
write_report("report/", results, bench)
```

To make the run pre-registered, put the choices above in a declaration file and run
`anchorgate run declaration.yaml --out report/`; the report then carries the file's hash.
