# anchorgate

An evaluation protocol with a gate for spatiotemporal predictions.

`anchorgate` takes the saved predictions of any model of a gridded or station field,
scored at one or several lead times, and answers six questions with a verdict and the
rule that produced it:

| component | question | verdict |
|---|---|---|
| C0 fold audit | Is the holdout actually out of sample at this horizon? | PASS when every assignment declared out of sample has zero target-period overlap |
| C1 anchoring | Does the model carry information an untrained reference does not? | PASS when the seed mean of the skill score against the reference is positive |
| C2 seed-resolved factorial | Is a difference between configurations larger than the seed? | PASS when a paired, within-seed contrast survives Holm over the declared family |
| C3 fusion decomposition | Does an ensemble add anything that recalibrating its best member does not? | PASS when the combination gain is positive on a purged split |
| C4 anomaly skill | Is there skill once the seasonal cycle is removed? | PASS when the per-cell temporal anomaly correlation and the MSESS are both positive |
| C5 spatial diagnostics | Where on the map does the model win? | PASS when one configuration has the lowest error in a majority of cells and in more cells than every reference |

A [declaration](declaration.md) fixes the design of the evaluation; every component reads its
choices from it, records the parameters it used and lists any choice overridden at call time,
and the report identifies the design by its hash.

## Install

```
pip install anchorgate              # once released
pip install -e ".[dev]"             # from a checkout
pip install "anchorgate[remote]"    # to read CHIRPS remotely in the regimes example
```

## Sixty seconds

```python
from anchorgate import Benchmark, Forecast, Observations, anchor, audit, ladder

obs = Observations(series=field, time=months, cutoff=cutoff_index)
runs = [Forecast(pred=p, target=y, origin_time=origins, lead=leads, model="mine", seed=s)
        for s, p in zip(seeds, predictions)]
bench = Benchmark(runs, obs)
bench.validate()
print(audit(bench, ladder(bench), out_of_sample=["purged"]).summary())
print(anchor(bench).summary())
```

Or, from a declaration file:

```
anchorgate run declaration.yaml --out report/
```

## What it refuses to be

Not a metrics library (`xskillscore`, `scores`, `climpred` and `hydroeval` already do
that), not a training framework. Inputs are saved predictions; outputs are verdicts, the
rules behind them, and the numbers.

## Examples

- `examples/bring_your_own_model.py`: a per-cell ridge regression with three bootstrap
  seeds on a synthetic field; every component in under a minute.
- `examples/eight_regimes.py`: eight CHIRPS regions, references only; how strong the
  reference is and how much ordinary folds leak outside one case study.
- `examples/enso_ridge.py`: a per-cell ridge on lagged Nino 3.4 (and a damped-persistence
  baseline) on the same deposit and purged block as the case study; the one configuration
  on this benchmark that clears the gate, with its skill horizon.
- `examples/boyaca_case_study/`: the published monthly precipitation benchmark over the
  Colombian Andes, reproduced from its deposit and checked against the published numbers.
