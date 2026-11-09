# The declaration

A YAML file that fixes the design of the evaluation. It says which origins are scored, which references
are built, what the gate is, how many seeds are expected, which contrasts form the
family, which strata are reported and what will be concluded on PASS and on FAIL. The
package hashes the file's bytes and prints the hash in every report, so a declaration
changed after scoring is visibly a different report. A declaration built in memory
(`Declaration.from_dict`) is hashed too and marked as coming from memory. Two hashes identify a
declaration: the file hash (its bytes) and the design hash (every section except `data`, in
canonical form), so the same design run on another machine carries the same design hash. Every
component records in its result the parameters it actually used and lists any choice that was
set at call time instead of taken from the declaration; the report prints both. The hash is
provenance: it says which design a report followed, not when the design was written.

```yaml
name: boyaca-monthly-precipitation
data:
  observations: {format: field, path: dataset.nc, variable: total_precipitation, cutoff: "2016-07"}
  forecasts:
    - {format: directory, pattern: "output/GNN_TAT/SEED*/predictions.npy", model: GNN-TAT, features: BASIC}
    - {format: netcdf, path: convlstm_seed42.nc}
    - {format: long_table, path: ridge.csv, model: ridge, features: LAGS}
scoring: {purged: {start: 21, stop: 33}}
references: [climatology, persistence, seasonal_naive]
gate: {metric: msess, reference: climatology, threshold: 0.0, min_seeds: 2}
seeds_expected: 3
folds:
  - {kind: released, name: released}
  - {kind: shuffled, name: shuffled, k: 5, seed: 42}
  - {kind: contiguous, name: contiguous, k: 5}
  - {kind: purged, name: purged, out_of_sample: true}
contrasts: [["GNN-TAT[BASIC]", "ConvLSTM[BASIC]"]]
factorial: {metric: pooled_r2, aggregate: lead_mean, n_perm: 20000}
fusion:
  base: [{model: ConvLSTM, features: BASIC}, {model: GNN-TAT, features: BASIC}]
  ensemble: {model: Late-Fusion, features: BASIC}
  assignment: purged
strata:
  elevation: {variable: elevation, edges: [1500, 2800], labels: [low, mid, high]}
  region: {path: region_labels.npy}
conclusions:
  pass: "the configuration extracts information beyond the per-cell climatology"
  fail: "no configuration exceeds the per-cell climatology on the purged block"
```

## Sections

| section | meaning |
|---|---|
| `data.observations` | `format: field` reads a variable from a NetCDF field with a `cutoff` month; `format: netcdf` reads a file written by `Observations.to_netcdf` |
| `data.forecasts` | one entry per run or per glob: `netcdf`, `long_table` (`origin_time, lead, cell, pred, target`) or `directory` (`SEED<n>/.../predictions.npy` with `targets.npy` beside it; origins are recovered from the observations) |
| `scoring` | `{origins: all}`, `{origins: [..]}`, `{origins: {start, stop}}` or `{purged: {start, stop}}`; the purged form also tells C0 how to build the fit set |
| `references` | any of `climatology`, `persistence`, `seasonal_naive`; the gate's reference is added when missing |
| `gate` | `metric` (`msess`, `pooled_r2`, `per_cell_nse`), `reference`, `threshold`, `min_seeds` |
| `folds` | the assignments C0 audits; `out_of_sample: true` marks the ones a score may be reported from |
| `contrasts` | the C2 family, as `model[features]` labels; every pair when omitted |
| `factorial` | `metric`, `aggregate` (`lead_mean`, `lead_peak`, `pooled`), `n_perm` |
| `fusion` | `base` configurations, optional `ensemble`, `assignment` (`purged`, `shuffled`, `contiguous`, `released` or a fold name) |
| `strata` | per stratum set, a `variable` with `edges` and `labels` (from the observations file or a `path`), or a `path` to a `.npy` of labels, one per cell |
| `conclusions` | free text: what will be concluded on PASS and on FAIL |

## Running it

```
anchorgate validate  declaration.yaml
anchorgate audit     declaration.yaml --out report/
anchorgate anchor    declaration.yaml --out report/ --categorical
anchorgate factorial declaration.yaml --out report/
anchorgate fusion    declaration.yaml --out report/ --assignment purged
anchorgate anomaly   declaration.yaml --out report/ --conformal
anchorgate spatial   declaration.yaml --out report/ --leakage
anchorgate run       declaration.yaml --out report/
```

`report/report.md` opens with the verdict table; `report/report.json` holds every number
for machines, with the SHA-256 of each input array and of the declaration.
