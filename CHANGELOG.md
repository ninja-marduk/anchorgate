# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog, and the project follows semantic versioning. The protocol itself
("AnchorGate v1.0", the set of rules) is versioned separately from this package.

## [Unreleased]

### Added
- Data contract: `Forecast`, `Observations`, `Benchmark` with validation of identical
  targets, target-observation consistency and training-period exclusion.
- Loaders: in-memory arrays, long tables, NetCDF, and the `SEED<n>/.../predictions.npy`
  directory layout.
- References built from observations on the training period only: per-cell period
  climatology, persistence, seasonal naive, and training-period terciles.
- Component 0, fold audit: target-period overlap of any fold assignment, effective
  independent units, and purged holdout construction with an embargo of `horizon - 1`.
- Component 1, anchoring: pooled R2, per-cell NSE, MSESS, RMSE, MAE, bias, by lead and
  inside a mask; tercile categorical scores; the gate on the seed mean of MSESS.
- Declaration file with SHA-256 hash carried into every result.
- Component 2, seed-resolved factorial: responses per run (lead mean, lead peak or
  pooled), randomised complete block ANOVA with the seed as block, within-block
  permutation test, paired contrasts with Holm correction over a declared family and
  Cliff's delta, best-of-seeds inflation, minimum detectable effect, effective units,
  Friedman and Nemenyi with the critical difference recomputed at the number of
  disjoint windows; Mann-Whitney group contrasts for tables whose units are not seeds.
- Component 3, calibration versus combination: out-of-fold affine recalibration and
  combination on any fold assignment, the best base learner chosen on the fit origins,
  oracle picks as the selection advantage, the in-sample gap, a supplied ensemble scored
  for reference, and the ladder of rungs scored on the same origins.
- Component 4, anomaly skill: per-cell temporal and per-origin pattern anomaly
  correlations against the training climatology, the pooled correlation as the inflated
  comparison, anomaly R2 and standard-deviation ratio, MSESS, and split-conformal
  intervals by origin truncated at zero with conditional coverage by stratum.
- Component 5, spatial diagnostics: per-cell winner map, skill by strata, collapse
  diagnostic (mean-field correlation, s.d. ratio, variance explained by the cell mean),
  graph audit of an adjacency (edge length, elevation contrast, degree) and the
  training-only versus full-record correlation leakage.
- Declaration sections `contrasts`, `factorial`, `fusion` and `strata`.
- Every result records the parameters it used and the deviations from the declaration; the
  declaration carries a file hash and a design hash (sections other than `data`), and the report
  prints the declared conclusions next to the verdicts.
- Markdown and JSON reports; command line interface (`validate`, `audit`, `anchor`,
  `factorial`, `fusion`, `anomaly`, `spatial`, `run`).
