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
- Markdown and JSON reports; command line interface (`validate`, `audit`, `anchor`).
