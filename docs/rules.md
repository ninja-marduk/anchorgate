# The rules

Each component applies one rule and prints it next to its verdict. The rules are the
protocol; the numbers are evidence for or against them.

## C0, fold audit

A score reported as out of sample must come from an assignment with no held-out target
period in its fit set.

Consecutive forecast origins share target periods: origin `o` at lead `h + 1` verifies
against the same month as origin `o + 1` at lead `h`. Folds of whole origins, random or
contiguous, therefore keep most held-out targets inside the fit set. The only clean
holdout is a contiguous block of origins with an embargo of `horizon - 1` origins on each
side. The audit also counts the independent units behind the origins (distinct target
periods, disjoint windows), so that no later test writes an `N` the record does not hold.

## C1, anchoring

A configuration passes when the lower bound of the block-bootstrap interval (level
`level`, blocks of `block` origins, `n_boot` resamples, declared in the gate) of the
mean over at least `min_seeds` seeds of the skill score against the reference exceeds
the threshold; it fails when the upper bound does not; an interval that straddles the
threshold is INSUFFICIENT, and fewer seeds cannot pass. The same interval per lead gives
the skill horizon: the last lead of the unbroken run, from the shortest lead, whose lower
bound clears the threshold.

Every forecast and every reference (per-cell climatology, persistence, seasonal naive)
is scored on identical targets. Pooled R2 and per-cell NSE are both reported, because a
climatology scores high on the first and above zero on the second through the seasonal
cycle it carries; the gate uses the mean-squared-error skill score against the
climatology.

## C2, seed-resolved factorial

A difference between configurations is claimed only when its paired, within-seed
difference survives Holm correction over the declared family; otherwise the ranking is
unresolved at this number of seeds and the minimum detectable effect is reported.
Single-seed configurations are listed, not compared.

The design is a randomised complete block with the seed as block. The dispersion that
matters is that of the paired difference, never the marginal spread of either arm. The
inflation from quoting the best seed and the Nemenyi critical difference at the number
of disjoint windows are printed next to every contrast.

## C3, fusion decomposition

An ensemble's gain is claimed only over its recalibrated best base learner, chosen on the
fit origins, on an assignment with no target-period overlap and at least `min_seeds`
paired runs; the mean combination gain and the mean total gain over the raw best must
both be positive, and both are printed with their spread over runs.

The gain over the raw best base learner splits into a calibration term (what an affine
recalibration of the best single learner already delivers) and a combination term (what
combining adds on top). The base learner is chosen on the fit origins; choosing it on the
scored origins is a selection advantage no forecaster has, and its size is printed.

## C4, anomaly skill

A configuration shows anomaly skill when the lower bound of the block-bootstrap
interval of the seed mean of its per-cell temporal anomaly correlation (median over
cells) is positive; it fails when the upper bound is not. The MSESS against climatology
is reported next to it (component 1 already gates on it) and the pooled anomaly
correlation is reported but never used.

Anomalies are taken against the training climatology, on which that climatology scores
exactly zero. The correlation is computed per cell over time and per origin over space;
pooling both axes conflates a static spatial pattern with forecast skill.

## C5, spatial diagnostics

A spatial prior is credited only in the cells where the configuration has the lowest
seed-mean error; the component passes when one configuration wins a majority of cells and
more cells than every untrained reference.

The winner map names the cells. Skill by strata, the collapse diagnostic (a forecast of
each cell's mean has a standard-deviation ratio near zero) and the graph audit (edge
lengths, elevation contrasts, degree, and whether the correlation term saw the evaluation
period) are reported alongside.

## The declaration

Every verdict above is computed under a declaration written before scoring, and every
report prints its hash. A declaration edited after scoring produces a visibly different
report. See [the declaration](declaration.md).
