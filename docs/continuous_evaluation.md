# Continuous evaluation: what it would cost

A protocol that scores saved predictions in a minute can run every month, when the new
observation field arrives. This page estimates what that costs on one cloud provider
(Amazon Web Services, region us-east-1), at the list prices published on 21 November 2026,
for a monthly precipitation benchmark over a 61 by 65 cell box with a map dashboard. The
numbers are orders of magnitude; the assumptions and the unit prices are listed so that
they can be redone for another region, provider or grid.

## Three scales

| Scale | What it includes | USD per month | USD per year | Set-up (person-days) |
|---|---|---|---|---|
| A. Minimal serverless | one scheduled container task (download, crop, a cheap statistical model, `anchorgate run`, maps and report to object storage), pre-rendered tiles, static map dashboard behind a CDN, logs and two alarms | 1 to 8 | 15 to 100 | 4 to 6 |
| B. A plus deep models | monthly CPU inference of the deposited deep models, quarterly retraining on a spot GPU instance, statistical downscaling of the lead maps to 1 km, a dynamic tile server, a benchmark history table | 25 to 55 (40 to 70 with a relational database) | 300 to 700 | 15 to 25 |
| C. Production | B plus an always-on API behind a load balancer, a web application firewall, DNS and certificates, backups, monitoring and alarms, a quarterly campaign of three seeds by three architectures | 110 to 200 (up to about 260 with a NAT gateway, a multi-zone database and synthetic canaries) | 1,300 to 2,400 | 30 to 45 |

The scientific computation is cheap at every scale: the monthly pipeline task costs a few
cents, deep-model inference about 0.12 USD, downscaling about 0.23 USD, and quarterly
retraining 5 to 30 USD per quarter. What separates the scales is the infrastructure that
stays switched on for the 730 hours of a month (tile server, API, database, load balancer,
firewall), which is 70 to 90 % of the cost of B and C.

## Zero-cost alternative

Scale A fits inside a public continuous-integration service with static hosting: a
monthly cron workflow on GitHub Actions (unlimited minutes on public repositories),
GitHub Pages for the dashboard, and the monthly reports and cloud-optimised GeoTIFFs as
release assets or in object storage with free egress (Cloudflare R2, 10 GB free). What is
lost against the cloud provider: native alarms (use the workflow's failure notification),
a size limit per repository (use releases for the archive), and a six-hour limit per job,
which this pipeline does not approach.

## Load assumptions

| Assumption | Value |
|---|---|
| Runs | one per month, plus one or two manual retries |
| Input | one global monthly CHIRPS file; the regional crop is under 1 MB |
| Output at 5 km | twelve lead maps of 3,965 cells, scores, verdicts and a report; under 5 MB per month |
| Output at 1 km (B and C) | twelve maps of about 400,000 cells per model; about 60 MB per month |
| Dashboard traffic | 200 to 2,000 visits per month, under 50 GB of egress, inside the CDN free tier |
| Deep-model inference | two models of 98K and 148K parameters at twelve leads; seconds on CPU, a 30-minute task reserved for the container cold start |
| Retraining | 1 to 3 GPU-hours per seed and architecture on a spot g6.xlarge |

## Unit prices used

Prices read on the provider's own pricing pages are marked [provider]; prices that could
only be confirmed on third-party trackers because the provider's table loads with
JavaScript are marked [tracker] and may deviate by 10 to 20 %.

| Service | Price | Status |
|---|---|---|
| Lambda | 0.20 USD per million requests; 0.0000166667 USD per GB-second; 1 million requests and 400,000 GB-seconds free per month | [provider] |
| Fargate (x86) | 0.04048 USD per vCPU-hour; 0.004446 USD per GB-hour | [provider] |
| S3 Standard | 0.023 USD per GB-month; 0.005 USD per 1,000 PUT; 0.0004 USD per 1,000 GET | [tracker] / [provider] |
| CloudFront | 1 TB of egress and 10 million requests free per month; 0.085 USD per GB beyond | [provider] |
| EventBridge Scheduler | 14 million invocations free per month | [provider] |
| CloudWatch | 5 GB of logs and 10 alarms free; 0.50 USD per GB and 0.10 USD per alarm beyond | [provider] |
| App Runner (tile server) | 0.064 USD per vCPU-hour active; 0.007 USD per GB-hour provisioned | [provider] |
| Application Load Balancer | 0.0225 USD per hour plus 0.008 USD per capacity-unit-hour | [provider] |
| RDS PostgreSQL db.t4g.micro | 0.016 USD per hour; 0.115 USD per GB-month of storage | [tracker] |
| DynamoDB on demand | 0.625 USD per million writes; 0.125 USD per million reads; 25 GB free | [provider] |
| WAF | 5 USD per web ACL; 1 USD per managed rule group; 0.60 USD per million requests | [provider] |
| Route 53 | 0.50 USD per hosted zone; 0.40 USD per million queries | [provider] |
| ECR | 0.10 USD per GB-month | [provider] |
| EC2 g6.xlarge (one L4 GPU) | 0.805 USD per hour on demand; 0.35 to 0.60 USD per hour spot, depending on the day and the tracker | [tracker] |
| GitHub Actions | free on public repositories; 2,000 minutes per month included on private ones | [provider] |

## What a continuous benchmark needs at any scale

1. An immutable report per month: forecasts, the month's observation, per-cell and
   aggregate scores, verdicts, and a manifest with SHA-256 hashes of inputs and outputs,
   the package version, the container digest and the seed.
2. A declaration committed before the month's data arrive. A change of declaration opens a
   new series in the leaderboard instead of rewriting the old one.
3. A leaderboard whose first column is the climatology: one row per model, declaration
   version, lead and month, aggregated over moving windows of 12 and 36 months.
4. A frozen reference: the per-cell climatology computed once per declaration version and
   stored with its hash. A re-published observation month is recorded as a new truth and
   re-scored without deleting the earlier score.
5. An alarm when the month is not published within the expected delay after the
   observation field appears.
6. One command that regenerates any month's report on a laptop from the archived inputs.

The infrastructure cost of these six points is cents. Their cost is engineering: about two
person-days at scale A, four at B and six at C, included in the set-up figures above.

## Caveats

- Spot prices change daily; the g6.xlarge spot price varied between 0.35 and 0.60 USD per
  hour on the same day across trackers.
- If dashboard traffic exceeds 1 TB per month, CDN egress stops being free; at 5 TB it
  would be about 340 USD per month, the only item able to change the order of magnitude.
- Local taxes on digital services are not included. CHIRPS, SRTM and ERA5 are free.
