# Churn Prediction Pipeline

**One model per lifecycle stage, because a four-month-old account and a four-year-old one churn for different reasons — and at four times the rate.**

[![CI](https://github.com/Rishabh-792/mlops-churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Rishabh-792/mlops-churn-prediction/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![models](https://img.shields.io/badge/models-CatBoost%20%C3%976-orange)

A configuration-driven churn pipeline: fetch → preprocess → segment → train → score.
Every model metric below is produced by `python -m pipelines.training_pipeline`
and committed to [`artifacts/metrics.json`](artifacts/metrics.json). CI retrains
on every push and **fails the build if the committed metrics stop reproducing**.

That gate compares `artifacts/metrics.json` against its committed self — it does
not parse this README, so the table below is transcribed by hand and can drift
from the artifact even while CI is green. The artifact is the source of truth;
if the two disagree, the artifact is right.

## Why segment first

The usual approach is one global model with `tenure` as a feature. That buries
the most important fact in this dataset:

| Segment | Tenure | Customers | Churn rate |
|---|---|---:|---:|
| `guest` | 0–4 months | 1,238 | **54.9%** |
| `casual` | 5–24 months | 1,972 | 33.0% |
| `power_user` | 25+ months | 3,833 | 14.0% |

A single model spends its capacity learning the tenure gradient. Splitting first
lets each model learn *within* a regime where the base rate is stable, and it
lets the business act differently on a 55%-risk cohort than on a 14% one.

Each segment then gets two models on disjoint feature views — an **activity**
view (spend, contract, payment behaviour) and a **profile** view (demographics,
subscribed services). Their probabilities are averaged. Six models total.

```
                    ┌── activity ──┐
   guest ───────────┤              ├── avg ──┐
                    └── profile ───┘         │
                    ┌── activity ──┐         │
   casual ──────────┤              ├── avg ──┼──► risk score ──► band ──► action
                    └── profile ───┘         │
                    ┌── activity ──┐         │
   power_user ──────┤              ├── avg ──┘
                    └── profile ───┘
```

## Results

Held-out test split, never seen during training or early stopping.
Regenerate with `python -m pipelines.training_pipeline`.

| Segment | View | ROC-AUC | PR-AUC | F1 | Precision | Recall | Test n | Base rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| power_user | activity | 0.812 | 0.356 | 0.422 | 0.286 | 0.806 | 767 | 0.141 |
| power_user | profile | 0.776 | 0.334 | 0.394 | 0.270 | 0.732 | 767 | 0.141 |
| casual | activity | 0.767 | 0.577 | 0.621 | 0.536 | 0.738 | 395 | 0.329 |
| casual | profile | 0.773 | 0.615 | 0.634 | 0.551 | 0.746 | 395 | 0.329 |
| guest | activity | 0.759 | 0.790 | 0.710 | 0.700 | 0.721 | 248 | 0.548 |
| guest | profile | 0.775 | 0.796 | 0.744 | 0.730 | 0.757 | 248 | 0.548 |
| **weighted** | | **0.783** | **0.494** | **0.526** | | | **1,410** | |

**What the weighted row is, and is not.** It is an `n_test`-weighted mean over
the six *per-view* models. There are **1,410 distinct held-out customers**
(767 + 395 + 248); each contributes to two views, so the six `n_test` values
sum to 2,820 while the evaluation set is half that. The weighting is
unaffected — both views of a segment carry equal weight, so the duplication
cancels — but the population size is 1,410.

**These are per-view metrics, not the ensemble's.** `prediction_pipeline`
averages the two view probabilities per segment (the diagram above); that
averaged ensemble is *not* separately scored in `artifacts/metrics.json`, so
no number here describes it. Scoring it is the first roadmap item below.

**Reading these honestly.** ROC-AUC is flattered by class imbalance, so PR-AUC
is reported alongside it and tracks each segment's base rate as it should
(0.36 at 14% positives, 0.80 at 55%). Precision on `power_user` is low by
design: `scale_pos_weight` trades precision for recall so the model surfaces
~80% of churners in the segment that is cheapest to retain and most expensive
to lose. If you need the other trade, move the decision threshold — the models
emit probabilities, not argmax labels.

**Do not read those probabilities as absolute risks.** `scale_pos_weight`
deliberately shifts them away from the base rate, which is decalibration by
construction: mean predicted risk on `power_user` is 0.43 against an actual
0.14. Brier scores are in the metrics artifact, but a Brier score is a proper
score, not evidence of calibration — on `power_user` it is worse than
predicting the base rate for everyone. Fit a per-segment isotonic or Platt
calibrator on the validation split before treating a score as a probability,
and note that the shared 0.70 `high` band is an absolute cut across segments
whose true rates span 0.14 to 0.55.

**Comparison point.** A single global model on this dataset is commonly
reported around 0.84 ROC-AUC, but that figure is uncited here and is not
directly comparable: it is a full-population AUC, while the 0.783 above is a
weighted mean of within-segment AUCs. Segmentation restricts the range of
tenure — the strongest single predictor — inside each segment, which lowers
within-segment AUC while buying per-segment actionability. Training one global
model on the same split would make this an internal, reproducible comparison;
that is a roadmap item, not a claim made here.

## Quickstart

```bash
pip install -r requirements-dev.txt

python -m scripts.download_data              # fetch + verify SHA-256
python -m pipelines.preprocessing_pipeline   # clean, engineer, validate
python -m pipelines.training_pipeline        # train 6 models, write metrics
python -m pipelines.prediction_pipeline      # score every customer

pytest -q                                    # 33 tests, 85% cov, no network
mlflow ui                                    # inspect runs
```

Or the whole thing in a container:

```bash
docker build -t churn-pipeline .
docker run --rm churn-pipeline
```

## Reproducibility

The dataset is **not committed**. `scripts/download_data.py` fetches it and
verifies it against a SHA-256 pinned in `configs/config.json`; a mismatch
aborts rather than training on unverified input.

```
dataset   IBM Telco Customer Churn, 7,043 customers x 21 columns
sha256    16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91
seed      42 (splits, CatBoost)
tracking  MLflow, SQLite-backed (mlflow.db) — no server required
```

## Architecture

```
configs/config.json          One file describes a run: schema, segment bounds,
                             split sizes, CatBoost params, output paths.
scripts/download_data.py     Checksum-verified acquisition.
pipelines/
  preprocessing_pipeline.py  Validate schema → clean → engineer → parquet.
  feature_pipeline.py        Segment, then build both feature views.
  training_pipeline.py       Train 6 models, log to MLflow, write metrics.json.
  prediction_pipeline.py     Route, score, band, recommend action.
utils/
  settings_manager.py        JSON → typed dataclasses. No stringly-typed config.
  feature_builders.py        Segmentation and the two feature views.
  model_training_utils.py    Three-way split, class weighting, evaluation.
  model_tune_utils.py        Optuna search under precision/recall constraints.
  prediction_utils.py        Ensemble loading, routing, risk banding.
  pipeline_errors.py         Coded exception registry, JSON-serialisable.
deployment/
  inference.py               SageMaker handler reusing the same routing logic.
  deploy_sagemaker.py        Reference provisioning script (never run live).
```

Design choices worth calling out:

- **Typed configuration.** `SettingsManager` parses JSON into dataclasses, so a
  malformed config fails at load with a registered error code rather than as an
  `AttributeError` thirty seconds into training.
- **Coded errors.** Every failure carries a `SYS-xxx` code and serialises to
  JSON for structured cloud logging.
- **Three-way split.** Models early-stop on validation and are scored on a test
  split they have never seen. Only test metrics reach the artifact.
- **Unscorable is reported, not guessed.** A segment with too few rows to train
  is skipped, and its customers come back unscored rather than being routed to
  another segment's model. A test asserts this.

## Testing

```bash
pytest -q          # 33 tests, 85% line coverage
ruff check .       # lint
```

The suite runs on a synthetic frame carrying the same schema and the same
defects as the real file (`TotalCharges` as text, blanks for unbilled
accounts), so it needs no network and no download. It covers segment boundary
conditions, schema rejection, the derived features, split disjointness, risk
banding, and one full train-and-score cycle.

## CI/CD

- **lint-and-test** — ruff + pytest on Python 3.10/3.11/3.12, with coverage.
- **train** — fetches the real dataset, runs the full pipeline, and enforces a
  quality gate: floors on ROC-AUC/PR-AUC/F1, plus a check that
  `artifacts/metrics.json` still reproduces within tolerance. This is what
  stops the README from drifting away from reality.
- **docker** — builds the image and runs the test suite inside it.

## Deployment

`deployment/inference.py` is a SageMaker handler that reuses `EnsembleModels`,
so online and offline scoring cannot drift apart. **It has not been deployed to
a live endpoint** — it is reference code, and `deployment/deploy_sagemaker.py`
sketches the provisioning path. Treat the AWS portion as a design artifact rather than a
running system.

## Status and roadmap

**Complete at v1.0.0, not under active development.** A reference pipeline
meant to be read and run, not a maintained dependency. CI is green on Python
3.10/3.11/3.12 and retrains on every push. The items below are known gaps
rather than planned work — the ensemble being unscored and the Optuna module
being unwired are real limitations of what is here, not a queue.

- **Score the ensemble.** `metrics.json` currently reports the six per-view
  models; the averaged ensemble that actually scores customers is unmeasured.
- **Calibrate.** Per-segment isotonic/Platt on the validation split, with
  reliability curves in the artifact, so the risk bands mean what they say.
- **A global-model baseline** on the same split, to replace the uncited 0.84
  literature figure with an internal, reproducible comparison.
- `utils/model_tune_utils.py` (Optuna) is written but **unwired** — no pipeline
  or test imports it, and `tuning.enabled` is read by nothing. Wire it or
  delete it and drop the `optuna` dependency.
- Drift monitoring on the scored population.

## License

MIT — see [LICENSE](LICENSE).
