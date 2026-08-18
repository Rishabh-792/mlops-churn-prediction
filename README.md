# Churn Prediction Pipeline

**One model per lifecycle stage, because a four-month-old account and a four-year-old one churn for different reasons — and at four times the rate.**

[![CI](https://github.com/Rishabh-792/mlops-churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Rishabh-792/mlops-churn-prediction/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![models](https://img.shields.io/badge/models-CatBoost%20%C3%976-orange)

A configuration-driven churn pipeline: fetch → preprocess → segment → train → score.
Every number below is produced by `python -m pipelines.training_pipeline` and
committed to [`artifacts/metrics.json`](artifacts/metrics.json). CI retrains on
every push and **fails the build if the committed metrics stop reproducing**, so
the results in this README cannot quietly drift away from the code.

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
| casual | activity | 0.767 | 0.578 | 0.621 | 0.536 | 0.739 | 395 | 0.329 |
| casual | profile | 0.773 | 0.615 | 0.634 | 0.551 | 0.746 | 395 | 0.329 |
| guest | activity | 0.759 | 0.790 | 0.710 | 0.700 | 0.721 | 248 | 0.548 |
| guest | profile | 0.775 | 0.796 | 0.744 | 0.731 | 0.757 | 248 | 0.548 |
| **weighted** | | **0.783** | **0.494** | **0.526** | | | **2,820** | |

**Reading these honestly.** ROC-AUC is flattered by class imbalance, so PR-AUC
is reported alongside it and tracks each segment's base rate as it should
(0.36 at 14% positives, 0.80 at 55%). Precision on `power_user` is low by
design: `scale_pos_weight` trades precision for recall so the model surfaces
~80% of churners in the segment that is cheapest to retain and most expensive
to lose. If you need the other trade, move the decision threshold — the
probabilities are calibrated (Brier scores are in the metrics artifact), not
argmaxed.

**Comparison point.** Published single-model baselines on this dataset land
around 0.84 ROC-AUC. The segmented ensemble scores lower on that metric
precisely because segmentation removes tenure — the strongest single predictor
— from within-segment variance. The trade buys per-segment calibration and
actionability. Reporting only the number that flatters the design would be the
easier choice and the wrong one.

## Quickstart

```bash
pip install -r requirements-dev.txt

python -m scripts.download_data              # fetch + verify SHA-256
python -m pipelines.preprocessing_pipeline   # clean, engineer, validate
python -m pipelines.training_pipeline        # train 6 models, write metrics
python -m pipelines.prediction_pipeline      # score every customer

pytest -q                                    # 33 tests, no network needed
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
pytest -q          # 33 tests
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

## Roadmap

- Enable the Optuna search in CI (`tuning.enabled`); it is implemented and
  tested but not part of the committed run.
- Drift monitoring on the scored population.
- Calibration curves per segment in the metrics artifact.

## License

MIT — see [LICENSE](LICENSE).
