# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GitHub Actions (CI/CD)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────────┐   │
│  │ Validate │→ │  Upload  │→ │  Train   │→ │ Build & Deploy     │   │
│  │  config  │  │ dataset  │  │  model   │  │ ACR → Container App│   │
│  └──────────┘  └──────────┘  └────┬─────┘  └────────────────────┘   │
│                                    │                                 │
└────────────────────────────────────┼─────────────────────────────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                ▼                    ▼                    ▼
        ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
        │ Azure Blob   │    │ Azure ML      │    │ Azure        │
        │ Storage      │    │ MLflow        │    │ Container    │
        │ datasets/    │    │ Tracking +    │    │ Registry     │
        │              │    │ Registry      │    │ (ACR)        │
        └──────────────┘    └───────┬───────┘    └──────┬───────┘
                                    │                   │
                                    ▼                   ▼
                            ┌───────────────┐    ┌──────────────┐
                            │ FastAPI       │    │ Azure        │
                            │ App (serving) │    │ Container    │
                            │ loaded model  │    │ Apps         │
                            │ from registry │    │ (deployment) │
                            └───────────────┘    └──────────────┘
```

## Component Design

### Training Pipeline (`train.py`)
```
config/train.yaml  ──┐
                     ├──→ TrainConfig ──→ PipelineBuilder ──→ sklearn Pipeline
config/hyper        │                      │
parameters.yaml  ───┘                      │
                                           ▼
                                    GridSearchCV (optional)
                                           │
                                           ▼
                                     Evaluate → Metrics
                                           │
                                           ▼
                                    MLflow Logging
                                           │
                                           ▼
                                    Model Registry Promotion
```

### Serving API (`app.py`)
```
Startup:
  MLflow Client ──→ get_latest_versions("water_potability_model", "Production")
                         │
                         ▼
                  mlflow.sklearn.load_model(model_uri)
                         │
                         ▼
                  sklearn_pipeline (loaded in memory)

Requests:
  POST /predict ──→ _predict_item() ──→ pipeline.predict() + predict_proba()
                         │
                         ▼
                  Response: prediction, probability, confidence, model_version, timestamp
                         │
                         ▼
                  _update_stats() → _prediction_stats (in-memory)
                  _prediction_logger.log_prediction() → Blob / logger
```

### Dataset Flow (`dataset_manager.py`)
```
Upload:   local CSV → Azure Blob → <version>/water_potability.csv + latest_water_potability.csv
Download: Azure Blob → local CSV (specify version or "latest")
List:     Scan blob prefix → extract version timestamps
```

### Prediction Logging (`prediction_logger.py`)
```
Factory → BlobPredictionLogger (if Azure credentials available)
       → LoggingPredictionLogger (fallback, just logs)

Blob Path: predictions/<YYYY>/<MM>/<DD>/predictions.jsonl
Format:    JSON Lines (one JSON object per line)
```

## File Map

| File | Role |
|---|---|
| `train.py` | YAML-driven training with grid search + MLflow + promotion |
| `app.py` | FastAPI serving 6 endpoints, stats, prediction logging |
| `dataset_manager.py` | Azure Blob dataset versioning abstraction |
| `prediction_logger.py` | Prediction persistence (Blob or log) |
| `config/train.yaml` | Training pipeline configuration |
| `config/hyperparameters.yaml` | Hyperparameter search spaces |
| `Dockerfile` | Container image for API |
| `docker-compose.yml` | Local dev: API + MLflow server + training |
| `.github/workflows/mlops.yml` | CI/CD pipeline |

## Legacy Code (preserved, not modified)

| Path | Purpose |
|---|---|
| `modeling/ml_model_dev.py` | Original grid search training (hardcoded local MLflow) |
| `modeling/data_utils.py` | Dataset loading utilities |
| `modeling/eda.py` | Exploratory data analysis |
| `modeling/ml_model_test.py` | Test script for locally saved models |
| `config/train_config.yaml` | Legacy training config (used by `ml_model_dev.py`) |
| `get_model_for_production.py` | Local model copy script |
| `test_post_request.py` | HTTP test script |
| `kubernetes_deployment/` | Local Kind cluster manifests |
