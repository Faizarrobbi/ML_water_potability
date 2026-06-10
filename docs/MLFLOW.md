# MLflow — MLOps Tracking & Registry

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    MLflow Tracking Server                 │
│  (local: http://127.0.0.1:5000  |  cloud: Azure ML URI)  │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐   │
│  │ Backend Store│  │ Artifact Store│  │ Model Registry │   │
│  │ (SQLite/DB)  │  │ (filesystem/ │  │ (stage-based)  │   │
│  │              │  │  Blob/S3)    │  │                │   │
│  └─────────────┘  └──────────────┘  └────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

## Components

### Tracking Server
The central entry point. Receives logged params, metrics, artifacts, and model data.

- **Local**: `http://127.0.0.1:5000` (used by legacy `modeling/ml_model_dev.py`)
- **Cloud**: URI obtained from Azure ML Workspace (used by modern `train.py` and `app.py`)

### Backend Store
Persists experiment metadata: run names, params, metrics, tags, timestamps.

- **Local**: `mlflow.db` (SQLite file in project root)
- **Cloud**: Azure ML Workspace managed backend

### Artifact Store
Persists large binary artifacts: model pickles, plots, metadata JSON files.

- **Local**: `mlruns/` directory
- **Cloud**: Azure Blob Storage container (typically the workspace's linked storage)

### Model Registry
Manages model lifecycle through stages: None → Staging → Production → Archived.

- **Local**: `mlruns/models/<model_name>/` (filesystem-based)
- **Cloud**: Azure ML Model Registry (UI available in Azure Portal)

## How This Project Uses MLflow

### Training (`train.py`)
```python
mlflow.set_tracking_uri(uri)            # from env or Azure ML
mlflow.set_experiment("water-potability-mlops")
mlflow.start_run(run_name="train_<version>")
    mlflow.log_param("classifier_type", "random_forest")
    mlflow.log_metric("test_f1_score", 0.87)
    mlflow.log_artifact("outputs/model_metadata.json")
    mlflow.sklearn.log_model(model, artifact_path="model",
                             registered_model_name="water_potability_model")
mlflow.end_run()
```

### Serving (`app.py`)
```python
client = mlflow.MlflowClient()
latest = client.get_latest_versions("water_potability_model",
                                     stages=["Production"])
model_uri = f"models:/water_potability_model/{latest[0].version}"
pipeline = mlflow.sklearn.load_model(model_uri)
```

## Local vs Cloud

| Aspect | Local (`mlflow server`) | Cloud (Azure ML) |
|---|---|---|
| Start | `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns` | Already running, just set the URI |
| Backend | `mlflow.db` (SQLite) | Azure-managed PostgreSQL |
| Artifacts | `mlruns/` | Azure Blob container (linked to workspace) |
| Registry | Local filesystem | Azure ML Registry |
| Authentication | None | Azure AD / service principal |
| Suitability | Development, experimentation | Production, CI/CD, team collaboration |

## Model Stages & Promotion

Models move through stages:
1. **None** — freshly registered, unassigned
2. **Staging** — validated, ready for pre-production
3. **Production** — actively serving via API
4. **Archived** — superseded by newer version

The training pipeline (`train.py`) includes automatic promotion logic:
- After training, compares the current `test_f1_score` against the existing Production model
- Promotes to Production if the metric is >= current production model
- Archives the previous Production version

## Key Commands

```bash
# Start local MLflow server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000

# View experiments
mlflow experiments list

# Transition model stage
mlflow models --version <version> stage Production

# Fetch tracking URI from Azure ML
az ml workspace show \
  --resource-group <rg> \
  --name <workspace> \
  --query mlflow_tracking_uri -o tsv
```

## Common Issues

1. **Model not found at startup**: Ensure a model version exists in the Production stage.
   ```bash
   mlflow models transition --name water_potability_model --version <n> --stage Production
   ```

2. **MLFLOW_TRACKING_URI not set**: The API and training scripts require this to load/save models.

3. **Azure ML extension not installed**:
   ```bash
   az extension add --name ml -y
   ```

4. **Registered model name mismatch**: `train.py` registers as `water_potability_model`, `app.py` loads the same name. Both use env `MODEL_NAME` for override.
