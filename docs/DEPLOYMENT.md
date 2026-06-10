# Deployment

## Local Development (Docker Compose)

```bash
# 1. Clone
git clone <repo>
cd ML_water_potability

# 2. Configure
cp .env.example .env
# Edit .env with your settings

# 3. Start services
docker compose up -d

# 4. (Optional) Train model
docker compose --profile train run train

# 5. Access
curl http://localhost:5000/health
```

Services:
| Service | Port | URL |
|---|---|---|
| FastAPI | 5000 | http://localhost:5000 |
| MLflow | 5001 | http://localhost:5001 |

## Production (Azure Container Apps — CI/CD)

Triggered automatically on push to `main` when relevant files change. The GitHub Actions workflow:

1. Validates Python syntax and YAML config
2. Uploads dataset to Azure Blob Storage (versioned)
3. Trains model with config-driven pipeline
4. Logs to Azure ML MLflow with model registration
5. Promotes best model to Production stage
6. Builds Docker image → pushes to ACR (SHA + version tags)
7. Deploys to Azure Container Apps with env vars
8. Runs health check against deployed endpoint

### Prerequisites

Azure resources (see [README.md](../README.md)):
- Storage account + `datasets` container
- Azure ML Workspace
- Azure Container Registry
- Container App Environment + Container App

GitHub Secrets:
- `AZURE_CREDENTIALS` — service principal JSON
- `AZURE_RESOURCE_GROUP` — e.g., `mlops-rg`
- `AZURE_STORAGE_ACCOUNT` — e.g., `waterpotabilitystorage`
- `AZURE_ACR_NAME` — e.g., `waterpotabilityacr`
- `AZURE_CONTAINER_APP_NAME` — e.g., `water-potability-api`
- `AZURE_CONTAINER_APP_ENV` — e.g., `water-potability-env`
- `AZURE_ML_WORKSPACE` — e.g., `water-potability-mlw`

### Rollback

Container Apps supports revision management:
```bash
az containerapp revision list \
  --name water-potability-api \
  --resource-group mlops-rg \
  --query "[].{Name:name, Active:active}" -o table

az containerapp revision activate \
  --name water-potability-api \
  --resource-group mlops-rg \
  --revision <previous-revision>
```

Docker images are tagged with both the commit SHA and dataset version for easy rollback:
```
waterpotabilityacr.azurecr.io/water-potability-api:<sha>
waterpotabilityacr.azurecr.io/water-potability-api:<dataset-version>
```

## Manual ACR Deployment

```bash
az acr login --name waterpotabilityacr

docker build -t waterpotabilityacr.azurecr.io/water-potability-api:latest .
docker push waterpotabilityacr.azurecr.io/water-potability-api:latest

az containerapp update \
  --name water-potability-api \
  --resource-group mlops-rg \
  --image waterpotabilityacr.azurecr.io/water-potability-api:latest
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `MLFLOW_TRACKING_URI` | Yes (cloud) / No (local) | — | MLflow server URI |
| `MODEL_NAME` | No | `water_potability_model` | Registered model name |
| `MODEL_STAGE_OR_ALIAS` | No | `Production` | Model stage to load |
| `DATASET_VERSION` | No | `unknown` | Deployed dataset version |
| `AZURE_STORAGE_CONNECTION_STRING` | For blob features | — | Blob connection string |
| `AZURE_STORAGE_ACCOUNT` | For blob features | — | Blob account name |
| `AZURE_STORAGE_CONTAINER` | No | `datasets` | Blob container name |
