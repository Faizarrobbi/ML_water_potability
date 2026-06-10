# Dataset Management

## Azure Blob Storage Layout

```
datasets/
├── 2026-06-09_120000/
│   └── water_potability.csv
├── 2026-06-08_093000/
│   └── water_potability.csv
├── latest_water_potability.csv          ← alias (always points to latest)
└── predictions/
    └── 2026/06/09/
        └── predictions.jsonl            ← logged predictions (JSON Lines)
```

## DatasetManager API

```python
from dataset_manager import DatasetManager

dm = DatasetManager()

# Upload local file with auto-versioning
version = dm.upload_dataset("dataset/water_potability.csv")
print(version)  # e.g., "2026-06-09_120000"

# Upload with explicit version
dm.upload_dataset("dataset/water_potability.csv", version="experiment_v2")

# Download specific version
path = dm.download_dataset("2026-06-09_120000")
path = dm.download_dataset("latest")  # alias

# List all versions
versions = dm.list_versions()
print(versions)  # ["2026-06-09_120000", "2026-06-08_093000", ...]

# Get latest version string
latest_ver = dm.get_latest_version()

# Download latest
path, version = dm.download_latest()
```

## Connection

The DatasetManager auto-detects Azure credentials from environment:

1. `AZURE_STORAGE_CONNECTION_STRING` — connection string auth
2. `AZURE_STORAGE_ACCOUNT` — managed identity / DefaultAzureCredential auth
3. None of the above → local filesystem fallback

## Dataset Versioning in Training

In `config/train.yaml`:
```yaml
use_blob_dataset: true   # downloads latest from Blob before training
file_csv: dataset/water_potability.csv  # fallback if blob unavailable
```

When `use_blob_dataset: true`, the training pipeline:
1. Connects to Azure Blob via DatasetManager
2. Downloads the latest version to `dataset/water_potability.csv`
3. Sets the dataset version from the blob version string

## CI/CD Dataset Upload

Every push to `main` triggers dataset upload:
- Versioned: `<timestamp>/water_potability.csv`
- Latest alias: `latest_water_potability.csv` (overwritten)
