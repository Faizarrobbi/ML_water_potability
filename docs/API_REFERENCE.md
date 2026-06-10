# API Reference

Base URL: `http://localhost:5000` (local) or Container Apps URL (production)

## GET /

Service status.

**Response:**
```json
{
  "status": "running",
  "service": "water-potability-api",
  "model_name": "water_potability_model",
  "model_version": 5,
  "dataset_version": "2026-06-09_120000"
}
```

---

## GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": 5
}
```

Status values: `healthy` (model loaded), `degraded` (model not loaded).

---

## POST /predict

Predict water potability for a single sample.

**Request body:**
```json
{
  "ph": 7.0,
  "Hardness": 200.0,
  "Solids": 20000.0,
  "Chloramines": 7.5,
  "Sulfate": 330.0,
  "Conductivity": 400.0,
  "Organic_carbon": 14.0,
  "Trihalomethanes": 70.0,
  "Turbidity": 3.5
}
```

All fields are optional (null values are imputed by the pipeline).

**Response:**
```json
{
  "prediction": 1,
  "label": "Potable",
  "probability": 0.9123,
  "confidence": "High",
  "model_version": 5,
  "timestamp": "2026-06-09T19:00:00+07:00"
}
```

**Confidence levels:**
| Probability | Label |
|---|---|
| >= 0.9 | High |
| >= 0.7 | Medium |
| < 0.7 | Low |

---

## POST /predict-with-stats

Predict + enriched response with model metadata and cumulative statistics.

**Request body:** Same as `/predict`.

**Response:**
```json
{
  "prediction": 1,
  "label": "Potable",
  "probability": 0.9123,
  "confidence": "High",
  "model_version": 5,
  "timestamp": "2026-06-09T19:00:00+07:00",
  "input_timestamp": "2026-06-09T19:00:00+07:00",
  "model_name": "water_potability_model",
  "model_stage_or_alias": "Production",
  "mlflow_run_id": "abc123...",
  "dataset_version": "2026-06-09_120000",
  "prediction_stats": {
    "total_requests_since_start": 42,
    "potable_count": 30,
    "not_potable_count": 12,
    "avg_confidence": 0.8432,
    "current_model_version": 5,
    "daily_counts": {
      "2026-06-09": 42
    },
    "model_version_counts": {
      "5": 42
    }
  }
}
```

---

## GET /prediction-stats

Cumulative prediction statistics since container start.

**Response:**
```json
{
  "total_requests_since_start": 100,
  "potable_count": 65,
  "not_potable_count": 35,
  "potable_percentage": 65.0,
  "not_potable_percentage": 35.0,
  "avg_confidence": 0.8421,
  "current_model_version": 5,
  "daily_counts": {
    "2026-06-09": 80,
    "2026-06-10": 20
  },
  "model_version_counts": {
    "5": 100
  },
  "last_prediction_at": "2026-06-10T12:00:00Z"
}
```

> **Note:** Statistics are stored in-memory and reset on container restart. For production, replace with a database or Azure Table Storage.

---

## GET /model-info

Metadata about the currently loaded model.

**Response:**
```json
{
  "model_name": "water_potability_model",
  "model_version": 5,
  "model_stage_or_alias": "Production",
  "mlflow_run_id": "abc123...",
  "dataset_version": "2026-06-09_120000",
  "loaded_model_uri": "models:/water_potability_model/5",
  "api_started_timezone": "Asia/Jakarta"
}
```

---

## Input Schema

All fields are `Optional[float]`. Missing or null values are passed to the sklearn pipeline's imputer.

| Field | Description |
|---|---|
| `ph` | pH level of water |
| `Hardness` | Water hardness (mg/L) |
| `Solids` | Total dissolved solids (ppm) |
| `Chloramines` | Chloramine content (ppm) |
| `Sulfate` | Sulfate content (mg/L) |
| `Conductivity` | Electrical conductivity (μS/cm) |
| `Organic_carbon` | Total organic carbon (mg/L) |
| `Trihalomethanes` | Trihalomethane content (μg/L) |
| `Turbidity` | Water turbidity (NTU) |

## Error Responses

**Model not loaded (missing MLFLOW_TRACKING_URI):**
```json
{"error": "Model not loaded"}
```

**Validation error (invalid input types):**
```json
{
  "detail": [
    {
      "loc": ["body", "ph"],
      "msg": "Input should be a valid number",
      "type": "float_parsing"
    }
  ]
}
```
