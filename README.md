# ENGIE Wind Power Prediction (Docker + FastAPI)

End-to-end pipeline to **train** a DNN model for wind-turbine active power prediction and **serve** it through a FastAPI inference service using Docker Compose.

---

## Project layout

```text
.
├── training/
│   ├── train.py              # Training pipeline (prep + DNN training + save artifacts)
│   ├── data/                 # Input CSVs (engie_X.csv, engie_Y.csv)
│   ├── requirements.txt      # Python deps for training
│   └── Dockerfile            # Training image
├── inference/
│   ├── api/
│   │   ├── main.py           # FastAPI service (/health, /schema, /predict)
│   │   ├── requirements.txt  # Python deps for API
│   │   └── tests/            # API tests (pytest)
│   ├── Dockerfile            # Inference image (FastAPI + uvicorn)
│   └── models/               # Local artifacts directory (optional, for non-Docker runs)
├── docker-compose.yml        # Orchestration: training + inference + shared models volume
└── README.md
```

---

## Prerequisites

- Docker + Docker Compose
- (Optional) Python 3.12 for local execution

---

## Quick start (Docker Compose)

### 1) Train the model (writes artifacts to the shared volume)

```bash
docker compose up --build training
```

When the container finishes, it should exit with code `0` and the artifacts are stored in the **Docker volume** `models`.

### 2) Start the inference API

```bash
docker compose up --build inference
```

API will be available on:

- http://localhost:8000
- Swagger UI: http://localhost:8000/docs

### 3) Health check

```bash
curl http://localhost:8000/health
```

---

## Inference API

Base URL: `http://localhost:8000`

### Endpoints

- `GET /health` — service status + model metadata
- `GET /schema` — expected request format + required features
- `POST /predict` — run predictions

### Example request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "rows": [
      {
        "MAC_CODE": "WT2",
        "Date_time": 2,
        "Pitch_angle": 85.3,
        "Pitch_angle_std": 0.08,
        "Hub_temperature": 8.5,
        "Hub_temperature_std": 0.12,
        "Generator_converter_speed_std": 0.03,
        "Generator_speed": 1450.8,
        "Generator_speed_std": 18.7,
        "Generator_bearing_1_temperature": 48.9,
        "Generator_bearing_1_temperature_std": 0.95,
        "Generator_bearing_2_temperature_std": 0.82,
        "Generator_stator_temperature": 42.1,
        "Generator_stator_temperature_std": 1.4,
        "Gearbox_bearing_1_temperature": 54.7,
        "Gearbox_bearing_1_temperature_std": 1.1,
        "Gearbox_bearing_2_temperature_std": 0.92,
        "Gearbox_inlet_temperature": 50.2,
        "Gearbox_inlet_temperature_std": 0.68,
        "Gearbox_oil_sump_temperature": 57.8,
        "Gearbox_oil_sump_temperature_std": 0.83,
        "Nacelle_angle": 287.15,
        "Nacelle_angle_std": 3.2,
        "Nacelle_temperature": 16.5,
        "Nacelle_temperature_std": 0.35,
        "Absolute_wind_direction": 289.8,
        "Outdoor_temperature": 2.3,
        "Outdoor_temperature_std": 0.18,
        "Grid_frequency": 50.02,
        "Grid_frequency_std": 0.015,
        "Grid_voltage": 686.2,
        "Grid_voltage_std": 1.5
      }
    ]
  }'
```

### Notes

- Send `MAC_CODE` (e.g., `WT1..WT4`): the API encodes it into `MAC_CODEencoded` internally when required.
- The safest workflow is: open `GET /schema` and provide all features listed in `model_input_features`.
- Missing or null critical features may be rejected.

---

## Model artifacts

Training writes these files to the shared Docker volume mounted at `/models`:

- `model.keras`
- `scaler_x.pkl`
- `scaler_y.pkl`
- `label_encoder.pkl`
- `feature_cols.json`
- `metadata.json`

Inference loads them from `MODELS_DIR` (set to `/models` in Docker Compose).

---

## Local runs (without Docker)

### Training

```bash
python training/train.py
```

You can override paths using environment variables:

```bash
DATA_X=training/data/engie_X.csv \
DATA_Y=training/data/engie_Y.csv \
OUTDIR=inference/models \
python training/train.py
```

### Inference (FastAPI)

```bash
cd inference/api
MODELS_DIR=../models uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Tests

Run API tests:

```bash
python -m pytest -q inference/api/tests
```

---

## Configuration

### Training (`training/train.py`)

- `DATA_X` — path to `engie_X.csv` (default: `training/data/engie_X.csv`)
- `DATA_Y` — path to `engie_Y.csv` (default: `training/data/engie_Y.csv`)
- `OUTDIR` — output directory for artifacts (default: `../inference/models` relative to `training/`)

### Inference (`inference/api/main.py`)

- `MODELS_DIR` — directory containing model artifacts (default depends on your code; in Docker it is `/models`)

---

## Troubleshooting

- **“Missing artifact” / model not found**: run training first:
  ```bash
  docker compose up --build training
  ```
- **GPU warnings (CUDA not found)**: expected on CPU-only machines; TensorFlow will run on CPU.
- **Large Docker build context**: ensure datasets are mounted as volumes (as in `docker-compose.yml`) and not copied into images.
- **Rebuild images**:
  ```bash
  docker compose build --no-cache
  ```
```