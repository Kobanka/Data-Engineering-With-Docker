# ENGIE Wind Power Prediction

End-to-end training and inference for wind turbine power prediction using Docker Compose.

## Project layout

- training/
  - train.py: model training pipeline
  - data/: input CSVs (engie_X.csv, engie_Y.csv)
- inference/
  - api/main.py: FastAPI inference service
  - api/tests/: API tests
- docker-compose.yml: training + inference with a shared models volume

## Prerequisites

- Docker and Docker Compose
- (Optional) Python 3.12 for local runs

## Quick start with Docker Compose

1. Train and write model artifacts to the shared volume:

   docker compose run --rm training

2. Start the inference API:

   docker compose up --build inference

3. Check health:

   curl http://localhost:8000/health

## Inference API

Base URL: http://localhost:8000

Endpoints:

- GET /health: service status and metadata
- GET /schema: request schema and required features
- POST /predict: run predictions

Example request:

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rows": [
      {
        "MAC_CODE": "WT1",
        "Date_time": 123.0,
        "Pitch_angle": 0.1,
        "Generator_speed": 1200.0
      }
    ]
  }'

Notes:

- If the model expects MAC_CODEencoded, send MAC_CODE and the API will encode it.
- All features listed in /schema -> model_input_features must be present and non-null.

## Model artifacts

The training container writes these files to the shared models volume:

- model.keras
- scaler_x.pkl
- scaler_y.pkl
- label_encoder.pkl
- feature_cols.json
- metadata.json

The inference container loads them from /models (set via MODELS_DIR).

## Local runs (without Docker)

Training:

  /home/aniss/repos/engie-data-engineering-with-docker/engie-env/bin/python training/train.py

Inference:

  cd inference/api
  /home/aniss/repos/engie-data-engineering-with-docker/engie-env/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000

## Tests

Run API tests:

  /home/aniss/repos/engie-data-engineering-with-docker/engie-env/bin/python -m pytest -q inference/api/tests

## Configuration

Training (training/train.py):

- DATA_X: path to engie_X.csv (default: training/data/engie_X.csv)
- DATA_Y: path to engie_Y.csv (default: training/data/engie_Y.csv)
- OUTDIR: output directory for artifacts (default: inference/models)

Inference (inference/api/main.py):

- MODELS_DIR: directory containing model artifacts (default: inference/models)

## Troubleshooting

- If Docker build fails with large context transfers, ensure data is mounted (as in docker-compose.yml) and not copied into the image.
- If you see "Missing artifact" errors, run the training container first.
