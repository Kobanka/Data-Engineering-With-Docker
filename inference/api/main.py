import os, json
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
MODELS_DIR = os.getenv("MODELS_DIR", DEFAULT_MODELS_DIR)

MODEL_PATH = os.path.join(MODELS_DIR, "model.keras")
SCALER_X_PATH = os.path.join(MODELS_DIR, "scaler_x.pkl")
SCALER_Y_PATH = os.path.join(MODELS_DIR, "scaler_y.pkl")
FEATURE_COLS_PATH = os.path.join(MODELS_DIR, "feature_cols.json")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")


model = None
scaler_x = None
scaler_y = None
label_encoder = None
feature_cols: List[str] = []
metadata: Dict[str, Any] = {}


class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(
        ...,
        description="List of JSON objects (one per sample/row). Extra keys are ignored."
    )


class PredictResponse(BaseModel):
    n_rows: int
    n_features: int
    predictions: List[float]
    errors: Optional[List[Optional[str]]] = None
    metadata: Optional[Dict[str, Any]] = None


def load_artifacts():
    global model, scaler_x, scaler_y, label_encoder, feature_cols, metadata

    required = [
        MODEL_PATH,
        SCALER_X_PATH,
        SCALER_Y_PATH,
        FEATURE_COLS_PATH,
        LABEL_ENCODER_PATH,
    ]
    for p in required:
        if not os.path.exists(p):
            raise RuntimeError(f"Missing artifact: {p}")

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler_x = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    feature_cols = json.load(open(FEATURE_COLS_PATH))

    if os.path.exists(METADATA_PATH):
        metadata = json.load(open(METADATA_PATH))
    else:
        metadata = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="ENGIE Wind Power Prediction API",
    version="1.1.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_dir": MODELS_DIR,
        "n_features": len(feature_cols),
        "uses_mac_code_encoded": ("MAC_CODEencoded" in feature_cols),
        "metadata": metadata,
    }


@app.get("/schema")
def schema():
    return {
        "request_example": {
            "rows": [
                {
                    "MAC_CODE": "WT1",
                    "Date_time": 123.0,
                    "Pitch_angle": 0.1,
                    "Generator_speed": 1200.0
                }
            ]
        },
        "required_raw_keys": ["MAC_CODE"] if ("MAC_CODEencoded" in feature_cols) else [],
        "model_input_features": feature_cols,  # includes MAC_CODEencoded if used by the model
        "ignored_keys": ["ID", "TARGET"],
        "notes": [
            "Send MAC_CODE (e.g., WT1/WT2/WT3/WT4). The API will compute MAC_CODEencoded internally.",
            "All features in model_input_features must be present and non-null after preprocessing.",
        ],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None or scaler_x is None or scaler_y is None:
        raise HTTPException(status_code=500, detail="Artifacts not loaded")

    df = pd.DataFrame(req.rows)

    # Build MAC_CODEencoded if the trained model expects it
    if "MAC_CODEencoded" in feature_cols:
        if label_encoder is None:
            raise HTTPException(status_code=500, detail="Label encoder not loaded")

        if "MAC_CODE" not in df.columns:
            raise HTTPException(status_code=400, detail="Missing required field: MAC_CODE")

        # LabelEncoder raises on unseen labels, so we pre-check [web:34][web:35]
        known = set(getattr(label_encoder, "classes_", []))
        if not known:
            raise HTTPException(status_code=500, detail="Label encoder has no classes_")

        unseen_mask = ~df["MAC_CODE"].isin(known)
        if unseen_mask.any():
            unseen = df.loc[unseen_mask, "MAC_CODE"].astype(str).unique().tolist()
            raise HTTPException(
                status_code=400,
                detail=f"Unseen MAC_CODE values: {unseen}. Known: {list(label_encoder.classes_)}"
            )

        df["MAC_CODEencoded"] = label_encoder.transform(df["MAC_CODE"])

    # Build X strictly from training feature order
    X = df.reindex(columns=feature_cols)

    # Validate missing features (entire column missing OR all-null)
    missing_any = X.isna().all(axis=0)
    missing_cols = missing_any[missing_any].index.tolist()
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features (entirely absent or all null): {missing_cols[:50]}"
        )

    # Strict: if any NaNs remain, reject
    if X.isna().any().any():
        bad_rows = X.isna().any(axis=1).to_numpy().nonzero()[0].tolist()
        raise HTTPException(
            status_code=400,
            detail=f"Some rows contain null/missing feature values. Bad row indices: {bad_rows[:50]}"
        )

    X_np = X.to_numpy(dtype=np.float32)
    X_s = scaler_x.transform(X_np)

    y_s = model.predict(X_s, verbose=0).reshape(-1, 1)
    y = scaler_y.inverse_transform(y_s).ravel()

    return PredictResponse(
        n_rows=len(y),
        n_features=len(feature_cols),
        predictions=[float(v) for v in y],
        errors=[None] * len(y),
        metadata={"model": metadata} if metadata else None,
    )