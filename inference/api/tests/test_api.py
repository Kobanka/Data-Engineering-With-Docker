import pytest
from fastapi.testclient import TestClient
from inference.api.main import app


@pytest.fixture()
def client():
    with TestClient(app) as client:
        yield client


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_predict_happy_path(client):
    schema = client.get("/schema").json()
    feats = schema["model_input_features"]  # <-- CHANGED KEY

    # Build one dummy row with all model input features
    # Note: do NOT send MAC_CODEencoded; the API computes it from MAC_CODE.
    row = {f: 0.0 for f in feats if f != "MAC_CODEencoded"}

    # Provide MAC_CODE when model expects MAC_CODEencoded
    if "MAC_CODEencoded" in feats:
        row["MAC_CODE"] = "WT3"  # must be a known label, otherwise 400 [web:34][web:35]

    r = client.post("/predict", json={"rows": [row]})
    assert r.status_code == 200

    data = r.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1


def test_predict_rejects_unseen_mac_code(client):
    schema = client.get("/schema").json()
    feats = schema["model_input_features"]
    if "MAC_CODEencoded" not in feats:
        pytest.skip("Model does not use MAC_CODEencoded")

    row = {f: 0.0 for f in feats if f != "MAC_CODEencoded"}
    row["MAC_CODE"] = "WT999"  # unseen

    r = client.post("/predict", json={"rows": [row]})
    assert r.status_code == 400