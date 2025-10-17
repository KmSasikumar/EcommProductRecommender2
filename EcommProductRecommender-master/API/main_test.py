import pytest
from fastapi.testclient import TestClient
from main import app, API_KEYS_DB

client = TestClient(app)

def test_get_recommendations_invalid_api_key():
    response = client.post("/recommend", headers={"X-API-Key": "invalid-key"}, json={"user_id": "user0"})
    assert response.status_code == 403
    assert response.json() == {"detail": "Could not validate credentials"}

def test_get_recommendations_missing_model_path():
    API_KEYS_DB["testkey_missing_model"] = {"mappings_path": "some/path"}
    response = client.post("/recommend", headers={"X-API-Key": "testkey_missing_model"}, json={"user_id": "user0"})
    assert response.status_code == 404
    assert "Model file not found" in response.json()["detail"]
    del API_KEYS_DB["testkey_missing_model"]

import os

def test_get_recommendations_missing_mappings_path():
    # Create a dummy model file so the check passes
    dummy_model_path = "dummy_model.h5"
    try:
        with open(dummy_model_path, "w") as f:
            f.write("dummy")
        API_KEYS_DB["testkey_missing_mappings"] = {"model_path": dummy_model_path}
        response = client.post("/recommend", headers={"X-API-Key": "testkey_missing_mappings"}, json={"user_id": "user0"})
        assert response.status_code == 404
        assert "Mappings file not found" in response.json()["detail"]
    finally:
        del API_KEYS_DB["testkey_missing_mappings"]
        if os.path.exists(dummy_model_path):
            os.remove(dummy_model_path)