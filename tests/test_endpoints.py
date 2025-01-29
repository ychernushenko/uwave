import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from server import app  # noqa

client = TestClient(app)


@pytest.fixture(scope="module")
def trained_model_id():
    response = client.post("/train?role=user")
    assert response.status_code == 200
    model_id = response.json()["model_id"]
    assert isinstance(model_id, str)
    return model_id


def test_train():
    response = client.post("/train?role=user")
    assert response.status_code == 200
    assert "model_id" in response.json()
    model_id = response.json()["model_id"]
    assert isinstance(model_id, str)


def test_get_model(trained_model_id):
    model_id = trained_model_id
    response = client.get(f"/model?id={model_id}&role=user")
    assert response.status_code == 200
    assert "classification_report" in response.json()
    assert "confusion_matrix" in response.json()
