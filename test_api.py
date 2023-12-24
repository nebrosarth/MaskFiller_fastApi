import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Masked Language Model API"}


def test_predict():
    text_to_predict = "This is a [MASK] example."
    response = client.post("/predict", json={"text": text_to_predict})
    assert response.status_code == 200
    assert "score" in response.json()[0]  # Assuming the response is a list of predictions
    assert "sequence" in response.json()[0]


@pytest.mark.parametrize("invalid_input", [
    {"text": ""},  # Empty text
    {"text": 123},  # Non-string input
    {"invalid_key": "test"}  # Invalid key in JSON payload
])
def test_predict_invalid_input(invalid_input):
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422  # FastAPI returns 422 Unprocessable Entity for invalid input
