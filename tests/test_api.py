import asyncio
from fastapi.testclient import TestClient
from scalable_ml_pipeline.api import app, config, startup_event

# Set the configuration value for tests
config.run_location = "local"

asyncio.run(startup_event())

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_valid_data():
    """Test the predict endpoint with valid data."""
    payload = [
        {
            "age": 49,
            "workclass": "Private",
            "fnlgt": 160187,
            "education": "9th",
            "education_num": 5,
            "marital_status": "Married-spouse-absent",
            "occupation": "Other-source",
            "relationship": "Not-in-family",
            "race": "Black",
            "sex": "Female",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 16,
            "native_country": "Jamaica"
        }
    ]
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_salary" in response.json()
    assert isinstance(response.json()["predicted_salary"], list)

def test_predict_invalid_data():
    """Test the predict endpoint with invalid data."""
    payload = "invalid_data"  # Invalid format
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_empty_data():
    """Test the predict endpoint with empty data."""
    payload = []
    response = client.post("/predict", json=payload)
    assert response.status_code == 400  # Bad Request
    assert response.json() == {"detail": "Input data is empty."}

def test_predict_missing_data_key():
    """Test the predict endpoint with missing 'data' key."""
    payload = {}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

def test_invalid_endpoint():
    """Test an invalid endpoint."""
    response = client.get("/invalid_endpoint")
    assert response.status_code == 404  # Not Found

def test_run_location_config():
    """Test the run_location configuration."""
    assert config.run_location == "local"