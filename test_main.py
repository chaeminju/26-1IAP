#API 동작 확인 (수정중)
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "확인중"}

def test_receive_log_success():
    payload = {
        "device_id": "jetson-01",
        "timestamp": "2024-06-01T12:00:00",
        "loss_value": 0.82,
        "risk_level": 3
    }

    response = client.post("/log/detection", json=payload)
    
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "log received"
    assert body["data"]["device_id"] == "jetson-01"
    assert body["data"]["risk_level"] == 3