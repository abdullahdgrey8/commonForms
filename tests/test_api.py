# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

client = TestClient(app)
TEST_USER = settings.TEST_USER_USERNAME
TEST_PASS = settings.TEST_USER_PASSWORD

def get_token():
    resp = client.post("/api/v1/auth/login", json={"username": TEST_USER, "password": TEST_PASS})
    return resp.json()["access_token"]

def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == settings.APP_NAME

def test_health():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

def test_login_success():
    resp = client.post("/api/v1/auth/login", json={"username": TEST_USER, "password": TEST_PASS})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_login_fail():
    resp = client.post("/api/v1/auth/login", json={"username": TEST_USER, "password": "wrong"})
    assert resp.status_code == 401

def test_pdf_requires_auth():
    with open("tests/sample.pdf", "rb") as f:
        resp = client.post("/api/v1/pdf/make-fillable", files={"pdf": f})
    assert resp.status_code == 401

@pytest.mark.skip(reason="Requires sample.pdf and model")
def test_pdf_processing():
    token = get_token()
    with open("tests/sample.pdf", "rb") as f:
        resp = client.post(
            "/api/v1/pdf/make-fillable",
            headers={"Authorization": f"Bearer {token}"},
            files={"pdf": ("test.pdf", f, "application/pdf")},
            data={"model": "FFDNet-S", "fast": "true"}
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"