"""Integration tests for CORS headers."""

import pytest

from src.main import app


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_cors_header_present(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers.get("Access-Control-Allow-Origin") == "*"
