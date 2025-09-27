"""Tests for the /swipe endpoint validations."""

import pytest

from src.main import app


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_swipe_valid_request(client):
    response = client.post(
        "/swipe",
        json={"user_id": 1, "target_id": 101, "action": "like"},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["user_id"] == 1
    assert body["target_id"] == 101
    assert body["action"] == "like"
    assert body["is_match"] is True


def test_swipe_missing_field(client):
    response = client.post(
        "/swipe",
        json={"user_id": 1, "action": "like"},
    )

    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == "Invalid request"
    assert "Missing field 'target_id'" in body["details"]


def test_swipe_invalid_action(client):
    response = client.post(
        "/swipe",
        json={"user_id": 1, "target_id": 202, "action": "superlike"},
    )

    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == "Invalid request"
    assert "either 'like' or 'pass'" in body["details"]


def test_swipe_non_json_payload(client):
    response = client.post(
        "/swipe",
        data="user_id=1&target_id=2&action=like",
        content_type="application/x-www-form-urlencoded",
    )

    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == "Invalid request"
    assert "application/json" in body["details"]
