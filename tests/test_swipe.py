"""Tests for the /swipe endpoint validations and persistence."""

import pytest

from src.main import app
from src.storage import (
    count_rows,
    create_swipe,
    create_user,
    fetch_matches_for_user,
    fetch_swipe_events,
)
from src.storage.models import Swipe, User


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _create_users():
    return (
        create_user(
            User(
                id=None,
                email="user1@example.com",
                full_name="User 1",
                preferences=["ai", "cloud"],
            )
        ),
        create_user(
            User(
                id=None,
                email="user2@example.com",
                full_name="User 2",
                preferences=["ai", "robotics"],
            )
        ),
    )


def test_swipe_valid_request_creates_records(client):
    _create_users()

    response = client.post(
        "/swipe",
        json={"user_id": 1, "target_id": 2, "action": "like"},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["is_match"] is False

    assert count_rows("swipes") == 1
    events = fetch_swipe_events()
    assert len(events) == 1
    assert events[0]["swipe_id"] == 1
    assert events[0]["payload"]["user_preferences"] == ["ai", "cloud"]


def test_swipe_creates_match_on_mutual_like(client):
    _create_users()
    create_swipe(Swipe(id=None, user_id=2, target_id=1, action="like"))

    response = client.post(
        "/swipe",
        json={"user_id": 1, "target_id": 2, "action": "like"},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["is_match"] is True
    assert body["match_score"] > 0
    assert body["common_interests"] == ["ai"]

    matches_user_one = fetch_matches_for_user(1)
    matches_user_two = fetch_matches_for_user(2)
    assert len(matches_user_one) == 1
    assert len(matches_user_two) == 1
    assert count_rows("match_scores") == 2


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
        json={"user_id": 1, "target_id": 2, "action": "superlike"},
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


def test_swipe_returns_not_found_for_unknown_user(client):
    response = client.post(
        "/swipe",
        json={"user_id": 999, "target_id": 1, "action": "like"},
    )

    assert response.status_code == 404
