"""Tests for the matches endpoint."""

import pytest

from src.main import app
from src.storage import create_matches, create_user
from src.storage.models import User


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_matches_endpoint_returns_persisted_data(client):
    """Ensure matches are read from the ORM-backed storage."""

    user = create_user(
        User(
            id=None,
            email="alice@example.com",
            full_name="Alice",
            preferences=["ai", "cloud"],
        )
    )
    partner = create_user(
        User(
            id=None,
            email="bob@example.com",
            full_name="Bob",
            title="CTO",
            company="InnovateLab",
            preferences=["ai", "robotics"],
        )
    )
    stranger = create_user(
        User(
            id=None,
            email="carol@example.com",
            full_name="Carol",
            preferences=["design"],
        )
    )

    create_matches(user.id, partner.id, score=88.5, common_interests=["ai"])

    response = client.get(f"/matches/{user.id}")
    other_response = client.get(f"/matches/{stranger.id}")

    assert response.status_code == 200
    assert other_response.status_code == 200

    matches = response.get_json()["matches"]
    assert len(matches) == 1
    first_match = matches[0]
    assert first_match["user_id"] == partner.id
    assert first_match["match_score"] == 88.5
    assert first_match["common_interests"] == ["ai"]

    assert other_response.get_json()["matches"] == []
