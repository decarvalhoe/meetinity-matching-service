"""Tests for the matches endpoint."""

import pytest

from src.main import app


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_matches_are_user_specific(client):
    """Ensure each user receives the appropriate match list."""

    response_user_one = client.get("/matches/1")
    response_user_two = client.get("/matches/2")
    response_unknown_user = client.get("/matches/999")

    assert response_user_one.status_code == 200
    assert response_user_two.status_code == 200
    assert response_unknown_user.status_code == 200

    matches_user_one = response_user_one.get_json()["matches"]
    matches_user_two = response_user_two.get_json()["matches"]
    matches_unknown_user = response_unknown_user.get_json()["matches"]

    assert matches_user_one != matches_user_two
    assert matches_unknown_user == []
