"""Tests for the matches endpoint."""

import pytest

from src.main import app
from src.storage import create_user
from src.storage.models import User


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def _build_user(**kwargs) -> User:
    base = {
        "id": None,
        "title": None,
        "company": None,
        "bio": None,
        "preferences": [],
    }
    base.update(kwargs)
    return User(**base)


def test_matches_endpoint_returns_sorted_paginated_scores(client):
    """Ensure matches are scored and paginated through the scoring engine."""

    alice = create_user(
        _build_user(
            email="alice@example.com",
            full_name="Alice",
            title="Product Manager",
            preferences=[
                {"type": "industry", "value": "tech"},
                {"type": "skills", "value": ["product strategy", "python", "leadership"]},
                {"type": "location", "value": "Paris,FR"},
                {"type": "connections", "value": ["mentor-1", "mentor-2"]},
                {"type": "objectives", "value": ["networking", "mentoring"]},
            ],
        )
    )

    bob = create_user(
        _build_user(
            email="bob@example.com",
            full_name="Bob",
            title="Product Manager",
            preferences=[
                "industry:tech",
                "skill:python",
                "skill:product strategy",
                "skill:growth",  # ensure some overlap but not perfect
                "location:Paris,FR",
                {"type": "connections", "value": ["mentor-1", "investor-1"]},
                {"type": "objectives", "value": ["networking", "fundraising"]},
            ],
        )
    )

    carol = create_user(
        _build_user(
            email="carol@example.com",
            full_name="Carol",
            title="Growth Lead",
            preferences=[
                "industry:tech",
                "skill:python",
                "skill:analytics",
                "location:Lyon,FR",
                {"type": "connections", "value": ["mentor-3"]},
                {"type": "objectives", "value": ["networking"]},
            ],
        )
    )

    dave = create_user(
        _build_user(
            email="dave@example.com",
            full_name="Dave",
            title="Operations Specialist",
            preferences=[
                "industry:finance",
                "skill:excel",
                "location:Berlin,DE",
                {"type": "objectives", "value": ["hiring"]},
            ],
        )
    )

    response = client.get(f"/matches/{alice.id}?page=1&page_size=2")
    assert response.status_code == 200
    payload = response.get_json()

    assert payload["user_id"] == alice.id
    assert payload["pagination"] == {"page": 1, "page_size": 2, "total": 3, "pages": 2}
    assert payload["weights"]["skills"] == pytest.approx(0.3)

    results = payload["results"]
    assert len(results) == 2
    assert [result["user_id"] for result in results] == [bob.id, carol.id]
    assert results[0]["score"] >= results[1]["score"]
    assert set(results[0]["breakdown"].keys()) == {
        "industry",
        "role",
        "skills",
        "location",
        "connections",
        "objectives",
    }

    second_page = client.get(f"/matches/{alice.id}?page=2&page_size=2")
    assert second_page.status_code == 200
    payload_page_two = second_page.get_json()
    assert [result["user_id"] for result in payload_page_two["results"]] == [dave.id]


def test_matches_endpoint_returns_404_for_unknown_user(client):
    response = client.get("/matches/9999")
    assert response.status_code == 404
    assert response.get_json()["error"] == "Not found"
