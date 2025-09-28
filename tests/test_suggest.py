"""Integration tests for the suggestion endpoint."""

import pytest

from src.algorithms import DEFAULT_WEIGHTS
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


def test_suggest_endpoint_returns_weighted_reasons(client):
    seeker = create_user(
        _build_user(
            email="seeker@example.com",
            full_name="Seeker",
            title="Data Scientist",
            preferences=[
                "industry:tech",
                {"type": "skills", "value": ["python", "ml", "data"]},
                {"type": "location", "value": {"city": "Paris", "country": "FR"}},
                {"type": "connections", "value": ["shared-1"]},
                {"type": "objectives", "value": ["networking", "mentoring"]},
            ],
        )
    )

    best_match = create_user(
        _build_user(
            email="ally@example.com",
            full_name="Ally",
            title="Data Scientist",
            preferences=[
                "industry:tech",
                "skill:python",
                "skill:ml",
                "skill:analytics",
                "location:Paris,FR",
                {"type": "connections", "value": ["shared-1", "shared-2"]},
                {"type": "objectives", "value": ["networking"]},
            ],
        )
    )

    _ = create_user(
        _build_user(
            email="other@example.com",
            full_name="Other",
            title="Marketing Lead",
            preferences=[
                "industry:marketing",
                "skill:content",
                "location:Madrid,ES",
                {"type": "objectives", "value": ["partnerships"]},
            ],
        )
    )

    response = client.get(f"/algorithm/suggest/{seeker.id}?limit=1&weight_skills=0.5")
    assert response.status_code == 200
    payload = response.get_json()

    assert payload["for_user"] == seeker.id

    overrides = dict(DEFAULT_WEIGHTS)
    overrides["skills"] = 0.5
    expected_skill_weight = overrides["skills"] / sum(overrides.values())
    assert payload["weights"]["skills"] == pytest.approx(round(expected_skill_weight, 4), abs=1e-4)

    suggestions = payload["suggestions"]
    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion["user_id"] == best_match.id
    assert suggestion["match_score"] == pytest.approx(suggestion["match_score"])
    assert 0.0 <= suggestion["preference_score"] <= 1.0

    reasons = suggestion["reasons"]
    assert reasons, "Expected weighted reasons to be present"
    contributions = [reason["contribution"] for reason in reasons]
    assert contributions == sorted(contributions, reverse=True)

    for reason in reasons:
        assert {
            "criterion",
            "label",
            "score",
            "weight",
            "contribution",
            "description",
        } <= reason.keys()
        assert reason["contribution"] == pytest.approx(
            reason["score"] * reason["weight"],
            abs=0.02,
        )
        assert str(reason["score"]) in reason["description"]
