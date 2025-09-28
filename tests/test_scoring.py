"""Unit tests for the scoring helpers."""

import math

import pytest

from src.algorithms import (
    DEFAULT_WEIGHTS,
    compute_match_score,
    score_connections,
    score_goals,
    score_industry,
    score_location,
    score_role,
    score_skills,
)


def test_score_industry_same_and_different():
    user = {"industry": "tech"}
    target_same = {"industry": "Tech"}
    target_other = {"industry": "finance"}

    assert score_industry(user, target_same) == 100.0
    assert score_industry(user, target_other) == 0.0


def test_score_role_partial_match():
    user = {"role": "Data Manager"}
    target = {"role": "Project Manager"}

    assert score_role(user, target) == 60.0
    assert score_role(user, {"role": "Engineer"}) == 0.0


def test_score_skills_uses_jaccard_index():
    user = {"skills": ["python", "ml", "leadership"]}
    target = {"skills": ["python", "leadership"]}

    assert score_skills(user, target) == pytest.approx(66.67, rel=1e-3)
    assert score_skills(user, {"skills": []}) == 0.0


def test_score_location_city_country_region():
    user = {"location": {"city": "Paris", "country": "FR", "region": "Europe"}}
    same_city = {"location": {"city": "Paris", "country": "FR"}}
    same_country = {"location": {"city": "Lyon", "country": "FR"}}
    same_region = {"location": {"country": "ES", "region": "Europe"}}
    far = {"location": {"city": "New York", "country": "US"}}

    assert score_location(user, same_city) == 100.0
    assert score_location(user, same_country) == 70.0
    assert score_location(user, same_region) == 40.0
    assert score_location(user, far) == 0.0


def test_score_connections_and_goals():
    user = {"connections": ["a", "b", "c"], "objectives": ["mentoring", "hiring"]}
    target = {"connections": ["b", "c", "d"], "objectives": ["mentoring", "growth"]}

    assert score_connections(user, target) == pytest.approx(50.0)
    assert score_goals(user, target) == pytest.approx(33.33, rel=1e-3)
    assert score_connections(user, {"connections": []}) == 0.0


def test_compute_match_score_with_custom_weights():
    user_profile = {
        "industry": "tech",
        "role": "Engineer",
        "skills": ["python", "ml"],
        "location": {"city": "Paris", "country": "FR"},
        "connections": ["c1", "c2"],
        "objectives": ["networking"],
    }
    target_profile = {
        "industry": "tech",
        "role": "Designer",
        "skills": ["python", "ux"],
        "location": {"city": "Lyon", "country": "FR"},
        "connections": ["c2", "c3"],
        "objectives": ["networking", "fundraising"],
    }

    overrides = {"skills": 3.0, "objectives": 1.0}
    expected_weights = dict(DEFAULT_WEIGHTS)
    expected_weights.update(overrides)
    total_weight = sum(expected_weights.values())
    expected_weights = {key: value / total_weight for key, value in expected_weights.items()}

    scoring = compute_match_score(user_profile, target_profile, {"weights": overrides})

    assert scoring["weights"]["skills"] == pytest.approx(expected_weights["skills"])
    assert scoring["weights"]["objectives"] == pytest.approx(expected_weights["objectives"])

    details = scoring["details"]
    manual_total = sum(details[key] * expected_weights[key] for key in DEFAULT_WEIGHTS)
    assert scoring["total"] == pytest.approx(manual_total, rel=1e-4)

    # Ensure normalization produces weights summing to 1
    assert math.isclose(sum(scoring["weights"].values()), 1.0, rel_tol=1e-9)
