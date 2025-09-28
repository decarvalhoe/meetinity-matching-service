"""Scoring helpers for computing match relevance."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

DEFAULT_WEIGHTS: Dict[str, float] = {
    "industry": 0.15,
    "role": 0.15,
    "skills": 0.3,
    "location": 0.1,
    "connections": 0.15,
    "objectives": 0.15,
}


def _to_lower(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    return lowered or None


def _to_set(values: Any) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, (set, frozenset)):
        return {str(item).strip().lower() for item in values if str(item).strip()}
    if isinstance(values, dict):
        iterable = values.get("items") or values.get("value") or values.get("values")
        if isinstance(iterable, Iterable) and not isinstance(iterable, (str, bytes)):
            return _to_set(iterable)
        return set()
    if isinstance(values, (list, tuple)):
        return {str(item).strip().lower() for item in values if str(item).strip()}
    if isinstance(values, (str, bytes)):
        return {str(values).strip().lower()} if str(values).strip() else set()
    return set()


def score_industry(user_profile: Mapping[str, Any], target_profile: Mapping[str, Any]) -> float:
    """Return a score expressing the industry alignment."""

    user_industry = _to_lower(user_profile.get("industry"))
    target_industry = _to_lower(target_profile.get("industry"))
    if not user_industry or not target_industry:
        return 0.0
    return 100.0 if user_industry == target_industry else 0.0


def score_role(user_profile: Mapping[str, Any], target_profile: Mapping[str, Any]) -> float:
    """Return a score expressing how similar the roles are."""

    user_role = _to_lower(user_profile.get("role"))
    target_role = _to_lower(target_profile.get("role"))
    if not user_role or not target_role:
        return 0.0
    if user_role == target_role:
        return 100.0
    if user_role.split()[-1:] == target_role.split()[-1:]:
        return 60.0
    return 0.0


def score_skills(user_profile: Mapping[str, Any], target_profile: Mapping[str, Any]) -> float:
    """Score skills overlap using the Jaccard index."""

    user_skills = _to_set(user_profile.get("skills"))
    target_skills = _to_set(target_profile.get("skills"))
    if not user_skills or not target_skills:
        return 0.0
    intersection = user_skills & target_skills
    union = user_skills | target_skills
    if not union:
        return 0.0
    return round(len(intersection) / len(union) * 100, 2)


def score_location(user_profile: Mapping[str, Any], target_profile: Mapping[str, Any]) -> float:
    """Score geographic proximity with different granularities."""

    user_location = user_profile.get("location") or {}
    target_location = target_profile.get("location") or {}
    if not isinstance(user_location, Mapping) or not isinstance(target_location, Mapping):
        return 0.0

    user_city = _to_lower(user_location.get("city"))
    target_city = _to_lower(target_location.get("city"))
    if user_city and target_city and user_city == target_city:
        return 100.0

    user_country = _to_lower(user_location.get("country"))
    target_country = _to_lower(target_location.get("country"))
    if user_country and target_country and user_country == target_country:
        return 70.0

    user_region = _to_lower(user_location.get("region") or user_location.get("continent"))
    target_region = _to_lower(target_location.get("region") or target_location.get("continent"))
    if user_region and target_region and user_region == target_region:
        return 40.0

    return 0.0


def score_connections(user_profile: Mapping[str, Any], target_profile: Mapping[str, Any]) -> float:
    """Score overlap of mutual connections using the Jaccard index."""

    user_connections = _to_set(user_profile.get("connections"))
    target_connections = _to_set(target_profile.get("connections"))
    if not user_connections or not target_connections:
        return 0.0
    intersection = user_connections & target_connections
    union = user_connections | target_connections
    if not union:
        return 0.0
    return round(len(intersection) / len(union) * 100, 2)


def score_goals(user_profile: Mapping[str, Any], target_profile: Mapping[str, Any]) -> float:
    """Score alignment of professional objectives using the Jaccard index."""

    user_goals = _to_set(user_profile.get("objectives"))
    target_goals = _to_set(target_profile.get("objectives"))
    if not user_goals or not target_goals:
        return 0.0
    intersection = user_goals & target_goals
    union = user_goals | target_goals
    if not union:
        return 0.0
    return round(len(intersection) / len(union) * 100, 2)


_SCORE_FUNCTIONS: Dict[str, Any] = {
    "industry": score_industry,
    "role": score_role,
    "skills": score_skills,
    "location": score_location,
    "connections": score_connections,
    "objectives": score_goals,
}


def _resolve_weights(context: Mapping[str, Any] | None) -> Dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    if context and isinstance(context.get("weights"), Mapping):
        for key, value in context["weights"].items():
            if key in weights:
                try:
                    weights[key] = float(value)
                except (TypeError, ValueError):
                    continue
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {key: value / total_weight for key, value in weights.items()}


def compute_match_score(
    user_profile: Mapping[str, Any],
    target_profile: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compute a weighted match score for two profiles.

    The function returns both the aggregated score and a detailed breakdown
    of the sub-scores for each criterion.
    """

    normalized_weights = _resolve_weights(context)
    breakdown: Dict[str, float] = {}
    for criterion, scorer in _SCORE_FUNCTIONS.items():
        breakdown[criterion] = float(scorer(user_profile, target_profile))

    total = 0.0
    for criterion, score_value in breakdown.items():
        weight = normalized_weights.get(criterion, 0.0)
        total += score_value * weight

    return {
        "total": round(total, 2),
        "weights": normalized_weights,
        "details": breakdown,
    }
