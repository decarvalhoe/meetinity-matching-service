"""Meetinity Matching Service.

This service handles user matching algorithms, profile suggestions,
and swipe-based interactions for the Meetinity platform.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.algorithms import (
    DEFAULT_WEIGHTS,
    compute_match_score,
    predict_preference_score,
)
from src.storage import (
    create_matches,
    create_swipe,
    get_user,
    has_mutual_like,
    init_db,
    list_users,
    log_swipe_event,
)
from src.storage.models import Swipe, SwipeEvent, User

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
init_db()

CRITERION_LABELS = {
    "industry": "Industrie",
    "role": "Rôle",
    "skills": "Compétences",
    "location": "Localisation",
    "connections": "Connexions",
    "objectives": "Objectifs",
}

PREFERENCE_WEIGHT = 0.4


def _preferences_set(user: User) -> set[str]:
    preferences = user.preferences or []
    return {pref.lower() for pref in preferences if isinstance(pref, str)}


def _calculate_match_score(user: User, target: User) -> float:
    """Basic score using overlap of stored preferences."""

    prefs_user = _preferences_set(user)
    prefs_target = _preferences_set(target)
    if not prefs_user and not prefs_target:
        return 50.0

    intersection = prefs_user & prefs_target
    union = prefs_user | prefs_target
    if not union:
        return 50.0
    return round((len(intersection) / len(union)) * 100, 2)


def _iterable_from_value(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        nested = (
            value.get("items")
            or value.get("value")
            or value.get("values")
            or value.get("list")
        )
        if nested is not None and nested is not value:
            return _iterable_from_value(nested)
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return (str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    if not text:
        return []
    return [text]


def _update_location(location: Dict[str, str], value: Any) -> None:
    def _assign(key: str, raw: Any) -> None:
        if raw is None:
            return
        text = str(raw).strip()
        if text:
            location[key] = text

    if value is None:
        return
    if isinstance(value, Mapping):
        for key in ("city", "country", "region", "continent"):
            if key in value:
                _assign(key, value[key])
        return

    text = str(value).strip()
    if not text:
        return
    tokens = [part.strip() for part in re.split(r"[,/|]", text) if part.strip()]
    if len(tokens) == 1:
        token = tokens[0]
        if len(token) <= 2 and token.isalpha():
            _assign("country", token)
        else:
            _assign("city", token)
        return
    if tokens:
        _assign("city", tokens[0])
    if len(tokens) >= 2:
        _assign("country", tokens[1])
    if len(tokens) >= 3:
        _assign("region", tokens[2])


def _build_scoring_profile(user: User) -> Dict[str, Any]:
    industry: str | None = None
    role = user.title
    skills: set[str] = set()
    objectives: set[str] = set()
    connections: set[str] = set()
    location: Dict[str, str] = {}

    for entry in user.preferences or []:
        key: str | None
        value: Any
        if isinstance(entry, Mapping):
            key = str(
                entry.get("type")
                or entry.get("key")
                or entry.get("name")
                or entry.get("category")
                or ""
            ).strip().lower()
            value = entry.get("value")
            if not key:
                continue
        else:
            text = str(entry).strip()
            if not text:
                continue
            raw_key, sep, remainder = text.partition(":")
            if sep:
                key = raw_key.strip().lower()
                value = remainder.strip()
            else:
                key = "skill"
                value = text

        if key == "industry":
            collected = list(_iterable_from_value(value))
            if collected:
                industry = collected[0]
        elif key == "role":
            collected = list(_iterable_from_value(value))
            if collected:
                role = collected[0]
        elif key in {"skill", "skills"}:
            skills.update(token for token in _iterable_from_value(value))
        elif key in {"objective", "objectives", "goal", "goals"}:
            objectives.update(token for token in _iterable_from_value(value))
        elif key in {"connection", "connections"}:
            connections.update(token for token in _iterable_from_value(value))
        elif key == "location":
            _update_location(location, value)
        elif key in {"city", "country", "region", "continent"}:
            _update_location(location, {key: value})
        else:
            # Backwards compatibility: treat unknown strings as skills
            skills.update(token for token in _iterable_from_value(value if value else entry))

    profile = {
        "id": user.id,
        "name": user.full_name,
        "industry": industry,
        "role": role,
        "skills": sorted(skills),
        "location": location,
        "connections": sorted(connections),
        "objectives": sorted(objectives),
    }
    return profile


def _parse_positive_int(raw: str | None, name: str, default: int) -> int:
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Le paramètre '{name}' doit être un entier positif.") from exc
    if value <= 0:
        raise ValueError(f"Le paramètre '{name}' doit être strictement positif.")
    return value


def _parse_weight_overrides(args: Mapping[str, str]) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    for criterion in DEFAULT_WEIGHTS.keys():
        param = f"weight_{criterion}"
        if param not in args:
            continue
        raw_value = args[param]
        try:
            overrides[criterion] = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Le poids '{param}' doit être un nombre valide."
            ) from exc
    return overrides


def _round_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    return {key: round(value, 4) for key, value in weights.items()}


def _format_breakdown(scoring: Mapping[str, Any]) -> Dict[str, float]:
    details = scoring.get("details", {}) if isinstance(scoring, Mapping) else {}
    return {key: round(float(value), 2) for key, value in details.items()}


def _format_weighted_reasons(scoring: Mapping[str, Any]) -> List[Dict[str, Any]]:
    details = scoring.get("details", {}) if isinstance(scoring, Mapping) else {}
    weights = scoring.get("weights", {}) if isinstance(scoring, Mapping) else {}
    reasons: List[Dict[str, Any]] = []
    for criterion, score_value in details.items():
        weight = float(weights.get(criterion, 0.0))
        contribution = round(float(score_value) * weight, 2)
        if contribution <= 0:
            continue
        label = CRITERION_LABELS.get(criterion, criterion.title())
        reasons.append(
            {
                "criterion": criterion,
                "label": label,
                "score": round(float(score_value), 2),
                "weight": round(weight, 4),
                "contribution": contribution,
                "description": f"{label}: {float(score_value):.1f} (poids {weight:.2f}, contribution {contribution:.1f})",
            }
        )
    reasons.sort(key=lambda item: item["contribution"], reverse=True)
    return reasons


def _combined_score(match_score: float, preference_score: float) -> float:
    match_component = (1.0 - PREFERENCE_WEIGHT) * match_score
    preference_component = PREFERENCE_WEIGHT * (preference_score * 100.0)
    return round(match_component + preference_component, 2)


def _score_candidates_for_user(
    user: User, context: Mapping[str, Any] | None
) -> Tuple[List[Tuple[User, Dict[str, Any]]], Dict[str, float]]:
    base_profile = _build_scoring_profile(user)
    scored: List[Tuple[User, Dict[str, Any]]] = []
    weights_used: Dict[str, float] | None = None

    for candidate in list_users(exclude_user_id=user.id):
        candidate_profile = _build_scoring_profile(candidate)
        scoring = compute_match_score(base_profile, candidate_profile, context)
        weights_used = scoring["weights"]
        preference_score = predict_preference_score(user, candidate, scoring)
        scoring["preference_score"] = preference_score
        scoring["combined_score"] = _combined_score(scoring["total"], preference_score)
        scored.append((candidate, scoring))

    if weights_used is None:
        weights_used = compute_match_score(base_profile, base_profile, context)["weights"]

    return scored, weights_used


@app.route("/health")
def health():
    """Health check endpoint.

    Returns:
        Response: JSON response with service status.
    """
    return jsonify({"status": "ok", "service": "matching-service"})


@app.route("/matches/<int:user_id>")
def get_matches(user_id):
    """Retrieve matches for a specific user using the scoring engine."""

    user = get_user(user_id)
    if user is None:
        return (
            jsonify({"error": "Not found", "details": "Utilisateur introuvable."}),
            404,
        )

    try:
        page = _parse_positive_int(request.args.get("page"), "page", 1)
        page_size = _parse_positive_int(
            request.args.get("page_size"), "page_size", 10
        )
        weight_overrides = _parse_weight_overrides(request.args)
    except ValueError as exc:
        return jsonify({"error": "Invalid request", "details": str(exc)}), 400

    context = {"weights": weight_overrides} if weight_overrides else None
    scored_candidates, weights_used = _score_candidates_for_user(user, context)
    scored_candidates.sort(
        key=lambda item: item[1].get("combined_score", item[1]["total"]),
        reverse=True,
    )

    total = len(scored_candidates)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = scored_candidates[start:end]

    results = [
        {
            "user_id": candidate.id,
            "name": candidate.full_name,
            "title": candidate.title,
            "company": candidate.company,
            "score": item_score.get("combined_score", item_score["total"]),
            "match_score": item_score["total"],
            "preference_score": round(
                float(item_score.get("preference_score", 0.5)),
                4,
            ),
            "breakdown": _format_breakdown(item_score),
        }
        for candidate, item_score in paginated
    ]

    response_body = {
        "user_id": user_id,
        "results": results,
        "weights": _round_weights(weights_used),
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "pages": math.ceil(total / page_size) if page_size else 0,
        },
    }
    return jsonify(response_body)


@app.route("/swipe", methods=["POST"])
def swipe():
    """Record a swipe action (like/pass) and detect matches."""

    if not request.is_json:
        return (
            jsonify(
                {
                    "error": "Invalid request",
                    "details": "Content type must be application/json.",
                }
            ),
            400,
        )

    data = request.get_json(silent=True)
    if data is None:
        return (
            jsonify(
                {
                    "error": "Invalid request",
                    "details": "Malformed JSON payload.",
                }
            ),
            400,
        )

    required_fields = {
        "user_id": int,
        "target_id": int,
        "action": str,
    }

    for field, expected_type in required_fields.items():
        if field not in data:
            return (
                jsonify(
                    {
                        "error": "Invalid request",
                        "details": f"Missing field '{field}'.",
                    }
                ),
                400,
            )

        value = data[field]
        if expected_type is int:
            if not isinstance(value, int) or isinstance(value, bool):
                return (
                    jsonify(
                        {
                            "error": "Invalid request",
                            "details": f"Field '{field}' must be an integer.",
                        }
                    ),
                    400,
                )
        elif not isinstance(value, expected_type):
            return (
                jsonify(
                    {
                        "error": "Invalid request",
                        "details": f"Field '{field}' must be of type {expected_type.__name__}.",
                    }
                ),
                400,
            )

    action = data["action"].lower()
    if action not in {"like", "pass"}:
        return (
            jsonify(
                {
                    "error": "Invalid request",
                    "details": "Field 'action' must be either 'like' or 'pass'.",
                }
            ),
            400,
        )

    user_id = data["user_id"]
    target_id = data["target_id"]

    user = get_user(user_id)
    target = get_user(target_id)
    if user is None or target is None:
        return (
            jsonify(
                {
                    "error": "Not found",
                    "details": "User or target not found in storage.",
                }
            ),
            404,
        )

    swipe = create_swipe(Swipe(id=None, user_id=user_id, target_id=target_id, action=action))

    is_match = False
    score_value = None
    common_interests: List[str] = []

    if action == "like" and has_mutual_like(user_id, target_id):
        is_match = True
        score_value = _calculate_match_score(user, target)
        common_interests = sorted(_preferences_set(user) & _preferences_set(target))
        create_matches(user_id, target_id, score_value, common_interests)

    event_payload = {
        "user_preferences": user.preferences,
        "target_preferences": target.preferences,
    }
    if is_match:
        event_payload["is_match"] = True
        event_payload["common_interests"] = common_interests

    log_swipe_event(
        SwipeEvent(
            id=None,
            swipe_id=swipe.id,
            event_type="swipe",
            user_id=user_id,
            target_id=target_id,
            action=action,
            score=score_value,
            payload=event_payload,
        )
    )

    response_body = {
        "user_id": user_id,
        "target_id": target_id,
        "action": action,
        "is_match": is_match,
    }
    if score_value is not None:
        response_body["match_score"] = score_value
        response_body["common_interests"] = common_interests

    return jsonify(response_body)


@app.route("/algorithm/suggest/<int:user_id>")
def suggest_profiles(user_id):
    """Generate personalized profile suggestions using the scoring engine."""

    user = get_user(user_id)
    if user is None:
        return (
            jsonify({"error": "Not found", "details": "Utilisateur introuvable."}),
            404,
        )

    try:
        limit = _parse_positive_int(request.args.get("limit"), "limit", 5)
        weight_overrides = _parse_weight_overrides(request.args)
    except ValueError as exc:
        return jsonify({"error": "Invalid request", "details": str(exc)}), 400

    context = {"weights": weight_overrides} if weight_overrides else None
    scored_candidates, weights_used = _score_candidates_for_user(user, context)
    scored_candidates.sort(
        key=lambda item: item[1].get("combined_score", item[1]["total"]),
        reverse=True,
    )

    suggestions = []
    for candidate, item_score in scored_candidates[:limit]:
        suggestions.append(
            {
                "user_id": candidate.id,
                "name": candidate.full_name,
                "title": candidate.title,
                "company": candidate.company,
                "score": item_score.get("combined_score", item_score["total"]),
                "match_score": item_score["total"],
                "preference_score": round(
                    float(item_score.get("preference_score", 0.5)),
                    4,
                ),
                "breakdown": _format_breakdown(item_score),
                "reasons": _format_weighted_reasons(item_score),
            }
        )

    return jsonify(
        {
            "for_user": user_id,
            "weights": _round_weights(weights_used),
            "suggestions": suggestions,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5004)
