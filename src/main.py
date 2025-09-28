"""Meetinity Matching Service.

This service handles user matching algorithms, profile suggestions,
and swipe-based interactions for the Meetinity platform.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.storage import (
    create_matches,
    create_swipe,
    fetch_matches_for_user,
    get_user,
    has_mutual_like,
    init_db,
    log_swipe_event,
)
from src.storage.models import Swipe, SwipeEvent, User

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
init_db()


def _preferences_set(user: User) -> set[str]:
    preferences = user.preferences or []
    return {pref.lower() for pref in preferences}


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


@app.route("/health")
def health():
    """Health check endpoint.
    
    Returns:
        Response: JSON response with service status.
    """
    return jsonify({"status": "ok", "service": "matching-service"})


@app.route("/matches/<int:user_id>")
def get_matches(user_id):
    """Retrieve matches for a specific user.
    
    Args:
        user_id (int): The ID of the user to get matches for.
        
    Returns:
        Response: JSON response with user matches and compatibility scores.
    """
    matches = fetch_matches_for_user(user_id)
    sanitized = []
    for match in matches:
        sanitized.append(
            {
                "id": match["id"],
                "user_id": match["user_id"],
                "name": match["name"],
                "title": match["title"],
                "company": match["company"],
                "match_score": match["match_score"],
                "common_interests": match["common_interests"],
                "created_at": match["created_at"],
            }
        )

    return jsonify({"matches": sanitized, "user_id": user_id})


@app.route("/swipe", methods=["POST"])
def swipe():
    """Record a swipe action (like/pass) and detect matches.
    
    Expected JSON payload:
        {
            "user_id": int,
            "target_id": int,
            "action": str  # 'like' or 'pass'
        }
        
    Returns:
        Response: JSON response with swipe result and match status.
    """
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
    common_interests: list[str] = []

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
    """Generate personalized profile suggestions using matching algorithm.
    
    Args:
        user_id (int): The ID of the user to generate suggestions for.
        
    Returns:
        Response: JSON response with suggested profiles and compatibility reasons.
    """
    suggestions = [
        {
            "user_id": 201,
            "name": "Marie Leroy",
            "title": "Entrepreneur",
            "score": 92,
            "reasons": [
                "Same industry",
                "Similar experience",
                "Mutual connections",
            ],
        },
        {
            "user_id": 202,
            "name": "Pierre Moreau",
            "title": "Investor",
            "score": 87,
            "reasons": ["Complementary skills", "Geographic proximity"],
        },
    ]
    return jsonify({"suggestions": suggestions, "for_user": user_id})


if __name__ == "__main__":
    app.run(debug=True, port=5004)
