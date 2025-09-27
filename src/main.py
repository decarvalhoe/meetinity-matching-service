"""Meetinity Matching Service.

This service handles user matching algorithms, profile suggestions,
and swipe-based interactions for the Meetinity platform.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


MATCHES_BY_USER = {
    1: [
        {
            "id": 1,
            "user_id": 101,
            "name": "Sophie Martin",
            "title": "Product Manager",
            "company": "TechCorp",
            "match_score": 85,
            "common_interests": ["startup", "product", "innovation"],
        },
        {
            "id": 2,
            "user_id": 102,
            "name": "Thomas Dubois",
            "title": "CTO",
            "company": "InnovateLab",
            "match_score": 78,
            "common_interests": ["tech", "leadership", "ai"],
        },
    ],
    2: [
        {
            "id": 3,
            "user_id": 103,
            "name": "Camille Lefevre",
            "title": "Data Scientist",
            "company": "DataWiz",
            "match_score": 88,
            "common_interests": ["data", "ai", "cloud"],
        }
    ],
}


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
    matches = MATCHES_BY_USER.get(user_id, [])
    return jsonify({"matches": matches, "user_id": user_id})


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

    # Basic match detection logic (to be enhanced with real data)
    is_match = action == "like" and target_id == 101  # Simulation

    return jsonify(
        {
            "user_id": user_id,
            "target_id": target_id,
            "action": action,
            "is_match": is_match,
        }
    )


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
