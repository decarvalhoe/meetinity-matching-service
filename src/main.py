"""Meetinity Matching Service.

This service handles user matching algorithms, profile suggestions,
and swipe-based interactions for the Meetinity platform.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


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
    matches = [
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
    ]
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
    data = request.get_json()
    user_id = data.get("user_id") if data else None
    target_id = data.get("target_id") if data else None
    action = data.get("action") if data else None  # 'like' or 'pass'

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
