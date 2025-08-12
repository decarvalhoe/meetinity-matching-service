from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "matching-service"})


@app.route("/matches/<int:user_id>")
def get_matches(user_id):
    """Récupérer les matches pour un utilisateur"""
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
    """Enregistrer un swipe (like/pass)"""
    data = request.get_json()
    user_id = data.get("user_id") if data else None
    target_id = data.get("target_id") if data else None
    action = data.get("action") if data else None  # 'like' or 'pass'

    # Logique de matching
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
    """Suggérer des profils basés sur l'algorithme de matching"""
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
