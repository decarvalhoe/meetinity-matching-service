"""Tests covering the ML training and inference pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.train_preferences import train_preference_model
from src.algorithms.ml import clear_model_cache, predict_preference_score
from src.analytics import PreferenceCollector
from src.config import refresh_settings
from src.storage import create_swipe, create_user, log_swipe_event
from src.storage.models import Swipe, SwipeEvent, User


def _make_user(email: str, name: str, preferences: list[str]) -> User:
    return create_user(
        User(
            id=None,
            email=email,
            full_name=name,
            preferences=preferences,
        )
    )


def _record_swipe_event(
    *,
    user: User,
    target: User,
    action: str,
    score: float,
    is_match: bool = False,
    common_interests: list[str] | None = None,
) -> None:
    swipe = create_swipe(
        Swipe(id=None, user_id=user.id, target_id=target.id, action=action)
    )
    payload = {
        "user_preferences": user.preferences,
        "target_preferences": target.preferences,
    }
    if is_match:
        payload["is_match"] = True
        payload["common_interests"] = common_interests or []
    log_swipe_event(
        SwipeEvent(
            id=None,
            swipe_id=swipe.id,
            event_type="swipe",
            user_id=user.id,
            target_id=target.id,
            action=action,
            score=score,
            payload=payload,
        )
    )


@pytest.mark.usefixtures("_clean_database")
def test_training_pipeline_produces_versioned_model(tmp_path, monkeypatch):
    seeker = _make_user(
        "seeker@example.com",
        "Seeker",
        ["industry:tech", "skill:python", "skill:data", "location:Paris"],
    )
    ally = _make_user(
        "ally@example.com",
        "Ally",
        ["industry:tech", "skill:python", "location:Paris"],
    )
    mentor = _make_user(
        "mentor@example.com",
        "Mentor",
        ["industry:tech", "skill:ml", "skill:data", "location:Lyon"],
    )
    outsider = _make_user(
        "outsider@example.com",
        "Outsider",
        ["industry:finance", "skill:sales", "location:Madrid"],
    )
    spam = _make_user(
        "spam@example.com",
        "Spam",
        ["industry:marketing", "skill:ads", "location:Berlin"],
    )

    _record_swipe_event(
        user=seeker,
        target=ally,
        action="like",
        score=88.0,
        is_match=True,
        common_interests=["skill:python"],
    )
    _record_swipe_event(
        user=ally,
        target=seeker,
        action="like",
        score=82.0,
        is_match=True,
        common_interests=["skill:python"],
    )
    _record_swipe_event(
        user=seeker,
        target=mentor,
        action="like",
        score=74.0,
    )
    _record_swipe_event(
        user=mentor,
        target=seeker,
        action="like",
        score=70.0,
    )
    _record_swipe_event(
        user=seeker,
        target=outsider,
        action="pass",
        score=18.0,
    )
    _record_swipe_event(
        user=mentor,
        target=spam,
        action="pass",
        score=10.0,
    )

    collector = PreferenceCollector()
    samples = collector.collect()
    assert len(samples) >= 6
    assert {sample.label for sample in samples} == {0, 1}

    metadata = train_preference_model(output_dir=str(tmp_path))
    model_path = Path(tmp_path) / metadata["model_path"]
    assert model_path.exists()

    latest = Path(tmp_path) / "latest.json"
    assert latest.exists()

    monkeypatch.setenv("PREFERENCE_MODELS_DIR", str(tmp_path))
    refresh_settings()
    clear_model_cache()

    positive = predict_preference_score(seeker, ally, {"total": 88.0})
    negative = predict_preference_score(seeker, spam, {"total": 18.0})

    assert 0.0 <= negative <= 1.0
    assert 0.0 <= positive <= 1.0
    assert positive > negative
