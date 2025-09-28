"""Machine learning helpers for preference scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Mapping, Sequence

from joblib import load

from src.analytics.collector import build_feature_vector
from src.config import get_settings
from src.storage.models import User


@dataclass(frozen=True)
class LoadedModel:
    feature_names: Sequence[str]
    model: Any


_MODEL_CACHE: Dict[str, LoadedModel] = {}
_CACHE_LOCK = Lock()


def _resolve_models_dir() -> Path:
    settings = get_settings()
    base = Path(settings.preference_models_dir)
    if not base.is_absolute():
        base = Path(__file__).resolve().parents[2] / base
    return base


def _load_latest_metadata() -> Dict[str, Any] | None:
    models_dir = _resolve_models_dir()
    latest_path = models_dir / "latest.json"
    if not latest_path.exists():
        return None
    try:
        return json.loads(latest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_model_from_metadata(metadata: Mapping[str, Any]) -> LoadedModel | None:
    model_path = metadata.get("model_path")
    if not model_path:
        return None
    models_dir = _resolve_models_dir()
    artifact_path = models_dir / model_path
    if not artifact_path.exists():
        return None
    cache_key = str(artifact_path.resolve())
    with _CACHE_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]
        artifact = load(artifact_path)
        feature_names = artifact.get("feature_names") if isinstance(artifact, dict) else None
        model = artifact.get("model") if isinstance(artifact, dict) else artifact
        if feature_names is None:
            # Assume alphabetical order of feature keys stored separately
            feature_names = metadata.get("feature_names") or []
        loaded = LoadedModel(feature_names=feature_names, model=model)
        _MODEL_CACHE[cache_key] = loaded
        return loaded


def _default_preference_score(match_score: float | None) -> float:
    if match_score is None:
        return 0.5
    normalized = max(0.0, min(1.0, match_score / 100.0))
    return normalized


def clear_model_cache() -> None:
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()


def predict_preference_score(
    user: User,
    candidate: User,
    scoring: Mapping[str, Any] | None = None,
) -> float:
    """Return the predicted probability that ``user`` will like ``candidate``."""

    metadata = _load_latest_metadata()
    match_score = None
    if scoring and isinstance(scoring, Mapping):
        match_score = float(scoring.get("total", 0.0))
    if not metadata:
        return _default_preference_score(match_score)

    loaded = _load_model_from_metadata(metadata)
    if loaded is None or loaded.model is None:
        return _default_preference_score(match_score)

    features = build_feature_vector(
        user.preferences,
        candidate.preferences,
        score=match_score,
        common_interests=None,
    )
    names = list(loaded.feature_names)
    if not names:
        names = sorted(features.keys())
    vector = [[float(features.get(name, 0.0)) for name in names]]
    try:
        proba = loaded.model.predict_proba(vector)[0][1]
        return float(proba)
    except Exception:
        return _default_preference_score(match_score)
