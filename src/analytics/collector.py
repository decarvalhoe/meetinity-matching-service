"""Utilities to collect interaction data for preference modelling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

import csv

from src.storage import fetch_swipe_events

PreferenceFeatures = Dict[str, float]


def _ensure_iterable(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return value
    return [value]


def _normalize_token(token: str, *, prefix: str | None = None) -> str:
    cleaned = token.strip().lower()
    if not cleaned:
        return ""
    if prefix:
        return f"{prefix}:{cleaned}"
    return cleaned


def _extract_tokens(preference: Any) -> Iterable[str]:
    if preference is None:
        return []
    if isinstance(preference, str):
        token = _normalize_token(preference)
        return [token] if token else []
    if isinstance(preference, Mapping):
        pref_type = str(preference.get("type") or preference.get("category") or "").strip()
        values = (
            preference.get("value")
            or preference.get("values")
            or preference.get("items")
            or preference.get("list")
        )
        tokens: List[str] = []
        for item in _ensure_iterable(values):
            normalized = _normalize_token(str(item))
            if not normalized:
                continue
            if pref_type:
                tokens.append(_normalize_token(normalized, prefix=pref_type))
            else:
                tokens.append(normalized)
        label = preference.get("label") or preference.get("name")
        if label:
            normalized = _normalize_token(str(label))
            if normalized:
                tokens.append(normalized)
        return tokens
    token = _normalize_token(str(preference))
    return [token] if token else []


def _collect_preference_tokens(preferences: Sequence[Any] | None) -> List[str]:
    tokens: List[str] = []
    if not preferences:
        return tokens
    for preference in preferences:
        for token in _extract_tokens(preference):
            if token and token not in tokens:
                tokens.append(token)
    return tokens


def build_feature_vector(
    user_preferences: Sequence[Any] | None,
    target_preferences: Sequence[Any] | None,
    *,
    score: float | None,
    common_interests: Sequence[str] | None = None,
) -> PreferenceFeatures:
    """Compute numerical features for a pair of preference lists."""

    user_tokens = set(_collect_preference_tokens(user_preferences))
    target_tokens = set(_collect_preference_tokens(target_preferences))
    union = user_tokens | target_tokens
    intersection = user_tokens & target_tokens

    shared_count = float(len(intersection))
    union_count = float(len(union))
    jaccard = (shared_count / union_count) if union_count else 0.0
    common_count = float(len(common_interests or []))

    return {
        "score": float(score or 0.0),
        "shared_preferences": shared_count,
        "preference_union": union_count,
        "preference_overlap": jaccard,
        "user_preference_count": float(len(user_tokens)),
        "target_preference_count": float(len(target_tokens)),
        "preference_count_delta": float(abs(len(user_tokens) - len(target_tokens))),
        "common_interest_count": common_count,
    }


@dataclass(slots=True)
class PreferenceSample:
    """Represents a single swipe event used for modelling preferences."""

    event_id: int
    user_id: int
    target_id: int
    label: int
    action: str
    is_match: bool
    created_at: datetime
    features: PreferenceFeatures

    def to_row(self, feature_names: Sequence[str]) -> Dict[str, Any]:
        row = {name: float(self.features.get(name, 0.0)) for name in feature_names}
        row["label"] = self.label
        row["is_match"] = int(self.is_match)
        row["user_id"] = self.user_id
        row["target_id"] = self.target_id
        row["event_id"] = self.event_id
        row["timestamp"] = self.created_at.isoformat()
        return row


class PreferenceCollector:
    """Collect swipe data points for machine learning training."""

    def __init__(self, events: Iterable[Mapping[str, Any]] | None = None):
        self._events = list(events) if events is not None else None

    def _iter_events(self) -> Iterator[Mapping[str, Any]]:
        if self._events is not None:
            yield from self._events
            return
        yield from fetch_swipe_events()

    def collect(self) -> List[PreferenceSample]:
        samples: List[PreferenceSample] = []
        for event in self._iter_events():
            if event.get("event_type") != "swipe":
                continue
            payload = event.get("payload") or {}
            user_prefs = payload.get("user_preferences")
            target_prefs = payload.get("target_preferences")
            features = build_feature_vector(
                user_prefs,
                target_prefs,
                score=event.get("score"),
                common_interests=payload.get("common_interests"),
            )
            created_at_text = event.get("created_at")
            created_at = (
                datetime.fromisoformat(created_at_text)
                if created_at_text
                else datetime.utcnow()
            )
            samples.append(
                PreferenceSample(
                    event_id=int(event.get("id")),
                    user_id=int(event.get("user_id")),
                    target_id=int(event.get("target_id")),
                    label=1 if event.get("action") == "like" else 0,
                    action=str(event.get("action")),
                    is_match=bool(payload.get("is_match")),
                    created_at=created_at,
                    features=features,
                )
            )
        return samples

    def feature_names(self) -> List[str]:
        samples = self.collect()
        if not samples:
            return []
        names = sorted(samples[0].features.keys())
        return names

    def export_csv(self, destination: Path, feature_names: Sequence[str] | None = None) -> None:
        samples = self.collect()
        if not samples:
            raise RuntimeError("No swipe events available to export.")
        names = list(feature_names or sorted(samples[0].features.keys()))
        header = [
            "event_id",
            "user_id",
            "target_id",
            "timestamp",
            "action",
            "label",
            "is_match",
            *names,
        ]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            for sample in samples:
                row = {
                    "event_id": sample.event_id,
                    "user_id": sample.user_id,
                    "target_id": sample.target_id,
                    "timestamp": sample.created_at.isoformat(),
                    "action": sample.action,
                    "label": sample.label,
                    "is_match": int(sample.is_match),
                }
                for name in names:
                    row[name] = float(sample.features.get(name, 0.0))
                writer.writerow(row)
