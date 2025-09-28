"""Dataclass-based models used by the storage layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class User:
    """Application user participating in the matching flow."""

    id: Optional[int]
    email: str
    full_name: str
    title: Optional[str] = None
    company: Optional[str] = None
    bio: Optional[str] = None
    preferences: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)


@dataclass(slots=True)
class Swipe:
    """Individual swipe decision between two users."""

    id: Optional[int]
    user_id: int
    target_id: int
    action: str
    created_at: datetime = field(default_factory=_utcnow)


@dataclass(slots=True)
class Match:
    """Represents a match between a user and another profile."""

    id: Optional[int]
    user_id: int
    matched_user_id: int
    created_at: datetime = field(default_factory=_utcnow)


@dataclass(slots=True)
class MatchScore:
    """Score associated with a specific match."""

    id: Optional[int]
    match_id: int
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)


@dataclass(slots=True)
class SwipeEvent:
    """Analytical event recorded for each swipe."""

    id: Optional[int]
    swipe_id: Optional[int]
    event_type: str
    user_id: int
    target_id: int
    action: str
    score: Optional[float] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)

