"""Train a preference prediction model using stored swipe events."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.collector import PreferenceCollector
from src.config import get_settings, refresh_settings


def _resolve_models_dir(directory: str | None = None) -> Path:
    if directory:
        base = Path(directory)
    else:
        settings = get_settings()
        base = Path(settings.preference_models_dir)
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    base.mkdir(parents=True, exist_ok=True)
    return base


def _prepare_dataset(samples, feature_names: Sequence[str] | None = None):
    if not samples:
        raise RuntimeError("The collector did not return any samples.")
    names = list(feature_names or sorted(samples[0].features.keys()))
    X = np.array(
        [[float(sample.features.get(name, 0.0)) for name in names] for sample in samples],
        dtype=float,
    )
    y = np.array([sample.label for sample in samples], dtype=int)
    return X, y, names


def train_preference_model(
    *,
    output_dir: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Collect swipe data and train a logistic regression model."""

    collector = PreferenceCollector()
    samples = collector.collect()
    if len(samples) < 6:
        raise RuntimeError("At least 6 swipe events are required to train a model.")

    X, y, feature_names = _prepare_dataset(samples)

    try:
        stratify = y if len(set(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        # Fallback to a simple split without stratification when data is scarce.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        roc_auc = float("nan")

    models_dir = _resolve_models_dir(output_dir)
    version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    model_filename = f"preference_model_{version}.joblib"
    artifact_path = models_dir / model_filename
    dump({"model": model, "feature_names": feature_names}, artifact_path)

    metadata = {
        "version": version,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_path": model_filename,
        "feature_names": feature_names,
        "metrics": {
            "accuracy": float(accuracy),
            "roc_auc": float(roc_auc),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "total_samples": len(samples),
        },
    }

    metadata_path = models_dir / f"preference_model_{version}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    latest_path = models_dir / "latest.json"
    latest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Optional directory where the model artifacts will be stored.",
    )
    parser.add_argument(
        "--test-size",
        dest="test_size",
        type=float,
        default=0.2,
        help="Fraction of the dataset reserved for evaluation (default: 0.2).",
    )
    parser.add_argument(
        "--refresh-settings",
        action="store_true",
        help="Reload environment based settings before training.",
    )
    args = parser.parse_args(argv)

    if args.refresh_settings:
        refresh_settings()

    try:
        metadata = train_preference_model(
            output_dir=args.output_dir,
            test_size=args.test_size,
        )
    except RuntimeError as exc:
        print(f"Training aborted: {exc}")
        return 1

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
