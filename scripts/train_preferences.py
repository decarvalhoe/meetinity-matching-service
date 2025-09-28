"""Train a preference prediction model using stored swipe events."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.collector import PreferenceCollector
from src.config import get_settings, refresh_settings


@dataclass
class _LogisticModel:
    """Lightweight logistic regression implementation.

    The model stores a bias term and one weight per feature.  Predictions are
    performed using the logistic sigmoid without relying on external
    dependencies such as NumPy or scikit-learn.
    """

    weights: Sequence[float]
    bias: float

    def predict_proba(self, matrix: Iterable[Sequence[float]]) -> list[list[float]]:
        results: list[list[float]] = []
        for row in matrix:
            z = self.bias
            for weight, value in zip(self.weights, row):
                z += weight * value
            # Clamp to avoid math range errors when z is large in magnitude
            z = max(-700.0, min(700.0, z))
            proba = 1.0 / (1.0 + math.exp(-z))
            results.append([1.0 - proba, proba])
        return results


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


def _prepare_dataset(
    samples, feature_names: Sequence[str] | None = None
) -> Tuple[list[list[float]], list[int], list[str]]:
    if not samples:
        raise RuntimeError("The collector did not return any samples.")
    names = list(feature_names or sorted(samples[0].features.keys()))
    matrix: list[list[float]] = []
    labels: list[int] = []
    for sample in samples:
        matrix.append([float(sample.features.get(name, 0.0)) for name in names])
        labels.append(int(sample.label))
    return matrix, labels, names


def _split_dataset(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    test_size: float,
    random_state: int,
) -> Tuple[
    list[list[float]],
    list[list[float]],
    list[int],
    list[int],
]:
    total = len(features)
    if total != len(labels):
        raise ValueError("Features and labels must have the same length.")
    indices = list(range(total))
    rng = random.Random(random_state)
    rng.shuffle(indices)
    test_count = max(1, int(total * test_size)) if total > 1 else 0
    test_indices = set(indices[:test_count])
    X_train: list[list[float]] = []
    X_test: list[list[float]] = []
    y_train: list[int] = []
    y_test: list[int] = []
    for idx, feature_row in enumerate(features):
        if idx in test_indices and len(X_test) < test_count:
            X_test.append(list(feature_row))
            y_test.append(int(labels[idx]))
        else:
            X_train.append(list(feature_row))
            y_train.append(int(labels[idx]))
    if not X_test:
        # Ensure there is at least one sample in the test set when possible.
        X_test.append(list(X_train[-1]))
        y_test.append(y_train[-1])
        X_train = X_train[:-1]
        y_train = y_train[:-1]
    return X_train, X_test, y_train, y_test


def _train_logistic_regression(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    learning_rate: float = 0.1,
    epochs: int = 300,
) -> _LogisticModel:
    if not features:
        raise RuntimeError("Cannot train a model without data.")
    feature_count = len(features[0])
    weights = [0.0] * feature_count
    bias = 0.0
    n_samples = len(features)
    for _ in range(max(1, epochs)):
        gradient_w = [0.0] * feature_count
        gradient_b = 0.0
        for row, label in zip(features, labels):
            z = bias
            for weight, value in zip(weights, row):
                z += weight * value
            z = max(-700.0, min(700.0, z))
            prediction = 1.0 / (1.0 + math.exp(-z))
            error = prediction - label
            for idx, value in enumerate(row):
                gradient_w[idx] += error * value
            gradient_b += error
        scale = learning_rate / float(n_samples)
        for idx in range(feature_count):
            weights[idx] -= scale * gradient_w[idx]
        bias -= scale * gradient_b
    return _LogisticModel(weights=weights, bias=bias)


def _predict_probabilities(
    model: _LogisticModel, features: Sequence[Sequence[float]]
) -> list[float]:
    probabilities = []
    for _, proba in model.predict_proba(features):
        probabilities.append(proba)
    return probabilities


def _accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        return float("nan")
    correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
    return correct / len(y_true)


def _roc_auc_score(y_true: Sequence[int], y_scores: Sequence[float]) -> float:
    positives = sum(1 for value in y_true if value == 1)
    negatives = sum(1 for value in y_true if value == 0)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranked = sorted(zip(y_scores, y_true))
    rank_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            rank_sum += rank
    auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return auc


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

    X_train, X_test, y_train, y_test = _split_dataset(
        X, y, test_size=test_size, random_state=random_state
    )

    model = _train_logistic_regression(X_train, y_train)

    y_pred_proba = _predict_probabilities(model, X_test)
    y_pred = [1 if value >= 0.5 else 0 for value in y_pred_proba]

    accuracy = _accuracy_score(y_test, y_pred)
    roc_auc = _roc_auc_score(y_test, y_pred_proba)

    models_dir = _resolve_models_dir(output_dir)
    version = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    model_filename = f"preference_model_{version}.json"
    artifact_path = models_dir / model_filename
    artifact = {
        "weights": list(model.weights),
        "bias": float(model.bias),
        "feature_names": feature_names,
        "type": "logistic_regression",
    }
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    metadata = {
        "version": version,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
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

    metadata_path = models_dir / f"preference_model_{version}.meta.json"
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
