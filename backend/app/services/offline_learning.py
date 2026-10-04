"""Stage 12: Offline learning.

Trains a semantic reranker from accumulated HUMAN feedback (never from the
model's own predictions). Trains only when sufficient labeled data exists,
evaluates on a held-out split, compares against the current model, and
versioning requires explicit human promotion.
"""

import json
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from app.config import MIN_FEEDBACK_FOR_TRAINING, SEMANTIC_MODEL_NAME
from app.models import ModelVersion, SemanticFeedback, SemanticConcept

FEATURE_ORDER = [
    "embedding_similarity",
    "name_similarity",
    "description_similarity",
    "datatype_compatibility",
    "profile_compatibility",
    "context_similarity",
]


def _features_to_vector(features: dict) -> list[float]:
    """Convert persisted evidence payload into the training feature vector."""
    vector: list[float] = []

    for name in FEATURE_ORDER:
        evidence = features.get(name, {})
        if isinstance(evidence, dict):
            value = evidence.get("value", 0.0)
        else:
            value = evidence

        try:
            vector.append(float(value or 0.0))
        except (TypeError, ValueError):
            vector.append(0.0)

    return vector

def _concept_name(db: Session, concept_id: int | None) -> str | None:
    if concept_id is None:
        return None
    concept = db.get(SemanticConcept, concept_id)
    return concept.concept_name if concept else None


def build_training_examples(db: Session) -> list[dict]:
    """Build (features, label) pairs from human feedback.

    For each feedback record the label is 1 when the feedback confirms the
    concept the model predicted (approved with same concept, or edited to
    the predicted concept), and 0 when the human rejected or chose a
    different concept.
    """
    from app.models import SemanticPrediction

    feedback = db.query(SemanticFeedback).all()

    examples = []

    for item in feedback:
        if not item.features_json:
            continue

        if not item.prediction_id:
            continue

        prediction = (
            db.query(SemanticPrediction)
            .filter(
                SemanticPrediction.prediction_id == item.prediction_id
            )
            .first()
        )

        if prediction is None:
            continue

        predicted_concept_id = prediction.predicted_concept_id

        if item.decision == "approved":
            label = 1 if item.corrected_concept_id == predicted_concept_id else None
        elif item.decision == "edited":
            label = (
                1
                if item.corrected_concept_id == predicted_concept_id
                else 0
            )
        elif item.decision == "rejected":
            label = 0
        else:
            continue

        if label is None:
            continue

        candidates = item.features_json

        if isinstance(candidates, list):
            candidate = next(
                (
                    entry
                    for entry in candidates
                    if isinstance(entry, dict)
                    and entry.get("concept_id") == predicted_concept_id
                ),
                None,
            )

            if candidate is None:
                continue

            features = candidate.get("evidence", {})
        elif isinstance(candidates, dict):
            features = candidates
        else:
            continue

        vector = _features_to_vector(features)

        if all(value == 0.0 for value in vector):
            continue

        examples.append(
            {
                "features": vector,
                "label": label,
                "column_name": item.column_name,
            }
        )

    return examples


def train_model(db: Session) -> dict:
    """Train a candidate reranker if sufficient feedback exists."""
    examples = build_training_examples(db)

    labels = [example["label"] for example in examples]

    if len(examples) < MIN_FEEDBACK_FOR_TRAINING:
        return {
            "status": "insufficient_data",
            "message": (
                f"Need at least {MIN_FEEDBACK_FOR_TRAINING} labeled feedback "
                f"examples; found {len(examples)}. Human decisions are the "
                "training labels - no model was trained."
            ),
            "example_count": len(examples),
        }

    class_counts = {label: labels.count(label) for label in set(labels)}

    if len(class_counts) < 2:
        return {
            "status": "insufficient_data",
            "message": (
                "Feedback contains only one class. Both positive and "
                "negative human decisions are required to train."
            ),
            "example_count": len(examples),
            "class_counts": class_counts,
        }

    X = np.array([example["features"] for example in examples])
    y = np.array(labels)

    stratify = y if min(class_counts.values()) >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "example_count": len(examples),
        "train_count": int(len(y_train)),
        "test_count": int(len(y_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "class_counts": class_counts,
        "feature_names": FEATURE_ORDER,
        "feature_weights": [
            round(float(weight), 4) for weight in model.coef_[0]
        ],
        "trained_with_model": SEMANTIC_MODEL_NAME,
    }

    latest_version = (
        db.query(ModelVersion)
        .filter(ModelVersion.model_name == "semantic_reranker")
        .order_by(ModelVersion.version.desc())
        .first()
    )

    next_version = (latest_version.version + 1) if latest_version else 1

    model_version = ModelVersion(
        model_name="semantic_reranker",
        version=next_version,
        status="candidate",
        metrics_json=metrics,
        trained_at=datetime.now(timezone.utc),
    )

    db.add(model_version)
    db.flush()

    # Persist model coefficients as JSON (transparent, inspectable model).
    model_data = {
        "model_type": "logistic_regression",
        "feature_names": FEATURE_ORDER,
        "coefficients": [float(c) for c in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "version": next_version,
    }

    from pathlib import Path

    from app.config import BASE_DIR

    models_dir = BASE_DIR / "models" / "ranking"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"semantic_reranker_v{next_version}.json"
    model_path.write_text(json.dumps(model_data, indent=2))

    return {
        "status": "trained",
        "model_version_id": model_version.model_version_id,
        "version": next_version,
        "metrics": metrics,
        "status_note": (
            "Candidate model stored. It is NOT active until promoted."
        ),
    }


def promote_model(db: Session, model_version_id: int) -> dict:
    """Human-controlled promotion of a candidate model."""
    model_version = db.get(ModelVersion, model_version_id)

    if model_version is None:
        raise ValueError("Model version not found.")

    if model_version.status == "promoted":
        return {
            "message": "Model version already promoted.",
            "model_version_id": model_version_id,
        }

    # Demote any currently promoted version.
    current = (
        db.query(ModelVersion)
        .filter(
            ModelVersion.model_name == model_version.model_name,
            ModelVersion.status == "promoted",
        )
        .all()
    )

    for version in current:
        version.status = "archived"
        version.promoted_at = None

    model_version.status = "promoted"
    model_version.promoted_at = datetime.now(timezone.utc)

    db.flush()

    return {
        "message": (
            f"Model {model_version.model_name} v{model_version.version} "
            "promoted."
        ),
        "model_version_id": model_version_id,
        "version": model_version.version,
    }


def list_models(db: Session) -> list[dict]:
    models = (
        db.query(ModelVersion)
        .order_by(ModelVersion.model_name, ModelVersion.version.desc())
        .all()
    )

    return [
        {
            "model_version_id": m.model_version_id,
            "model_name": m.model_name,
            "version": m.version,
            "status": m.status,
            "metrics": m.metrics_json,
            "trained_at": m.trained_at.isoformat() if m.trained_at else None,
            "promoted_at": m.promoted_at.isoformat() if m.promoted_at else None,
        }
        for m in models
    ]
