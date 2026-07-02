"""XGBoost position classifier — a genuinely supervised task with REAL labels.

We predict a player's position group from their per-90 style. This showcases the ML +
explainability (SHAP) stack honestly (no fabricated target) and yields two useful signals:
a per-player *style profile* (how FWD/MID/DEF-like they play) and *role mismatches*
(players whose stats look like a different position than they're listed in).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from footyvision.ml.features import PER90_FEATURES
from footyvision.ml.similarity import _target_index

# Position groups the classifier learns (Unknown excluded; tiny classes handled at fit).
_TRAIN_GROUPS = ("GK", "DEF", "MID", "FWD")


@dataclass
class TalentModel:
    model: XGBClassifier
    classes: list[str]
    features: list[str]
    test_accuracy: float
    n_train: int
    n_test: int


def train_position_classifier(
    frame: pd.DataFrame, test_size: float = 0.25, seed: int = 42
) -> TalentModel:
    data = frame[frame["position_group"].isin(_TRAIN_GROUPS)].copy()
    classes = sorted(data["position_group"].unique())
    code = {c: i for i, c in enumerate(classes)}
    x = data[list(PER90_FEATURES)].to_numpy(dtype=float)
    y = data["position_group"].map(code).to_numpy()

    # Stratify only if every class has at least two samples.
    counts = data["position_group"].value_counts()
    stratify = y if counts.min() >= 2 else None
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss",
        num_class=len(classes), random_state=seed,
    )
    model.fit(x_tr, y_tr)
    acc = float(accuracy_score(y_te, model.predict(x_te)))
    return TalentModel(model, classes, list(PER90_FEATURES), acc, len(x_tr), len(x_te))


def style_profile(tm: TalentModel, frame: pd.DataFrame, player_id: int) -> dict[str, float] | None:
    """Predicted probability the player belongs to each position group ('style fingerprint')."""
    idx = _target_index(frame, player_id)
    if idx is None:
        return None
    x = frame.loc[[idx], tm.features].to_numpy(dtype=float)
    proba = tm.model.predict_proba(x)[0]
    return {cls: round(float(p), 3) for cls, p in zip(tm.classes, proba, strict=False)}


def shap_importance(tm: TalentModel, frame: pd.DataFrame, top_n: int = 10) -> list[dict[str, Any]]:
    """Global mean |SHAP| importance per feature (which metrics define position)."""
    import shap

    x = frame[tm.features].to_numpy(dtype=float)
    values = np.abs(np.array(shap.TreeExplainer(tm.model).shap_values(x)))
    # Reduce over every axis except the feature axis, wherever it lands.
    feat_axis = next(ax for ax, size in enumerate(values.shape) if size == len(tm.features))
    other = tuple(ax for ax in range(values.ndim) if ax != feat_axis)
    mean_abs = values.mean(axis=other)
    ranked = sorted(zip(tm.features, mean_abs, strict=False), key=lambda kv: kv[1], reverse=True)
    return [{"feature": f, "mean_abs_shap": round(float(v), 4)} for f, v in ranked[:top_n]]


def role_mismatches(
    tm: TalentModel, frame: pd.DataFrame, min_prob: float = 0.6, top_n: int = 15
) -> list[dict[str, Any]]:
    """Players whose predicted position differs from their listed one (confidently)."""
    data = frame[frame["position_group"].isin(tm.classes)].copy()
    proba = tm.model.predict_proba(data[tm.features].to_numpy(dtype=float))
    pred_idx = proba.argmax(axis=1)
    out: list[dict[str, Any]] = []
    for row_pos, (_, r) in enumerate(data.iterrows()):
        pred = tm.classes[pred_idx[row_pos]]
        conf = float(proba[row_pos, pred_idx[row_pos]])
        if pred != r["position_group"] and conf >= min_prob:
            out.append({
                "player_id": int(r["player_id"]), "name": r["name"],
                "listed": r["position_group"], "plays_like": pred, "confidence": round(conf, 3),
            })
    out.sort(key=lambda m: m["confidence"], reverse=True)
    return out[:top_n]


# Trained lazily and cached — the loaded dataset is static within a process.
_CACHE: dict[str, TalentModel] = {}


def get_cached_model(frame: pd.DataFrame) -> TalentModel:
    if "model" not in _CACHE:
        _CACHE["model"] = train_position_classifier(frame)
    return _CACHE["model"]
