"""XGBoost position classifier — a genuinely supervised task with REAL labels.

We predict a player's position group from their per-90 style. This showcases the ML +
explainability (SHAP) stack honestly (no fabricated target) and yields two useful signals:
a per-player *style profile* (how FWD/MID/DEF-like they play) and *role mismatches*
(players whose stats look like a different position than they're listed in).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from footyvision.ml.features import FOOT_FEATURES, PER90_FEATURES
from footyvision.ml.similarity import _target_index

# Directory containing pre-trained model artifacts.
MODELS_DIR = Path(__file__).resolve().parents[3] / "models" / "talent"

# Position groups the classifier learns (Unknown excluded; tiny classes handled at fit).
_TRAIN_GROUPS = ("GK", "DEF", "MID", "FWD")


def classifier_features(frame: pd.DataFrame, target: str = "position_group") -> list[str]:
    """Per-90 style metrics, plus preferred foot only where a side is being predicted.

    Foot is the one feature that encodes a side, and the measurements say to use it
    exactly there and nowhere else:

    | target             | per-90 only | + foot |
    |--------------------|-------------|--------|
    | position_group (4) | 0.916       | 0.908  |
    | position_role (10) | 0.720       | 0.712  |
    | primary_position   | 0.392       | 0.489  |

    On the side-agnostic targets it is noise and costs accuracy; on the exact position it
    is worth ten points and cuts pure left/right confusions from 134 errors to 96. Height
    was ablated the same way and added nothing anywhere, so it is never modelled.

    Frames built by hand in tests have no foot columns, so they are added only if present.
    """
    features = list(PER90_FEATURES)
    if target == "primary_position":
        features += [c for c in FOOT_FEATURES if c in frame.columns]
    return features


@dataclass
class TalentModel:
    model: XGBClassifier
    classes: list[str]
    features: list[str]
    test_accuracy: float
    n_train: int
    n_test: int
    target: str = "position_group"


def train_position_classifier(
    frame: pd.DataFrame,
    test_size: float = 0.25,
    seed: int = 42,
    target: str = "position_group",
) -> TalentModel:
    """Fit the position classifier at whatever granularity `target` names.

    `position_group` gives the four broad groups; `position_role` gives the ten
    side-agnostic roles. The exact StatsBomb position is deliberately not shipped as a
    product surface: see `position_role` in ml/features.py for why its left/right half is
    not learnable from these features.
    """
    data = frame[frame[target] != "Unknown"].copy()
    if target == "position_group":
        data = data[data[target].isin(_TRAIN_GROUPS)]
    # A class with a single example cannot be split into train and test.
    counts_all = data[target].value_counts()
    data = data[data[target].isin(counts_all[counts_all >= 2].index)]

    classes = sorted(data[target].unique())
    code = {c: i for i, c in enumerate(classes)}
    features = classifier_features(data, target)
    x = data[features].to_numpy(dtype=float)
    y = data[target].map(code).to_numpy()

    # Stratify only if every class has at least two samples.
    counts = data[target].value_counts()
    stratify = y if counts.min() >= 2 else None
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        num_class=len(classes),
        random_state=seed,
    )
    model.fit(x_tr, y_tr)
    acc = float(accuracy_score(y_te, model.predict(x_te)))
    return TalentModel(model, classes, features, acc, len(x_tr), len(x_te), target)


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

    # Subsample to at most 150 rows: global SHAP rankings converge with ~100 samples,
    # while computing TreeExplainer over thousands of samples in multi-class consumes
    # substantial memory and CPU, risking cold-start OOM on constrained runtimes (e.g. 512MB).
    data = frame
    if len(frame) > 150:
        data = frame.sample(150, random_state=42)

    x = data[tm.features].to_numpy(dtype=float)
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
            out.append(
                {
                    "player_id": int(r["player_id"]),
                    "name": r["name"],
                    "listed": r["position_group"],
                    "plays_like": pred,
                    "confidence": round(conf, 3),
                }
            )
    out.sort(key=lambda m: m["confidence"], reverse=True)
    return out[:top_n]


# Trained lazily, persisted to disk, and cached in memory.
_CACHE: dict[str, TalentModel] = {}
_SHAP_CACHE: dict[str, list[dict[str, Any]]] = {}


def get_cached_model(frame: pd.DataFrame) -> TalentModel:
    if "model" not in _CACHE:
        disk_path = MODELS_DIR / "position_group.joblib"
        if disk_path.is_file():
            try:
                loaded = joblib.load(disk_path)
                if isinstance(loaded, TalentModel) and set(loaded.features).issubset(frame.columns):
                    _CACHE["model"] = loaded
            except Exception:
                pass
        if "model" not in _CACHE:
            _CACHE["model"] = train_position_classifier(frame)
            try:
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                joblib.dump(_CACHE["model"], disk_path, compress=3)
            except Exception:
                pass
    return _CACHE["model"]


def get_cached_role_model(frame: pd.DataFrame) -> TalentModel:
    """The finer-grained sibling: ten side-agnostic roles instead of four groups."""
    if "role" not in _CACHE:
        disk_path = MODELS_DIR / "position_role.joblib"
        if disk_path.is_file():
            try:
                loaded = joblib.load(disk_path)
                if isinstance(loaded, TalentModel) and set(loaded.features).issubset(frame.columns):
                    _CACHE["role"] = loaded
            except Exception:
                pass
        if "role" not in _CACHE:
            _CACHE["role"] = train_position_classifier(frame, target="position_role")
            try:
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                joblib.dump(_CACHE["role"], disk_path, compress=3)
            except Exception:
                pass
    return _CACHE["role"]


def get_cached_importance(
    tm: TalentModel, frame: pd.DataFrame, top_n: int = 5
) -> list[dict[str, Any]]:
    key = str(top_n)
    if key not in _SHAP_CACHE:
        disk_path = MODELS_DIR / "shap_importance.json"
        if disk_path.is_file():
            try:
                with open(disk_path, encoding="utf-8") as f:
                    stored = json.load(f)
                if isinstance(stored, list) and len(stored) >= top_n:
                    _SHAP_CACHE[key] = stored[:top_n]
            except Exception:
                pass
        if key not in _SHAP_CACHE:
            _SHAP_CACHE[key] = shap_importance(tm, frame, top_n=top_n)
            try:
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                with open(disk_path, "w", encoding="utf-8") as f:
                    json.dump(_SHAP_CACHE[key], f, indent=2)
            except Exception:
                pass
    return _SHAP_CACHE[key]
