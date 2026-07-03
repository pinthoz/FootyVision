"""Market Value Predictor (LightGBM).

Target = FIFA 16 player value (SoFIFA), a widely-used market-value proxy that aligns
temporally with our StatsBomb La Liga 2015/16 features. Values are fuzzy-matched to our
players by name. We train on log(value) (values are heavy-tailed), evaluate on a held-out
split, explain with SHAP, and surface "bargains" (players worth more than their value
suggests, i.e. large positive residual in performance-implied value).
"""
from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rapidfuzz import fuzz, process
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from footyvision.ml.features import PER90_FEATURES


def _normalize_name(name: str) -> str:
    """Lowercase and strip accents so 'Suárez'/'Modrić' match their ASCII forms."""
    ascii_ = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return ascii_.lower().strip()


def match_values(
    features: pd.DataFrame,
    values: pd.DataFrame,
    keep_cols: tuple[str, ...] = ("value_eur",),
    score_cutoff: int = 90,
) -> pd.DataFrame:
    """Fuzzy-match `features.name` to `values.name`, attaching `keep_cols`.

    Cross-source name matching is imperfect (Spanish multi-surnames vs short Transfermarkt
    names), so we normalise accents and use a high cutoff — a wrong value label is worse
    than a missing one. `values` should be unique by name.
    """
    values = values.drop_duplicates("name")
    norm_to_orig: dict[str, str] = {}
    for original in values["name"]:
        norm_to_orig.setdefault(_normalize_name(original), original)
    choices = list(norm_to_orig)
    lookup = values.set_index("name")

    rows = []
    for _, r in features.iterrows():
        match = process.extractOne(
            _normalize_name(r["name"]), choices, scorer=fuzz.WRatio, score_cutoff=score_cutoff
        )
        if match is None:
            continue
        matched = norm_to_orig[match[0]]
        extra = {c: lookup.loc[matched, c] for c in keep_cols}
        rows.append({**r.to_dict(), "matched_name": matched, "match_score": match[1], **extra})
    return pd.DataFrame(rows)


@dataclass
class ValueModel:
    model: LGBMRegressor
    features: list[str]
    r2: float
    mae_eur: float
    n_train: int
    n_test: int
    value_col: str


def train_value_model(
    merged: pd.DataFrame,
    feature_cols: list[str] | None = None,
    value_col: str = "value_eur",
    seed: int = 42,
) -> ValueModel:
    """Train LightGBM on log(value); report R² and MAE (in €) on a held-out split.

    `feature_cols` defaults to the per-90 style features; pass e.g. PER90 + ['age'] to add
    age (a strong value driver). LightGBM handles missing values natively.
    """
    features = list(feature_cols) if feature_cols else list(PER90_FEATURES)
    data = merged[merged[value_col] > 0].copy()
    x = data[features].to_numpy(dtype=float)
    y = np.log1p(data[value_col].to_numpy(dtype=float))

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, random_state=seed)
    model = LGBMRegressor(
        n_estimators=400, learning_rate=0.03, num_leaves=15,
        subsample=0.9, colsample_bytree=0.9, random_state=seed, verbose=-1,
    )
    model.fit(x_tr, y_tr)

    pred_te = model.predict(x_te)
    r2 = float(r2_score(y_te, pred_te))
    mae_eur = float(mean_absolute_error(np.expm1(y_te), np.expm1(pred_te)))
    return ValueModel(model, features, r2, mae_eur, len(x_tr), len(x_te), value_col)


def predict_values(vm: ValueModel, merged: pd.DataFrame) -> pd.DataFrame:
    """Add `predicted_value` and `value_residual` (actual - predicted) columns."""
    out = merged.copy()
    pred_log = vm.model.predict(out[vm.features].to_numpy(dtype=float))
    out["predicted_value"] = np.expm1(pred_log).round(0)
    out["value_residual"] = (out[vm.value_col] - out["predicted_value"]).round(0)
    return out


def bargains(vm: ValueModel, merged: pd.DataFrame, top_n: int = 15) -> list[dict[str, Any]]:
    """Players the model rates far above their actual value (performance-implied bargains)."""
    scored = predict_values(vm, merged)
    scored = scored.sort_values("value_residual")  # most negative = underpriced vs performance
    rows = []
    for _, r in scored.head(top_n).iterrows():
        rows.append({
            "player_id": int(r["player_id"]), "name": r["name"],
            "position_group": r["position_group"],
            "actual_value": float(r[vm.value_col]),
            "predicted_value": float(r["predicted_value"]),
            "upside": float(-r["value_residual"]),
        })
    return rows


def shap_importance(vm: ValueModel, merged: pd.DataFrame, top_n: int = 10) -> list[dict[str, Any]]:
    """Global mean |SHAP| per feature for the value model."""
    import shap

    x = merged[vm.features].to_numpy(dtype=float)
    values = np.abs(np.array(shap.TreeExplainer(vm.model).shap_values(x)))
    feat_axis = next(ax for ax, size in enumerate(values.shape) if size == len(vm.features))
    other = tuple(ax for ax in range(values.ndim) if ax != feat_axis)
    mean_abs = values.mean(axis=other)
    ranked = sorted(zip(vm.features, mean_abs, strict=False), key=lambda kv: kv[1], reverse=True)
    return [{"feature": f, "mean_abs_shap": round(float(v), 4)} for f, v in ranked[:top_n]]


def parse_value_eur(raw: Any) -> float:
    """Parse a SoFIFA value ('€5M', '€900K', 5000000, '') into euros; 0 if unknown."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().replace("€", "").replace(",", "")
    if not s:
        return 0.0
    mult = 1.0
    if s[-1].upper() == "M":
        mult, s = 1_000_000.0, s[:-1]
    elif s[-1].upper() == "K":
        mult, s = 1_000.0, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return 0.0
