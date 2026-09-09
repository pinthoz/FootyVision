"""Market value as a range, and the players the market appears to have underpriced.

Everything here is deliberately awkward to read as a valuation, because it is not one.
The model explains about a fifth of the variation in euros and beats "predict the median
for everybody" by roughly eight hundred thousand of them — a real margin, and a narrow
one. So no response carries a bare number: the band, the share of held-out players that
band actually caught, and the error of the do-nothing baseline travel with every answer,
and `/value/bargains` marks which rows survive the model's own uncertainty.

The training population is men in the four leagues Transfermarkt 2015/16 covers. A request
for anyone else is refused rather than answered, because the per-90 features of a Liga F
forward will happily produce a number from a model that has never seen a women's league
and holds no label from one.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from footyvision.api.schemas import PlayerValue, ValueBargain, ValueBargains, ValueModelInfo
from footyvision.db.base import get_session

# No import of footyvision.ml.value here, and so none of lightgbm. What the endpoints
# serve is a table of already-priced players plus a handful of scalars, and unpickling a
# fitted LGBMRegressor to read them would drag the library into a container that has no
# room for it. The training code is imported lazily, in the one function that fits.

router = APIRouter(tags=["value"])

# The trained model and the frame it priced, written here by `footyvision value-report`.
# This is not just a speed cache. The training labels come from two Kaggle CSVs totalling
# 47MB that are gitignored and therefore absent from every deployment, so in production
# this artifact is the only way these endpoints can answer at all.
ARTIFACT = Path(__file__).resolve().parents[3] / "models" / "value" / "value_model.joblib"

# Training walks the feature frame, fuzzy-matches five thousand names and fits three
# LightGBM models — several seconds, far too slow per request and static between imports.
_CACHE: dict[str, object] = {}


def _train(session: Session):
    """Fit from the Kaggle CSVs. Only possible where they are present, i.e. locally."""
    from footyvision.config import get_settings
    from footyvision.etl.transfermarkt import read_market_values_2016
    from footyvision.ml.features import PER90_FEATURES, cached_feature_frame
    from footyvision.ml.value import match_values, predict_values, train_value_model

    features = cached_feature_frame(session, min_minutes=get_settings().min_minutes)
    # Men only. The value source is four men's leagues, so a women's-league player can
    # only ever match a man of a similar name — 76 of them did before this guard, and
    # every one of those labels was somebody else's.
    if "gender" in features.columns:
        features = features[features["gender"] == "male"]
    merged = match_values(features, read_market_values_2016(), keep_cols=("value_eur",))
    model = train_value_model(merged, feature_cols=[*PER90_FEATURES, "age"])
    return _metrics_of(model), predict_values(model, merged)


def _metrics_of(model) -> dict:
    """The scalars the API reports, lifted out of the fitted model.

    Stored instead of the model itself so that reading them back needs no lightgbm: what
    is served is a number, and a number does not need the machine that produced it.
    """
    return {
        "features": model.features,
        "r2_log": round(model.r2, 3),
        "r2_eur": round(model.r2_eur, 3),
        "mae_eur": round(model.mae_eur, 0),
        "baseline_mae_eur": round(model.baseline_mae_eur, 0),
        "n_train": model.n_train,
        "n_test": model.n_test,
        "interval_coverage": (
            round(model.interval_coverage, 3) if model.interval_coverage is not None else None
        ),
    }


def _fitted(session: Session):
    """The saved model and priced frame; fitted from the CSVs when they are to hand.

    Unlike the talent models, which retrain silently when their artifact is stale, this one
    cannot: the labels are not in the repository. A deployment without the artifact has no
    way to produce an answer, so it says so with a 503 rather than raising a FileNotFound
    from inside a request.
    """
    if "model" not in _CACHE:
        if ARTIFACT.is_file():
            try:
                model, priced = joblib.load(ARTIFACT)
                _CACHE["model"], _CACHE["priced"] = model, priced
            except Exception:
                pass
    if "model" not in _CACHE:
        try:
            _CACHE["model"], _CACHE["priced"] = _train(session)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The value model is unavailable: its training labels are two Kaggle "
                    "CSVs that are not part of the repository, and no pre-fitted artifact "
                    "was deployed. Run `footyvision value-report` where the data is "
                    "present to write one."
                ),
            ) from exc
    return _CACHE["model"], _CACHE["priced"]


def save_artifact(model, priced) -> None:
    """Persist the metrics and the priced table so a deployment can answer without either
    the Kaggle CSVs or lightgbm."""
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((_metrics_of(model), priced), ARTIFACT)


def _model_info(metrics: dict) -> ValueModelInfo:
    return ValueModelInfo(**metrics)


@router.get("/value/model-info", response_model=ValueModelInfo)
def model_info(session: Session = Depends(get_session)) -> ValueModelInfo:
    """What the value model is worth, on both scales and against the do-nothing baseline."""
    vm, _ = _fitted(session)
    return _model_info(vm)


@router.get("/players/{player_id}/value", response_model=PlayerValue)
def player_value(player_id: int, session: Session = Depends(get_session)) -> PlayerValue:
    """A predicted price range for one player.

    404 rather than a number for anyone outside the trained population — the women's
    competitions, and any man the value source does not list. The features exist for them
    and the model would answer happily; the answer would just be unfounded.
    """
    vm, priced = _fitted(session)
    rows = priced[priced["player_id"] == player_id]
    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "No value estimate for this player. The model is trained on men in the "
                "four leagues Transfermarkt 2015/16 covers, and only players matched to "
                "one of those values are priced."
            ),
        )
    # A player with two seasons has two rows; price the one he played most of.
    if "minutes" in rows:
        rows = rows.sort_values("minutes", ascending=False)
    row = rows.iloc[0]

    market = float(row["value_eur"]) if float(row["value_eur"] or 0) > 0 else None
    return PlayerValue(
        player_id=player_id,
        name=str(row["name"]),
        position_group=str(row["position_group"]),
        predicted_eur=float(row["predicted_value"]),
        predicted_low_eur=float(row["predicted_low"]),
        predicted_high_eur=float(row["predicted_high"]),
        market_value_eur=market,
        residual_eur=(None if market is None else float(row["value_residual"])),
        model=_model_info(vm),
    )


@router.get("/value/bargains", response_model=ValueBargains)
def value_bargains(
    session: Session = Depends(get_session),
    top_n: int = Query(15, ge=1, le=100),
    clearing_band_only: bool = Query(
        False,
        description=(
            "Keep only players whose market value falls below the bottom of the predicted "
            "band, i.e. where the gap is larger than the model's own uncertainty."
        ),
    ),
) -> ValueBargains:
    """Players the model prices furthest above what the market paid."""
    vm, priced = _fitted(session)
    scored = priced[priced["value_eur"] > 0].copy()
    scored["clears_band"] = scored["value_eur"] < scored["predicted_low"]
    if clearing_band_only:
        scored = scored[scored["clears_band"]]
    # Most negative residual first: market value furthest below what performance implies.
    scored = scored.sort_values("value_residual").head(top_n)

    return ValueBargains(
        results=[
            ValueBargain(
                player_id=int(r["player_id"]),
                name=str(r["name"]),
                position_group=str(r["position_group"]),
                market_value_eur=float(r["value_eur"]),
                predicted_eur=float(r["predicted_value"]),
                predicted_low_eur=float(r["predicted_low"]),
                predicted_high_eur=float(r["predicted_high"]),
                upside_eur=float(-r["value_residual"]),
                clears_band=bool(r["clears_band"]),
            )
            for _, r in scored.iterrows()
        ],
        model=_model_info(vm),
    )
