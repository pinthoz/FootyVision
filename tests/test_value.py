"""Unit tests for the market-value predictor — synthetic data, no scraping."""

from __future__ import annotations

import numpy as np
import pandas as pd

from footyvision.ml.features import PER90_FEATURES
from footyvision.ml.value import match_values, parse_value_eur, train_value_model


def test_parse_value_eur():
    assert parse_value_eur("€5M") == 5_000_000
    assert parse_value_eur("€900K") == 900_000
    assert parse_value_eur("€27.5M") == 27_500_000
    assert parse_value_eur(4_000_000) == 4_000_000
    assert parse_value_eur("") == 0.0
    assert parse_value_eur(None) == 0.0


def test_match_values_fuzzy():
    features = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "name": ["Lionel Messi", "Luis Suárez", "Nobody Here"],
            "position_group": ["FWD", "FWD", "MID"],
        }
    )
    values = pd.DataFrame(
        {
            "name": ["Lionel Andrés Messi", "Luis Suarez", "Cristiano Ronaldo"],
            "value_eur": [120_000_000.0, 80_000_000.0, 100_000_000.0],
        }
    )
    merged = match_values(features, values, score_cutoff=80)
    by_name = dict(zip(merged["name"], merged["value_eur"], strict=False))
    assert by_name["Lionel Messi"] == 120_000_000.0
    assert by_name["Luis Suárez"] == 80_000_000.0
    assert "Nobody Here" not in by_name  # no confident match


def _merged(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        feats = {f: float(rng.random()) for f in PER90_FEATURES}
        # Value driven mostly by xg + progressive passes, plus noise.
        value = 1e6 + 4e7 * feats["xg_per90"] + 2e7 * feats["progressive_passes_per90"]
        value *= 1 + rng.normal(0, 0.1)
        rows.append(
            {
                "player_id": i,
                "name": f"P{i}",
                "position_group": "FWD",
                "value_eur": max(1e5, value),
                **feats,
            }
        )
    return pd.DataFrame(rows)


def test_train_value_model_runs_and_reports_metrics():
    vm = train_value_model(_merged())
    assert vm.features == list(PER90_FEATURES)
    assert vm.n_train > 0 and vm.n_test > 0
    assert np.isfinite(vm.r2)
    assert vm.mae_eur >= 0
