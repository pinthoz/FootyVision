"""What the deployed API may load, and the artifacts it cannot run without.

The API used to import scikit-learn, xgboost and lightgbm and load three fitted models:
405MB warm and 505MB at peak against a 512MB container, which is how it came to be
restarted under load. It now serves their outputs from files under models/ and imports
none of them. Nothing but these tests stops a single misplaced import from quietly
putting the 250MB back.
"""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from footyvision.ml import precompute

# The libraries the `train` extra carries. A deployment installs none of them.
TRAINING_ONLY = ("sklearn", "xgboost", "lightgbm", "shap", "rapidfuzz")


def _loaded_after(code: str) -> set[str]:
    """Run `code` in a fresh interpreter and report which training modules it loaded.

    A subprocess because this one has already imported all of them for other tests.
    """
    probe = f"{code}\nimport sys\nprint(','.join(m for m in {TRAINING_ONLY!r} if m in sys.modules))"
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    return {m for m in out.splitlines()[-1].split(",") if m} if out else set()


def test_importing_the_api_loads_no_training_library():
    assert _loaded_after("import footyvision.api.main") == set()


def test_loading_sofifa_data_needs_no_model_library():
    """Its value parser used to live in ml/value.py, so reading a CSV imported lightgbm."""
    assert _loaded_after("import footyvision.etl.sofifa") == set()


def test_the_value_artifact_reads_back_without_lightgbm():
    """It used to hold a pickled LGBMRegressor, so reading eight scalars imported lightgbm."""
    from footyvision.api.routers.value import ARTIFACT

    assert ARTIFACT.is_file(), f"missing {ARTIFACT}: run `footyvision value-report`"
    assert _loaded_after(f"import joblib; joblib.load({str(ARTIFACT)!r})") == set()


def test_published_predictions_carry_what_the_api_reads():
    payload = precompute.load()
    assert payload is not None, f"missing {precompute.ARTIFACT}: run `footyvision precompute`"
    assert set(payload["models"]) == set(precompute.TARGETS)
    for meta in payload["models"].values():
        assert {"classes", "test_accuracy", "balanced_accuracy", "per_class_recall"} <= set(meta)
    assert payload["players"] and payload["top_features"]
    assert {"db_matches", "played_in", "None"} <= set(payload.get("teams", {}))


def _stale_payload() -> dict:
    return {"built_on": "2026-01-01", "pool": 99_999, "players": {}, "models": {}}


def test_a_stale_file_is_served_and_marked_when_refitting_is_impossible(monkeypatch):
    """In production the training extra is absent. Refitting there would cost 250MB and
    restart the instance, so the old answers are served with a flag saying so."""
    from footyvision.api.routers import talent as router

    def cannot_train(frame):
        raise ImportError("No module named 'xgboost'")

    monkeypatch.setattr(router.precompute, "load", _stale_payload)
    monkeypatch.setattr(router.precompute, "build", cannot_train)
    monkeypatch.setattr(router, "_PREDICTIONS", {})

    served = router._predictions(pd.DataFrame({"player_id": [1, 2, 3]}))
    assert served["stale"] is True
    assert served["built_on"] == "2026-01-01"


def test_no_file_and_no_training_stack_is_a_503_not_a_crash(monkeypatch):
    from fastapi import HTTPException

    from footyvision.api.routers import talent as router

    def cannot_train(frame):
        raise ImportError("No module named 'xgboost'")

    monkeypatch.setattr(router.precompute, "load", lambda: None)
    monkeypatch.setattr(router.precompute, "build", cannot_train)
    monkeypatch.setattr(router, "_PREDICTIONS", {})

    with pytest.raises(HTTPException) as refused:
        router._predictions(pd.DataFrame({"player_id": [1]}))
    assert refused.value.status_code == 503
    assert "footyvision precompute" in refused.value.detail
