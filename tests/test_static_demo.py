"""The static demo must show exactly what the API would answer (shipped artifact)."""

import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from vessel_eta import artifact
from vessel_eta.serving.app import STATIC_DIR, create_app
from vessel_eta.serving.predictor import Predictor, Static

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build", ROOT / "scripts" / "build_static_demo.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


@pytest.mark.parametrize("n_points", [30, 120])
def test_precomputed_forecast_equals_api_answer(n_points):
    sample = json.loads((STATIC_DIR / "samples.json").read_text())["samples"][0]
    track = sample["track"][:n_points]

    with TestClient(create_app(artifact.DEFAULT_DIR)) as client:
        api = client.post("/v1/eta", json={"ship": sample["ship"], "track": track}).json()

    points = pl.DataFrame(track).with_columns(
        pl.col("ts").str.to_datetime(time_zone="UTC"),
        pl.col("sog", "cog").cast(pl.Float32),
        pl.col("heading", "status").cast(pl.Int16),
    )
    static = build.compact(Predictor.load().predict(points, Static(**sample["ship"])))
    assert static == build.compact(api)


def test_static_page_switches_config():
    html = (STATIC_DIR / "index.html").read_text()
    assert build.API_CONFIG in html  # the build replaces exactly this line
