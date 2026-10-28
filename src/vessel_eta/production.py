"""Train the production model on all labelled data, and re-calibrate it later.

- `train_production`: v2 with the base features (the configuration that won every
  comparison in the README) on every labelled arrival; 20% of ships are held out
  for early stopping and the conformal step.
- `recalibrate`: re-estimate only the interval shifts from the most recent arrivals,
  as the 2024 out-of-time test showed intervals go stale long before the model does.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from vessel_eta import artifact
from vessel_eta.config import PROCESSED_DIR
from vessel_eta.decompose import DecomposedETA, add_targets, conformal_widen, waiting_episodes
from vessel_eta.experiment import load_features
from vessel_eta.features import BASE_FEATURES
from vessel_eta.geo import Port
from vessel_eta.modeling import make_split

FAR_FUTURE = dt.datetime(2100, 1, 1, tzinfo=dt.UTC)
NO_WIDEN = {True: (0.0, 0.0), False: (0.0, 0.0)}


def _labelled_features(port: Port) -> tuple[pl.DataFrame, pl.DataFrame]:
    feats, voyages = load_features(port)
    points = pl.read_parquet(PROCESSED_DIR / f"eta_points_{port.name}.parquet")
    return add_targets(feats, waiting_episodes(points)), voyages


def train_production(
    port: Port,
    out: Path = artifact.DEFAULT_DIR,
    version: str = "v2",
    train_end: dt.datetime | None = None,
) -> dict:
    """train_end: use only arrivals before it (e.g. to keep the demo's voyages unseen)."""
    feats, voyages = _labelled_features(port)
    split = make_split(feats, voyages, test_start=FAR_FUTURE, train_end=train_end)
    model = DecomposedETA.fit(split.train, split.valid, BASE_FEATURES)
    model.calibrate(split.valid)
    arrivals = split.train["arrival_ts"].to_list() + split.valid["arrival_ts"].to_list()
    info = {
        "version": version,
        "port": port.name,
        "trained_on": {
            "first_arrival": min(arrivals).isoformat(),
            "last_arrival": max(arrivals).isoformat(),
            "voyages": split.train["voyage_id"].n_unique() + split.valid["voyage_id"].n_unique(),
            "ships": split.train["mmsi"].n_unique() + split.valid["mmsi"].n_unique(),
            "rows": split.train.height + split.valid.height,
        },
    }
    artifact.save(model, out, info)
    return info


def recalibrate(
    port: Port, path: Path = artifact.DEFAULT_DIR, days: int = 30, end: dt.datetime | None = None
) -> dict:
    """Re-estimate interval shifts on arrivals in the `days` days before `end`.

    `end` defaults to the newest arrival in the processed data (the data is historical);
    in a live system it would be now.
    """
    model, _ = artifact.load(path)
    feats, voyages = _labelled_features(port)
    end = end or voyages["arrival_ts"].max()
    start = end - dt.timedelta(days=days)
    recent = voyages.filter(pl.col("arrival_ts").is_between(start, end, closed="left"))
    recent = recent.select("voyage_id")
    rows = feats.join(recent, on="voyage_id", how="semi")
    old = dict(model.widen)
    model.widen = NO_WIDEN
    widen = conformal_widen(model.predict(rows), fallback=old)
    info = {
        "window": [start.isoformat(), end.isoformat()],
        "rows": rows.height,
        "voyages": recent.height,
        "previous": {"waiting": list(old[True]), "moving": list(old[False])},
    }
    artifact.save_widen(path, widen, info)
    return {**info, "widen": {"waiting": list(widen[True]), "moving": list(widen[False])}}
