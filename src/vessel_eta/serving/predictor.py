"""Online inference: one ship's recent AIS track -> ETA distribution.

The request is turned into exactly the table the training pipeline builds, and
features come from the same `build_features` call, so there is no separate serving
implementation that could drift (`tests/test_serving.py` checks the two agree).

What the client must send: the ship's current approach, oldest point first — ideally
everything since it entered the region or since its last silence of more than 6 h.
Features such as "minutes stationary so far" and "minutes since the approach started"
are measured from the first point sent, as they were in training; a 2-hour snippet of
a ship that has been at anchor for 3 days looks like a ship that just stopped.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from vessel_eta import artifact
from vessel_eta.decompose import WAITING_NOW_KN, DecomposedETA
from vessel_eta.features import build_features
from vessel_eta.geo import DEFAULT_PORT_CONFIG, Port, load_port
from vessel_eta.tracks import clean_tracks

VOYAGE = "request"
STATE_SCHEMA = {
    "bucket": pl.Datetime("ms", "UTC"),
    "port_waiting": pl.Int64,
    "port_inbound": pl.Int64,
    "port_in_port": pl.Int64,
}


class AlreadyInPort(ValueError):
    """The last position is inside the port: the ship has arrived."""


@dataclass(frozen=True)
class Static:
    mmsi: int
    vessel_type: int
    length: float | None = None
    width: float | None = None
    draft: float | None = None


@dataclass(frozen=True)
class PortState:
    """Port-wide counts over the last 30 minutes, if the caller tracks them."""

    waiting: int
    inbound: int
    in_port: int


def request_table(
    points: pl.DataFrame, static: Static, port: Port
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Raw points (ts, lat, lon, sog, cog, heading, status) -> the labelled-points layout.

    Cleans the track the way training did, refuses ships already in port, and keeps
    only the part after the last silence longer than `max_gap_h`.
    """
    df = (
        points.with_columns(
            pl.col("ts").cast(pl.Datetime("ms", "UTC")),
            pl.lit(static.mmsi, pl.Int32).alias("mmsi"),
            pl.lit(static.vessel_type, pl.Int16).alias("vessel_type"),
            pl.lit(static.length, pl.Float32).alias("length"),
            pl.lit(static.width, pl.Float32).alias("width"),
            pl.lit(static.draft, pl.Float32).alias("draft"),
        )
        .unique("ts", keep="first")
        .sort("ts")
    )
    n_in = df.height
    if df.height >= 2 * port.tracks.min_points:  # too short a track has no "main" sub-track
        df, _ = clean_tracks(df, port.tracks)
    if port.contains(df["lon"][-1:].to_numpy(), df["lat"][-1:].to_numpy())[0]:
        raise AlreadyInPort("the last position is inside the port: the ship has arrived")
    gap = pl.col("ts").diff() > pl.duration(hours=port.voyages.max_gap_h)
    df = df.filter(gap.fill_null(False).cum_sum() == gap.fill_null(False).cum_sum().max())
    gate_km = port.gate_distances_km(df["lon"].to_numpy(), df["lat"].to_numpy()).min(axis=1)
    df = df.with_columns(
        voyage_id=pl.lit(VOYAGE),
        gate_km=pl.Series(gate_km, dtype=pl.Float32),
        eta_min=pl.lit(None, pl.Float32),
    )
    return df, {"points_received": n_in, "points_used": df.height}


def request_features(
    table: pl.DataFrame,
    port: Port,
    from_port: bool = False,
    state: PortState | None = None,
) -> pl.DataFrame:
    """Feature row for the last point, via the training code path."""
    voyages = pl.DataFrame({"voyage_id": [VOYAGE], "from_port": [from_port]})
    if state is None:
        state_df = pl.DataFrame(schema=STATE_SCHEMA)
    else:
        # build_features joins the *previous* 30-min bucket, so file the counts there
        bucket = table["ts"].max().replace(second=0, microsecond=0)
        bucket = bucket.replace(minute=bucket.minute // 30 * 30) - dt.timedelta(minutes=30)
        state_df = pl.DataFrame(
            {"bucket": [bucket], "port_waiting": [state.waiting],
             "port_inbound": [state.inbound], "port_in_port": [state.in_port]},
            schema=STATE_SCHEMA,
        )  # fmt: skip
    feats = build_features(table, voyages, state_df, port, every_min=None)
    return feats.tail(1).with_columns(waiting_now=pl.col("sog_mean_30m") < WAITING_NOW_KN)


class Predictor:
    def __init__(self, model: DecomposedETA, meta: dict, port: Port):
        self.model, self.meta, self.port = model, meta, port

    @classmethod
    def load(cls, path: Path = artifact.DEFAULT_DIR, port_config: Path = DEFAULT_PORT_CONFIG):
        model, meta = artifact.load(path)
        return cls(model, meta, load_port(port_config))

    def predict(
        self,
        points: pl.DataFrame,
        static: Static,
        from_port: bool = False,
        state: PortState | None = None,
    ) -> dict:
        table, counts = request_table(points, static, self.port)
        row = request_features(table, self.port, from_port, state)
        pred = self.model.predict(row).row(0, named=True)
        last = table["ts"][-1]
        p_wait = pred["p_stop"]
        minutes = {k: float(pred[f"eta_{k}"]) for k in ("p10", "p50", "p90")}
        return {
            "as_of": last,
            "waiting_now": bool(pred["waiting_now"]),
            "p_wait_before_arrival": None if np.isnan(p_wait) else float(p_wait),
            "eta_minutes": minutes,
            "eta": {k: last + dt.timedelta(minutes=v) for k, v in minutes.items()},
            "distance_to_gate_km": float(row["gate_km"][0]),
            **counts,
            "model_version": self.meta.get("version"),
        }  # fmt: skip
