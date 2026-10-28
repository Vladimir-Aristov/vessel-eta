"""Point-level features for ETA prediction.

Every feature at time t uses only that ship's points up to t, plus port-wide
counts from the *previous* complete 30-minute bucket. `tests/test_features.py`
checks this by truncating tracks at t and recomputing.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from vessel_eta.geo import KM_PER_NM, Port

STATE_BUCKET = "30m"
WAITING_ZONE_KM = 40  # anchorages and the approach lanes
INBOUND_ZONE_KM = 50
NAIVE_CAP_MIN = 30 * 24 * 60

CATEGORICAL = ["vessel_type", "status"]
FEATURES = [
    # ship
    "vessel_type", "length", "width", "draft",
    # where
    "lat", "lon", "gate_km", "angels_km", "queens_km", "align",
    # motion now
    "sog", "cog_sin", "cog_cos", "status", "stationary",
    # recent history
    "sog_mean_30m", "sog_mean_2h", "stationary_share_2h", "stationary_share_6h",
    "closing_kmh_1h", "min_since_start", "stationary_min_total", "min_since_moving",
    "from_port",
    # simple physics
    "naive_eta_min", "naive_eta_2h_min",
    # when
    "hour_utc", "dow",
    # port state, previous 30 min
    "port_waiting", "port_inbound", "port_in_port",
]  # fmt: skip


def port_state(tracks: pl.DataFrame, port: Port) -> pl.DataFrame:
    """Port-wide counts per 30-min bucket from all cleaned cargo/tanker tracks.

    Uses the raw polygon test (no debouncing), because debouncing looks ahead.
    """
    lon, lat = tracks["lon"].to_numpy(), tracks["lat"].to_numpy()
    t = tracks.select("mmsi", "ts", "sog").with_columns(
        inside=pl.Series(port.contains(lon, lat)),
        gate_km=pl.Series(port.gate_distances_km(lon, lat).min(axis=1)),
        bucket=pl.col("ts").dt.truncate(STATE_BUCKET),
    )
    outside = ~pl.col("inside")
    return (
        t.group_by("bucket")
        .agg(
            pl.col("mmsi")
            .filter(outside & (pl.col("gate_km") < WAITING_ZONE_KM) & (pl.col("sog") < 0.5))
            .n_unique()
            .alias("port_waiting"),
            pl.col("mmsi")
            .filter(outside & (pl.col("gate_km") < INBOUND_ZONE_KM) & (pl.col("sog") > 3))
            .n_unique()
            .alias("port_inbound"),
            pl.col("mmsi").filter(pl.col("inside")).n_unique().alias("port_in_port"),
        )
        .sort("bucket")
    )


def _bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(a, dtype=np.float64)) for a in (lat1, lon1, lat2, lon2)
    )
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def build_features(
    points: pl.DataFrame,
    voyages: pl.DataFrame,
    state: pl.DataFrame,
    port: Port,
    every_min: int | None = 5,
) -> pl.DataFrame:
    """Labelled approach points -> feature table (keeps voyage_id, mmsi, ts, eta_min).

    Features are computed on every point, then (optionally) one row per
    `every_min` minutes per voyage is kept: neighbouring minutes are near-duplicates.
    """
    df = points.sort("voyage_id", "ts")
    lon, lat = df["lon"].to_numpy(), df["lat"].to_numpy()
    gd = port.gate_distances_km(lon, lat)
    gates = list(port.gates.values())
    nearest = gd.argmin(axis=1)
    g_lon = np.array([gates[i][0] for i in nearest])
    g_lat = np.array([gates[i][1] for i in nearest])
    bearing = _bearing_deg(lat, lon, g_lat, g_lon)
    names = list(port.gates)

    by_voyage = "voyage_id"
    stationary = pl.col("sog") < 0.5
    cog_rad = pl.col("cog").cast(pl.Float64).radians()
    df = (
        df.with_columns(
            angels_km=pl.Series(gd[:, names.index("angels_gate")], dtype=pl.Float32),
            queens_km=pl.Series(gd[:, names.index("queens_gate")], dtype=pl.Float32),
            _bearing=pl.Series(bearing),
            stationary=stationary,
            cog_sin=cog_rad.sin(),
            cog_cos=cog_rad.cos(),
        )
        .with_columns(
            align=((pl.col("cog").cast(pl.Float64) - pl.col("_bearing")).radians()).cos(),
            sog_mean_30m=pl.col("sog").rolling_mean_by("ts", "30m").over(by_voyage),
            sog_mean_2h=pl.col("sog").rolling_mean_by("ts", "2h").over(by_voyage),
            stationary_share_2h=stationary.cast(pl.Float32)
            .rolling_mean_by("ts", "2h")
            .over(by_voyage),
            stationary_share_6h=stationary.cast(pl.Float32)
            .rolling_mean_by("ts", "6h")
            .over(by_voyage),
            min_since_start=(pl.col("ts") - pl.col("ts").first().over(by_voyage)).dt.total_seconds()
            / 60,
            # minutes stationary so far: a gap counts if the point before it was stationary
            stationary_min_total=(
                (pl.col("ts").diff().dt.total_seconds().fill_null(0) / 60)
                * stationary.shift().fill_null(False).cast(pl.Float64)
            )
            .cum_sum()
            .over(by_voyage),
            _last_moving=pl.when(pl.col("sog") > 2)
            .then(pl.col("ts"))
            .forward_fill()
            .over(by_voyage),
            hour_utc=pl.col("ts").dt.hour(),
            dow=pl.col("ts").dt.weekday(),
        )
        .with_columns(
            min_since_moving=(pl.col("ts") - pl.col("_last_moving")).dt.total_seconds() / 60,
        )
    )

    # distance to the gate one hour ago -> closing speed
    lag = df.select(
        by_voyage, pl.col("ts").alias("ts_then"), pl.col("gate_km").alias("gate_km_then")
    ).sort("ts_then")
    df = (
        df.with_columns(_t_lag=pl.col("ts") - pl.duration(hours=1))
        .sort("_t_lag")
        .join_asof(
            lag,
            left_on="_t_lag",
            right_on="ts_then",
            by=by_voyage,
            strategy="backward",
            check_sortedness=False,  # both sides sorted on the time key just above
        )
        .with_columns(
            closing_kmh_1h=(pl.col("gate_km_then") - pl.col("gate_km"))
            / ((pl.col("ts") - pl.col("ts_then")).dt.total_seconds() / 3600)
        )
        .sort(by_voyage, "ts")
    )

    kn_to_kmmin = KM_PER_NM / 60
    df = df.with_columns(
        naive_eta_min=(pl.col("gate_km") / (pl.col("sog").clip(lower_bound=1) * kn_to_kmmin)).clip(
            upper_bound=NAIVE_CAP_MIN
        ),
        naive_eta_2h_min=(
            pl.col("gate_km") / (pl.col("sog_mean_2h").clip(lower_bound=1) * kn_to_kmmin)
        ).clip(upper_bound=NAIVE_CAP_MIN),
        _bucket=pl.col("ts").dt.truncate(STATE_BUCKET) - pl.duration(minutes=30),  # previous bucket
    )
    # buckets missing from the state table (NOAA feed outages) stay null, not 0
    df = df.join(state.rename({"bucket": "_bucket"}), on="_bucket", how="left")
    df = df.join(voyages.select("voyage_id", "from_port"), on="voyage_id", how="left")

    if every_min:
        df = df.filter(
            pl.col("ts").dt.truncate(f"{every_min}m").is_first_distinct().over(by_voyage)
        )
    return df.select("voyage_id", "mmsi", "ts", *FEATURES, "eta_min").with_columns(
        pl.col(c).cast(pl.Float32)
        for c in FEATURES
        if c not in CATEGORICAL + ["from_port", "stationary"]
    )
