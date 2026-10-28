"""Find port arrivals and label every approach point with time-to-arrival.

Per ship (cleaned track, sorted by time):

1. Mark points inside the port polygon. Inside-runs shorter than
   `min_port_stay_min` are boundary jitter and count as outside.
2. Each outside-run followed by an inside-run ends in an **arrival** at the
   first inside point. Outside-runs with no inside-run after them (passing
   traffic, or the data window ended) get no labels.
3. An outside-run is split at silences longer than `max_gap_h`; only the part
   right before the arrival is labelled. Otherwise a ship that passed by on day 3
   and called on day 25 would get a 22-day "ETA".
4. If the last outside point and the first inside point are more than
   `max_entry_gap_min` apart, the arrival time is unsure and the voyage is
   kept in the voyage table but not labelled.

Label: `eta_min` = minutes from the point to the arrival.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from vessel_eta.geo import Port
from vessel_eta.scalars import as_float

POINT_COLS = [
    "mmsi", "ts", "lat", "lon", "sog", "cog", "heading", "status",
    "vessel_type", "length", "width", "draft",
]  # fmt: skip


def _runs(flag: str) -> pl.Expr:
    """Run id of consecutive equal values of `flag` within each ship."""
    return pl.col(flag).rle_id().over("mmsi")


def mark_port(df: pl.DataFrame, port: Port) -> pl.DataFrame:
    """Add `in_port` (debounced) and `gate_km` (distance to nearest gate)."""
    raw_in = port.contains(df["lon"].to_numpy(), df["lat"].to_numpy())
    gate_km = port.gate_distances_km(df["lon"].to_numpy(), df["lat"].to_numpy()).min(axis=1)
    df = df.with_columns(_in=pl.Series(raw_in), gate_km=pl.Series(gate_km, dtype=pl.Float32))
    df = df.with_columns(_run=_runs("_in"))
    min_stay = pl.duration(minutes=port.voyages.min_port_stay_min)
    run_len = (pl.col("ts").max() - pl.col("ts").min()).over("mmsi", "_run")
    return df.with_columns(in_port=pl.col("_in") & (run_len >= min_stay)).drop("_in", "_run")


def build_voyages(df: pl.DataFrame, port: Port) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Clean tracks -> (voyages, labelled approach points)."""
    p = port.voyages
    df = mark_port(df.sort("mmsi", "ts"), port).with_columns(run=_runs("in_port"))

    runs = (
        df.group_by("mmsi", "run", maintain_order=True)
        .agg(
            pl.first("in_port"),
            pl.col("ts").first().alias("t_first"),
            pl.col("ts").last().alias("t_last"),
            pl.col("lon").first().alias("lon_first"),
            pl.col("lat").first().alias("lat_first"),
        )
        .sort("mmsi", "run")
    )
    outside = runs.filter(~pl.col("in_port")).join(
        runs.filter(pl.col("in_port")).select(
            "mmsi",
            (pl.col("run") - 1).alias("run"),
            pl.col("t_first").alias("arrival_ts"),
            pl.col("lon_first").alias("arr_lon"),
            pl.col("lat_first").alias("arr_lat"),
        ),
        on=["mmsi", "run"],
        how="inner",
    )
    outside = outside.with_columns(
        left_port=pl.col("run") > 0,  # the outside-run starts with a departure
        entry_gap_min=(pl.col("arrival_ts") - pl.col("t_last")).dt.total_seconds() / 60,
    )
    gate_names = list(port.gates)
    nearest = port.gate_distances_km(outside["arr_lon"], outside["arr_lat"]).argmin(axis=1)
    outside = outside.with_columns(
        entry_gate=pl.Series([gate_names[i] for i in nearest], dtype=pl.String),
        voyage_id=pl.format(
            "{}-{}", pl.col("mmsi"), pl.col("arrival_ts").dt.strftime("%Y%m%dT%H%M")
        ),
        labelled=pl.col("entry_gap_min") <= p.max_entry_gap_min,
    )

    # approach points: outside-run rows of voyages, last segment after any long silence
    pts = df.join(
        outside.select("mmsi", "run", "voyage_id", "arrival_ts", "labelled"),
        on=["mmsi", "run"],
        how="inner",
    ).sort("mmsi", "ts")
    gap = pl.col("ts").diff().over("voyage_id") > pl.duration(hours=p.max_gap_h)
    pts = pts.with_columns(seg=gap.fill_null(False).cum_sum().over("voyage_id"))
    pts = pts.filter(pl.col("seg") == pl.col("seg").max().over("voyage_id"))

    stats = pts.group_by("voyage_id").agg(
        pl.col("ts").min().alias("start_ts"),
        pl.len().alias("n_points"),
        (pl.col("ts").diff().max().dt.total_seconds() / 60).alias("max_gap_min"),
        pl.col("gate_km").first().alias("start_gate_km"),
        (pl.col("sog") < 0.5).mean().alias("stationary_share"),
        pl.col("draft").drop_nulls().mode().first().alias("draft"),
    )
    voyages = (
        outside.join(stats, on="voyage_id", how="inner")
        # The approach starts at a departure from this same port (berth shift, trip to an
        # anchorage) only if the labelled part was not cut at a silence. A ship that left,
        # went out of range and came back days later is arriving from sea.
        .with_columns(from_port=pl.col("left_port") & (pl.col("start_ts") == pl.col("t_first")))
        .join(
            df.group_by("mmsi").agg(
                pl.col("vessel_type").drop_nulls().mode().first(),
                pl.col("length").drop_nulls().mode().first(),
                pl.col("vessel_name").drop_nulls().mode().first(),
            ),
            on="mmsi",
            how="left",
        )
        .with_columns(
            duration_h=(pl.col("arrival_ts") - pl.col("start_ts")).dt.total_seconds() / 3600
        )
        .select(
            "voyage_id",
            "mmsi",
            "vessel_name",
            "vessel_type",
            "length",
            "draft",
            "start_ts",
            "arrival_ts",
            "duration_h",
            "n_points",
            "max_gap_min",
            "start_gate_km",
            "entry_gate",
            "entry_gap_min",
            "from_port",
            "stationary_share",
            "labelled",
        )  # fmt: skip
        .sort("arrival_ts", "mmsi")
    )

    labels = (
        pts.filter(pl.col("labelled"))
        .with_columns(
            eta_min=((pl.col("arrival_ts") - pl.col("ts")).dt.total_seconds() / 60).cast(pl.Float32)
        )
        .select("voyage_id", *POINT_COLS, "gate_km", "eta_min")
        .sort("voyage_id", "ts")
    )
    if labels.height and as_float(labels["eta_min"].min()) <= 0:
        raise AssertionError("non-positive ETA")
    return voyages, labels


def summarize(voyages: pl.DataFrame, labels: pl.DataFrame) -> dict[str, float]:
    lab = voyages.filter(pl.col("labelled"))
    q = np.quantile(lab["duration_h"].to_numpy(), [0.5, 0.9]) if lab.height else [np.nan] * 2
    return {
        "arrivals": voyages.height,
        "arrivals_labelled": lab.height,
        "arrivals_unsure_time": voyages.height - lab.height,
        "arrivals_from_sea": int((~lab["from_port"]).sum()),
        "ships": lab["mmsi"].n_unique(),
        "labelled_points": labels.height,
        "approach_h_median": float(q[0]),
        "approach_h_p90": float(q[1]),
    }
