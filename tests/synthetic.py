"""Tiny synthetic ports and ship tracks for tests."""

from __future__ import annotations

import datetime as dt

import polars as pl
import shapely

from vessel_eta.geo import Port, TrackParams, VoyageParams

T0 = dt.datetime(2023, 1, 1, tzinfo=dt.UTC)
GATE = (-118.20, 33.70)  # lon, lat: middle of the port's southern edge
DEG_LAT_PER_MIN_AT_10KN = 10 * 1.852 / 60 / 111.2  # ~0.00278

PORT = Port(
    name="test",
    polygon=shapely.box(-118.30, 33.70, -118.10, 33.80),
    gates={"gate": GATE},
    vessel_types=(70, 89),
    tracks=TrackParams(),
    voyages=VoyageParams(min_port_stay_min=10, max_gap_h=6, max_entry_gap_min=15),
)


def track(mmsi: int, rows: list[tuple[float, float, float, float]]) -> pl.DataFrame:
    """rows: (minute offset, lat, lon, sog)."""
    return pl.DataFrame(
        {
            "mmsi": [mmsi] * len(rows),
            "ts": [T0 + dt.timedelta(minutes=m) for m, *_ in rows],
            "lat": [r[1] for r in rows],
            "lon": [r[2] for r in rows],
            "sog": [r[3] for r in rows],
        },
        schema_overrides={"ts": pl.Datetime("ms", "UTC"), "sog": pl.Float32, "mmsi": pl.Int32},
    ).with_columns(
        cog=pl.lit(0.0, pl.Float32),
        heading=pl.lit(None, pl.Int16),
        status=pl.lit(0, pl.Int16),
        vessel_type=pl.lit(70, pl.Int16),
        vessel_name=pl.lit("TEST"),
        length=pl.lit(200.0, pl.Float32),
        width=pl.lit(30.0, pl.Float32),
        draft=pl.lit(10.0, pl.Float32),
    )


def approach(start_min: float, minutes: int, start_lat: float, lon: float = GATE[0]):
    """Rows of a ship steaming north at 10 kn, one per minute."""
    return [
        (start_min + i, start_lat + i * DEG_LAT_PER_MIN_AT_10KN, lon, 10.0) for i in range(minutes)
    ]


def stay(start_min: float, minutes: int, lat: float, lon: float = GATE[0]):
    return [(start_min + i, lat, lon, 0.0) for i in range(minutes)]
