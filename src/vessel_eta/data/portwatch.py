"""IMF PortWatch daily port activity (port calls and estimated trade volumes).

PortWatch estimates daily port calls and import/export volumes per port from
satellite AIS. For San Pedro Bay it covers both Los Angeles and Long Beach as one
port, `port664`. Source: https://portwatch.imf.org (IMF / UN Global Platform).
The raw data is downloaded by the code and not redistributed in this repository.

PortWatch is published in weekly batches, so a forecast made on day d could not
have seen day d-1. Features therefore use 7-day sums ending `LAG_DAYS` before the
prediction day (an assumption about the publication delay, deliberately
conservative).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import httpx
import polars as pl

from vessel_eta.config import DATA_DIR
from vessel_eta.scalars import as_date

URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services/"
    "Daily_Ports_Data/FeatureServer/0/query"
)
LA_LB_PORT_ID = "port664"
EXTERNAL_DIR = DATA_DIR / "external"
LAG_DAYS = 8
COLUMNS = {
    "portcalls_container": "pw_container_calls_7d",
    "portcalls_tanker": "pw_tanker_calls_7d",
    "import": "pw_import_7d",
    "export": "pw_export_7d",
}


def path_for(port_id: str = LA_LB_PORT_ID) -> Path:
    return EXTERNAL_DIR / f"portwatch_{port_id}.json"


def fetch(start: dt.date, end: dt.date, port_id: str = LA_LB_PORT_ID) -> Path:
    """Download daily rows for [start, end] (inclusive) to data/external/."""
    params = {
        "where": f"portid='{port_id}' AND date >= DATE '{start}' AND date <= DATE '{end}'",
        "outFields": "*",
        "returnGeometry": "false",
        "orderByFields": "date",
        "f": "json",
    }
    r = httpx.get(URL, params=params, timeout=120)
    r.raise_for_status()
    payload = r.json()
    if "error" in payload:
        raise RuntimeError(f"PortWatch error: {payload['error']}")
    if payload.get("exceededTransferLimit"):
        raise RuntimeError("PortWatch result truncated; fetch a shorter range")
    out = path_for(port_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    return out


def load(path: Path | None = None) -> pl.DataFrame | None:
    """Daily table (day, portcalls_*, import, export), or None if not downloaded."""
    path = path or path_for()
    if not path.exists():
        return None
    rows = [f["attributes"] for f in json.loads(path.read_text())["features"]]
    return (
        pl.DataFrame(rows)
        .with_columns(day=pl.col("date").str.slice(0, 10).str.to_date())
        .select("day", *COLUMNS)
        .sort("day")
    )


def lagged_features(daily: pl.DataFrame, lag_days: int = LAG_DAYS) -> pl.DataFrame:
    """day -> 7-day sums over [day - lag - 6, day - lag]."""
    full = pl.DataFrame(
        {
            "day": pl.date_range(
                as_date(daily["day"].min()), as_date(daily["day"].max()), "1d", eager=True
            )
        }
    ).join(daily, on="day", how="left")  # missing days stay null, never 0
    sums = full.select(
        "day", *[pl.col(c).rolling_sum(7, min_samples=7).alias(name) for c, name in COLUMNS.items()]
    )
    return sums.with_columns(pl.col("day") + pl.duration(days=lag_days))
