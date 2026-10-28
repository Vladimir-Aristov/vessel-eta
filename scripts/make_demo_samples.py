"""Pick real December 2024 approaches for the map demo (unseen by the shipped model).

The shipped artifact is trained on arrivals before 2024-12-01, so these voyages are a
genuine forecast. Writes src/vessel_eta/serving/static/samples.json.

uv run python scripts/make_demo_samples.py
"""

from __future__ import annotations

import datetime as dt
import json

import polars as pl

from vessel_eta.config import PROCESSED_DIR, ROOT
from vessel_eta.geo import load_port

FROM = dt.datetime(2024, 12, 1, tzinfo=dt.UTC)
OUT = ROOT / "src" / "vessel_eta" / "serving" / "static" / "samples.json"
MAX_POINTS = 1500  # 5-minute grid: ~5 days of track
PER_KIND = 3


def main() -> None:
    port = load_port()
    voyages = pl.read_parquet(PROCESSED_DIR / f"voyages_{port.name}.parquet").filter(
        pl.col("labelled") & (pl.col("start_ts") >= FROM) & ~pl.col("from_port")
    )
    kinds = {
        "goes straight in": voyages.filter(
            (pl.col("stationary_share") == 0) & pl.col("duration_h").is_between(8, 30)
        ),
        "waits first": voyages.filter(
            pl.col("stationary_share").is_between(0.3, 0.9) & (pl.col("duration_h") < 120)
        ),
        "tanker": voyages.filter(pl.col("vessel_type").is_between(80, 89)),
    }
    picked, seen = [], set()
    for kind, df in kinds.items():
        df = df.filter(~pl.col("voyage_id").is_in(list(seen))).sort("voyage_id")
        for row in df.sample(n=min(PER_KIND, df.height), seed=7).iter_rows(named=True):
            seen.add(row["voyage_id"])
            picked.append((kind, row))

    points = pl.read_parquet(PROCESSED_DIR / f"eta_points_{port.name}.parquet").filter(
        pl.col("voyage_id").is_in(list(seen))
    )
    samples = []
    for kind, v in picked:
        p = (
            points.filter(pl.col("voyage_id") == v["voyage_id"])
            .sort("ts")
            .filter(pl.col("ts").dt.truncate("5m").is_first_distinct())
            .tail(MAX_POINTS)
        )
        samples.append({
            "kind": kind,
            "name": v["vessel_name"] or f"MMSI {v['mmsi']}",
            "ship": {"mmsi": v["mmsi"], "vessel_type": v["vessel_type"],
                     "length": v["length"], "draft": v["draft"]},
            "arrival": v["arrival_ts"].isoformat(),
            "track": [
                {"ts": r["ts"].isoformat(), "lat": round(r["lat"], 5), "lon": round(r["lon"], 5),
                 "sog": r["sog"], "cog": r["cog"], "heading": r["heading"], "status": r["status"]}
                for r in p.iter_rows(named=True)
            ],
        })  # fmt: skip
    payload = {
        "port": {"polygon": [[lat, lon] for lon, lat in port.polygon.exterior.coords],
                 "gates": {k: [lat, lon] for k, (lon, lat) in port.gates.items()}},
        "samples": samples,
    }  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"{len(samples)} samples, {sum(len(s['track']) for s in samples)} points -> {OUT}")
    for s in samples:
        print(f"  {s['kind']:18s} {s['name']:28s} {len(s['track']):5d} pts  arrival {s['arrival']}")


if __name__ == "__main__":
    main()
