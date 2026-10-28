"""Build the static map demo (GitHub Pages): the same page, with forecasts precomputed.

Every slider position of every demo voyage is sent through `Predictor.predict` — the
code behind `POST /v1/eta` — and stored compactly, so the page needs no server.

uv run python scripts/build_static_demo.py [--out site]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import polars as pl

from vessel_eta.serving.predictor import AlreadyInPort, Predictor, Static

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "vessel_eta" / "serving" / "static"
API_CONFIG = 'const CONFIG = { mode: "api", samples: "/static/samples.json", forecasts: null };'
SITE_CONFIG = (
    'const CONFIG = { mode: "static", samples: "samples.json", forecasts: "forecasts.json" };'
)


def compact(res: dict) -> list:
    q = res["eta_minutes"]
    p_wait = res["p_wait_before_arrival"]
    return [
        round(q["p10"], 1), round(q["p50"], 1), round(q["p90"], 1),
        None if p_wait is None or math.isnan(p_wait) else round(p_wait, 3),
        int(res["waiting_now"]), round(res["distance_to_gate_km"], 1),
    ]  # fmt: skip


def main(out: Path) -> None:
    predictor = Predictor.load()
    samples = json.loads((STATIC / "samples.json").read_text())
    forecasts, n, t0 = [], 0, time.perf_counter()
    for s in samples["samples"]:
        points = pl.DataFrame(s["track"]).with_columns(
            pl.col("ts").str.to_datetime(time_zone="UTC"),
            pl.col("sog", "cog").cast(pl.Float32),
            pl.col("heading", "status").cast(pl.Int16),
        )
        static = Static(**s["ship"])
        per_point = []
        for i in range(1, points.height + 1):
            try:
                per_point.append(compact(predictor.predict(points.head(i), static)))
            except AlreadyInPort:
                per_point.append(None)
            n += 1
        forecasts.append(per_point)
    out.mkdir(parents=True, exist_ok=True)
    html = (STATIC / "index.html").read_text()
    if API_CONFIG not in html:
        raise SystemExit("index.html CONFIG line changed; update build_static_demo.py")
    (out / "index.html").write_text(html.replace(API_CONFIG, SITE_CONFIG))
    shutil.copy2(STATIC / "samples.json", out / "samples.json")
    (out / "forecasts.json").write_text(json.dumps(forecasts, separators=(",", ":")))
    (out / ".nojekyll").write_text("")
    took = time.perf_counter() - t0
    print(f"{n} forecasts for {len(forecasts)} voyages in {took:.0f} s -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "site")
    main(ap.parse_args().out)
