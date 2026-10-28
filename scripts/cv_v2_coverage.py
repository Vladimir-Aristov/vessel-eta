"""Is v2's interval under-coverage on March a model problem or a shift?

Grouped-by-ship 5-fold CV inside Jan-Feb (no March data): if coverage is ~0.80
in-period, the March gap is distribution shift (more waiting in March).

uv run python scripts/cv_v2_coverage.py
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from vessel_eta.config import PROCESSED_DIR, ROOT
from vessel_eta.decompose import DecomposedETA, add_targets, waiting_episodes
from vessel_eta.experiment import _table, interval_metrics, load_features
from vessel_eta.geo import load_port
from vessel_eta.modeling import SLICES, metrics

TEST_START = dt.datetime(2023, 3, 1, tzinfo=dt.UTC)


def main() -> None:
    port = load_port()
    feats, voyages = load_features(port)
    points = pl.read_parquet(PROCESSED_DIR / f"eta_points_{port.name}.parquet")
    feats = add_targets(feats, waiting_episodes(points))
    df = feats.join(voyages.select("voyage_id", "arrival_ts"), on="voyage_id").filter(
        pl.col("arrival_ts") < TEST_START
    )
    ships = np.sort(df["mmsi"].unique().to_numpy())
    rng = np.random.default_rng(0)
    fold_of = dict(zip(rng.permutation(ships).tolist(), np.arange(len(ships)) % 5, strict=True))
    df = df.with_columns(fold=pl.col("mmsi").replace_strict(fold_of, return_dtype=pl.Int64))
    parts = []
    for k in range(5):
        rest, held = df.filter(pl.col("fold") != k), df.filter(pl.col("fold") == k)
        rest_ships = np.sort(rest["mmsi"].unique().to_numpy())
        es = rng.choice(rest_ships, size=len(rest_ships) // 5, replace=False).tolist()
        fit_part, es_part = (
            rest.filter(~pl.col("mmsi").is_in(es)),
            rest.filter(pl.col("mmsi").is_in(es)),
        )
        model = DecomposedETA.fit(fit_part, es_part)
        parts.append(model.predict(held).with_columns(cqr=pl.lit(False)))
        parts.append(model.calibrate(es_part).predict(held).with_columns(cqr=pl.lit(True)))
        print(f"fold {k} done", flush=True)
    rows = []
    for cqr in (False, True):
        oof = pl.concat(parts).filter(pl.col("cqr") == cqr)
        for name, cond in {"all": pl.lit(True), **SLICES}.items():
            part = oof.filter(cond)
            rows.append({"intervals": "conformalised" if cqr else "raw quantiles", "slice": name,
                         **interval_metrics(part, "eta_p10", "eta_p50", "eta_p90"),
                         "mae_h": metrics(part, "eta_p50")["mae_h"]})  # fmt: skip
    table = _table(pl.DataFrame(rows))
    out = ROOT / "reports" / "cv_v2_coverage.md"
    out.write_text(
        "# v2 intervals, grouped-by-ship 5-fold CV inside Jan-Feb 2023\n\n" + table + "\n"
    )
    print(table)


if __name__ == "__main__":
    main()
