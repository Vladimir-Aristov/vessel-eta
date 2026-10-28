"""Score the shipped artifact on arrivals it never saw (default: December 2024).

uv run python scripts/evaluate_artifact.py
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from vessel_eta import artifact
from vessel_eta.experiment import _table, interval_metrics
from vessel_eta.geo import load_port
from vessel_eta.modeling import SLICES, baseline_predictions, metrics
from vessel_eta.production import _labelled_features

FROM = dt.datetime(2024, 12, 1, tzinfo=dt.UTC)


def main() -> None:
    port = load_port()
    model, meta = artifact.load(artifact.DEFAULT_DIR)
    feats, voyages = _labelled_features(port)
    arrivals = voyages.select("voyage_id", "arrival_ts")
    df = feats.join(arrivals, on="voyage_id")
    train, test = df.filter(pl.col("arrival_ts") < FROM), df.filter(pl.col("arrival_ts") >= FROM)
    pred = model.predict(baseline_predictions(train, test)).with_columns(
        pred_model=pl.col("eta_p50")
    )

    rows = []
    for name, cond in {"all": pl.lit(True), **SLICES}.items():
        part = pred.filter(cond)
        rows.append({
            "slice": name, "rows": part.height,
            "distance/speed MAE h": metrics(part, "pred_naive")["mae_h"],
            "model MAE h": metrics(part, "pred_model")["mae_h"],
            **{k: v for k, v in interval_metrics(part, "eta_p10", "eta_p50", "eta_p90").items()
               if k in ("coverage_p10_p90", "median_width_h")},
        })  # fmt: skip
    last = meta["trained_on"]["last_arrival"]
    n = test["voyage_id"].n_unique()
    print(f"artifact trained on arrivals up to {last}; test: {n} voyages from {FROM:%Y-%m-%d}")
    print(_table(pl.DataFrame(rows)))


if __name__ == "__main__":
    main()
