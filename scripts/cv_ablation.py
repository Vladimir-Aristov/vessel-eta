"""Grouped-by-ship 5-fold CV on the training period (Jan-Feb): how the model
configuration was chosen. The test month is not touched here.

uv run python scripts/cv_ablation.py
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from vessel_eta.config import ROOT
from vessel_eta.experiment import _table, load_features
from vessel_eta.features import BASE_FEATURES as FEATURES  # the v1 study
from vessel_eta.geo import load_port
from vessel_eta.modeling import SLICES, CalibratedNaive, grouped_cv

TEST_START = dt.datetime(2023, 3, 1, tzinfo=dt.UTC)
SHIP = ["vessel_type", "length", "width", "draft"]
WIDE = {"num_leaves": 63, "min_data_in_leaf": 200, "lambda_l2": 1.0}  # first, unregularised try

CONFIGS = {
    "wide trees": (FEATURES, WIDE),
    "wide, no ship size/type": ([f for f in FEATURES if f not in SHIP], WIDE),
    "regularised (chosen)": (FEATURES, {}),
    "regularised, no ship size/type": ([f for f in FEATURES if f not in SHIP], {}),
}


def score(df: pl.DataFrame, pred: np.ndarray) -> dict[str, float]:
    e = df.select("voyage_id", "sog", "gate_km", "eta_min").with_columns(
        ae=(pl.Series(pred) - pl.col("eta_min")).abs() / 60
    )
    row = {
        "mae_h": e["ae"].mean(),
        "voyage_mae_h": e.group_by("voyage_id").agg(pl.col("ae").mean())["ae"].mean(),
    }
    for name, cond in SLICES.items():
        row[name] = e.filter(cond)["ae"].mean()
    return row


def main() -> None:
    feats, voyages = load_features(load_port())
    train = feats.join(voyages.select("voyage_id", "arrival_ts"), on="voyage_id").filter(
        pl.col("arrival_ts") < TEST_START
    )
    # 2 points have no SOG: fall back to the 2-hour mean speed
    naive = train["naive_eta_min"].fill_null(train["naive_eta_2h_min"]).to_numpy()
    rows = [{"config": "naive distance / speed", **score(train, naive)}]
    # calibrated baseline, also out-of-fold by ship
    cal = np.zeros(train.height)
    ships = np.sort(train["mmsi"].unique().to_numpy())
    fold_of = dict(
        zip(
            np.random.default_rng(0).permutation(ships).tolist(),
            np.arange(len(ships)) % 5,
            strict=True,
        )
    )
    fold = train["mmsi"].replace_strict(fold_of, return_dtype=pl.Int64).to_numpy()
    for k in range(5):
        m = fold == k
        cal[m] = (
            CalibratedNaive().fit(train.filter(pl.Series(~m))).predict(train.filter(pl.Series(m)))
        )
    rows.append({"config": "calibrated naive", **score(train, cal)})
    for name, (features, params) in CONFIGS.items():
        rows.append({"config": name, **score(train, grouped_cv(train, features, params))})
        print(rows[-1], flush=True)
    table = _table(pl.DataFrame(rows))
    out = ROOT / "reports" / "cv_ablation.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("# Grouped-by-ship 5-fold CV, Jan-Feb 2023 (MAE, hours)\n\n" + table + "\n")
    print(table)


if __name__ == "__main__":
    main()
