"""Splits, baselines, metrics and the LightGBM ETA model.

Validation design
-----------------
- **Time**: train on voyages that arrived before `test_start`, test on the rest.
- **Vessel**: early stopping uses a held-out 20% of *ships* from the training period,
  so the model is not tuned on ships it has memorised. The test set is reported
  separately for ships seen in training and ships that never were.
- **Per voyage**: point MAE is dominated by long waits with many points; voyage MAE
  (mean of each voyage's own MAE) weights every arrival equally.
- **Slices you can see at prediction time** (moving near the port, moving far out,
  slow, stationary). Slicing by the *true* horizon is reported as a diagnostic only:
  "true ETA < 2 h" selects ships that went straight in, which flatters the naive
  distance/speed estimate.

Hyper-parameters were chosen by 5-fold grouped-by-ship CV on the training period
(`scripts/cv_ablation.py`); the test month was scored once.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

from vessel_eta.features import CATEGORICAL, FEATURES

SLICES = {
    "moving, < 30 km from gate": (pl.col("sog") >= 5) & (pl.col("gate_km") < 30),
    "moving, >= 30 km": (pl.col("sog") >= 5) & (pl.col("gate_km") >= 30),
    "slow (0.5-5 kn)": (pl.col("sog") >= 0.5) & (pl.col("sog") < 5),
    "stationary (< 0.5 kn)": pl.col("sog") < 0.5,
}
HORIZONS_H = [0, 2, 6, 12, 24, 72, np.inf]


# ------------------------------------------------------------------ splits


@dataclass
class Split:
    train: pl.DataFrame
    valid: pl.DataFrame  # held-out ships from the training period (early stopping)
    test: pl.DataFrame
    test_start: dt.datetime


def make_split(
    feats: pl.DataFrame, voyages: pl.DataFrame, test_start: dt.datetime, valid_frac=0.2, seed=0
) -> Split:
    arrivals = voyages.select("voyage_id", "arrival_ts")
    df = feats.join(arrivals, on="voyage_id", how="inner")
    train_all = df.filter(pl.col("arrival_ts") < test_start)
    test = df.filter(pl.col("arrival_ts") >= test_start)

    ships = np.sort(train_all["mmsi"].unique().to_numpy())
    rng = np.random.default_rng(seed)
    valid_ships = set(rng.choice(ships, size=int(len(ships) * valid_frac), replace=False).tolist())
    in_valid = pl.col("mmsi").is_in(list(valid_ships))
    train_ships = set(train_all["mmsi"].unique().to_list())
    test = test.with_columns(seen_ship=pl.col("mmsi").is_in(list(train_ships)))
    return Split(
        train=train_all.filter(~in_valid),
        valid=train_all.filter(in_valid),
        test=test,
        test_start=test_start,
    )


# ------------------------------------------------------------------ metrics


def metrics(df: pl.DataFrame, pred_col: str) -> dict[str, float]:
    e = df.select(
        "voyage_id",
        ((pl.col(pred_col) - pl.col("eta_min")).abs() / 60).alias("ae_h"),
        ((pl.col(pred_col) - pl.col("eta_min")) / 60).alias("err_h"),
    )
    per_voyage = e.group_by("voyage_id").agg(pl.col("ae_h").mean())
    return {
        "mae_h": e["ae_h"].mean(),
        "median_ae_h": e["ae_h"].median(),
        "voyage_mae_h": per_voyage["ae_h"].mean(),
        "median_bias_h": e["err_h"].median(),  # + = predicts too late
        "n_points": e.height,
        "n_voyages": per_voyage.height,
    }


def breakdown(df: pl.DataFrame, pred_cols: list[str], by: str = "observable") -> pl.DataFrame:
    """MAE (hours) per slice. by="observable": state at prediction time;
    by="true_horizon": diagnostic only (conditions on the outcome)."""
    parts: list[tuple[str, pl.DataFrame]] = []
    if by == "observable":
        parts = [(name, df.filter(cond)) for name, cond in SLICES.items()]
    else:
        h = pl.col("eta_min") / 60
        for lo, hi in zip(HORIZONS_H, HORIZONS_H[1:], strict=False):
            label = f"{lo}-{hi:g} h" if np.isfinite(hi) else f"> {lo} h"
            parts.append((label, df.filter((h >= lo) & (h < hi))))
    rows = []
    for name, part in parts:
        row = {"slice": name, "n": part.height, "median_true_eta_h": part["eta_min"].median() / 60}
        row.update({c: metrics(part, c)["mae_h"] for c in pred_cols})
        rows.append(row)
    return pl.DataFrame(rows)


# ------------------------------------------------------------------ baselines


class CalibratedNaive:
    """log ETA ~ log(distance / speed), fitted separately for moving and stopped ships.

    The straight-line estimate is systematically too early (pilot slow-down, lanes)
    and says nothing useful for a ship at anchor; a 2-parameter fit per state fixes
    the first and gives the typical wait for the second.
    """

    def fit(self, df: pl.DataFrame) -> CalibratedNaive:
        self.models = {}
        for moving in (True, False):
            part = df.filter((pl.col("sog") >= 0.5) == moving)
            x = np.log1p(part["naive_eta_2h_min"].fill_null(0).to_numpy())[:, None]
            self.models[moving] = LinearRegression().fit(x, np.log1p(part["eta_min"].to_numpy()))
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        x = np.log1p(df["naive_eta_2h_min"].fill_null(0).to_numpy())[:, None]
        moving = (df["sog"].fill_null(0) >= 0.5).to_numpy()
        out = np.where(moving, self.models[True].predict(x), self.models[False].predict(x))
        return np.expm1(out)


def baseline_predictions(train: pl.DataFrame, df: pl.DataFrame) -> pl.DataFrame:
    """Add pred_median / pred_naive / pred_calibrated, all fitted on `train` only."""
    cal = CalibratedNaive().fit(train)
    return df.with_columns(
        pred_median=pl.lit(train["eta_min"].median(), pl.Float64),
        pred_naive=pl.col("naive_eta_min").cast(pl.Float64),
        pred_calibrated=pl.Series(cal.predict(df)),
    )


# ------------------------------------------------------------------ model

# Chosen by grouped CV (scripts/cv_ablation.py): rows within a voyage are highly
# correlated, so the effective sample is ~500 voyages, not ~200k rows -> small,
# heavily regularised trees.
LGB_PARAMS = {
    "objective": "regression",  # on log1p(minutes): conditional median-ish, robust to the tail
    "metric": "l1",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 2000,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "verbose": -1,
    "seed": 0,
    "num_threads": 8,
}
EARLY_STOPPING = 150
MAX_ROUNDS = 6000


CAT_IDX = [FEATURES.index(c) for c in CATEGORICAL]


def _x(df: pl.DataFrame, features: list[str]) -> np.ndarray:
    # categorical codes (AIS type, nav status) are small non-negative ints; LightGBM uses them as-is
    return df.select(pl.col(features).cast(pl.Float64)).to_numpy()


def _y(df: pl.DataFrame) -> np.ndarray:
    return np.log1p(df["eta_min"].to_numpy().astype(np.float64))


@dataclass
class Fitted:
    booster: lgb.Booster
    features: list[str]
    best_iteration: int

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        x = _x(df, self.features)
        return np.expm1(self.booster.predict(x, num_iteration=self.best_iteration))


def fit_lgbm(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    features: list[str] = FEATURES,
    params: dict | None = None,
) -> Fitted:
    """Train on `train`, early-stop on `valid` (held-out ships)."""
    params = {**LGB_PARAMS, **(params or {})}
    cat = [features.index(c) for c in CATEGORICAL if c in features]
    dtr = lgb.Dataset(
        _x(train, features), _y(train), feature_name=features, categorical_feature=cat
    )
    dva = lgb.Dataset(_x(valid, features), _y(valid), reference=dtr)
    booster = lgb.train(
        params, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dva],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )  # fmt: skip
    return Fitted(booster, features, booster.best_iteration)


def grouped_cv(
    df: pl.DataFrame,
    features: list[str] = FEATURES,
    params: dict | None = None,
    n_splits: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Out-of-fold predictions (minutes); folds and early-stopping sets split by ship."""
    ships = np.sort(df["mmsi"].unique().to_numpy())
    rng = np.random.default_rng(seed)
    fold_of = dict(
        zip(rng.permutation(ships).tolist(), np.arange(len(ships)) % n_splits, strict=True)
    )
    fold = df["mmsi"].replace_strict(fold_of, return_dtype=pl.Int64).to_numpy()
    oof = np.zeros(df.height)
    for k in range(n_splits):
        test_mask = fold == k
        rest = df.filter(~pl.Series(test_mask))
        rest_ships = np.sort(rest["mmsi"].unique().to_numpy())
        es = rng.choice(rest_ships, size=max(1, len(rest_ships) // 5), replace=False).tolist()
        in_es = pl.col("mmsi").is_in(es)
        model = fit_lgbm(rest.filter(~in_es), rest.filter(in_es), features, params)
        oof[test_mask] = model.predict(df.filter(pl.Series(test_mask)))
    return oof


def importance(fitted: Fitted) -> pl.DataFrame:
    """Share of total split gain per feature."""
    gain = fitted.booster.feature_importance("gain", iteration=fitted.best_iteration)
    return (
        pl.DataFrame({"feature": fitted.booster.feature_name(), "gain": gain})
        .with_columns(share=pl.col("gain") / pl.col("gain").sum())
        .sort("gain", descending=True)
    )
