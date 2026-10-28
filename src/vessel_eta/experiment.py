"""End-to-end ETA experiment: features -> split -> baselines + LightGBM -> report."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl

from vessel_eta.config import PROCESSED_DIR, ROOT
from vessel_eta.data import pola, portwatch
from vessel_eta.features import FEATURE_SETS, build_features
from vessel_eta.geo import Port
from vessel_eta.modeling import (
    SLICES,
    baseline_predictions,
    breakdown,
    fit_lgbm,
    importance,
    make_split,
    metrics,
)

REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
PRED_COLS = ["pred_median", "pred_naive", "pred_calibrated", "pred_lgbm"]


def load_features(port: Port, processed: Path = PROCESSED_DIR) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Feature table (with PortWatch if downloaded) and labelled voyages."""
    voyages = pl.read_parquet(processed / f"voyages_{port.name}.parquet").filter("labelled")
    points = pl.read_parquet(processed / f"eta_points_{port.name}.parquet")
    state = pl.read_parquet(processed / f"port_state_{port.name}.parquet")
    feats = build_features(
        points, voyages, state, port, portwatch=portwatch.load(), pola_daily=pola.load()
    )
    return feats, voyages


def _cell(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 1 else f"{v:,.1f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def _table(df: pl.DataFrame) -> str:
    """Markdown table; floats to 1 decimal (3 for shares below 1)."""
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        cells = [_cell(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run(port: Port, test_start: dt.datetime, name: str = "eta_v1") -> dict:
    feats, voyages = load_features(port)
    split = make_split(feats, voyages, test_start)
    model = fit_lgbm(split.train, split.valid)

    test = baseline_predictions(split.train, split.test).with_columns(
        pred_lgbm=pl.Series(model.predict(split.test))
    )

    def overall(df: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame(
            [{"model": c.removeprefix("pred_"), **metrics(df, c)} for c in PRED_COLS]
        )

    seen = test.filter("seen_ship")
    unseen = test.filter(~pl.col("seen_ship"))
    result = {
        "name": name,
        "test_start": test_start.isoformat(),
        "sizes": {
            k: {
                "rows": d.height,
                "voyages": d["voyage_id"].n_unique(),
                "ships": d["mmsi"].n_unique(),
            }
            for k, d in [("train", split.train), ("valid", split.valid), ("test", test)]
        },
        "best_iteration": model.best_iteration,
        "test": overall(test).to_dicts(),
        "test_seen_ships": overall(seen).to_dicts(),
        "test_unseen_ships": overall(unseen).to_dicts(),
        "slices": breakdown(test, PRED_COLS).to_dicts(),
        "true_horizon_diagnostic": breakdown(test, PRED_COLS, by="true_horizon").to_dicts(),
        "importance": importance(model).head(15).to_dicts(),
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / f"{name}.json").write_text(json.dumps(result, indent=2, default=str))
    model.booster.save_model(MODELS_DIR / f"{name}.txt", num_iteration=model.best_iteration)

    ts = result["sizes"]["test"]
    keep = ["model", "mae_h", "median_ae_h", "voyage_mae_h", "median_bias_h"]
    md = [
        f"# {name}: ETA to LA/Long Beach",
        f"Train: arrivals before {test_start:%Y-%m-%d} (early stopping on 20% held-out ships); "
        f"test: arrivals from {test_start:%Y-%m-%d}. Errors in hours. "
        f"Test: {ts['voyages']} voyages, {ts['rows']:,} rows (one per 5 min).",
        "## Test, all", _table(overall(test).select(keep)),
        f"## Test, ships seen in training ({seen['voyage_id'].n_unique()} voyages)",
        _table(overall(seen).select(keep)),
        f"## Test, ships never seen ({unseen['voyage_id'].n_unique()} voyages)",
        _table(overall(unseen).select(keep)),
        "## MAE by state at prediction time", _table(breakdown(test, PRED_COLS)),
        "## MAE by true horizon (diagnostic: conditions on the outcome)",
        _table(breakdown(test, PRED_COLS, by="true_horizon")),
        "## Top features (gain share)",
        _table(importance(model).head(15).select("feature", "share")),
    ]  # fmt: skip
    (REPORTS_DIR / f"{name}.md").write_text("\n\n".join(md) + "\n")
    return result


# ------------------------------------------------------------------ v2: decomposed + quantiles


def interval_metrics(df: pl.DataFrame, lo: str, mid: str, hi: str) -> dict[str, float]:
    """Coverage of [lo, hi] (target 0.8 for P10-P90), width, mean pinball loss (hours)."""
    y = df["eta_min"].to_numpy() / 60
    q = {0.1: df[lo].to_numpy() / 60, 0.5: df[mid].to_numpy() / 60, 0.9: df[hi].to_numpy() / 60}
    pinball = np.mean([np.mean(np.maximum(a * (y - p), (a - 1) * (y - p))) for a, p in q.items()])
    return {
        "coverage_p10_p90": float(np.mean((y >= q[0.1]) & (y <= q[0.9]))),
        "below_p10": float(np.mean(y < q[0.1])),
        "above_p90": float(np.mean(y > q[0.9])),
        "median_width_h": float(np.median(q[0.9] - q[0.1])),
        "pinball_h": float(pinball),
    }


def residual_intervals(train: pl.DataFrame, df: pl.DataFrame, pred: str) -> pl.DataFrame:
    """Baseline interval: point forecast x empirical quantiles of the train log-residual,
    separately for waiting and moving ships."""
    out = []
    for waiting in (True, False):
        tr = train.filter(pl.col("waiting_now") == waiting)
        r = np.log1p(tr["eta_min"].to_numpy()) - np.log1p(tr[pred].to_numpy())
        q10, q50, q90 = np.quantile(r, [0.1, 0.5, 0.9])
        part = df.filter(pl.col("waiting_now") == waiting)
        base = np.log1p(part[pred].to_numpy())
        out.append(
            part.with_columns(
                base_p10=pl.Series(np.expm1(base + q10)),
                base_p50=pl.Series(np.expm1(base + q50)),
                base_p90=pl.Series(np.expm1(base + q90)),
            )
        )
    return pl.concat(out)


def run_v2(
    port: Port, test_start: dt.datetime, name: str = "eta_v2", feature_set: str = "base"
) -> dict:
    """feature_set: "base" (v2 as first built), "queue" (+ smoothed AIS queue and
    PortWatch) or "all" (+ Port of Los Angeles daily summary)."""
    features = FEATURE_SETS[feature_set]
    from sklearn.metrics import brier_score_loss, roc_auc_score

    from vessel_eta.decompose import DecomposedETA, add_targets, waiting_episodes

    feats, voyages = load_features(port)
    points = pl.read_parquet(PROCESSED_DIR / f"eta_points_{port.name}.parquet")
    feats = add_targets(feats, waiting_episodes(points))
    split = make_split(feats, voyages, test_start)

    v1 = fit_lgbm(split.train, split.valid)
    v2 = DecomposedETA.fit(split.train, split.valid, features)
    v2_raw_widen = dict(v2.widen)
    raw = v2.predict(split.test).select(
        raw_p10="eta_p10", raw_p50="eta_p50", raw_p90="eta_p90"
    )  # before the conformal step, for the report
    # conformal step on the held-out ships of the training period (never on March)
    v2.calibrate(split.valid)

    train_b = baseline_predictions(split.train, split.train)
    test = baseline_predictions(split.train, split.test).with_columns(
        pred_v1=pl.Series(v1.predict(split.test))
    )
    test = pl.concat([v2.predict(test), raw], how="horizontal").with_columns(
        pred_v2=pl.col("eta_p50")
    )
    test = residual_intervals(train_b, test, "pred_calibrated")

    cols = ["pred_naive", "pred_calibrated", "pred_v1", "pred_v2"]

    def point_table(df: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame([{"model": c.removeprefix("pred_"), **metrics(df, c)} for c in cols])

    def interval_table(df: pl.DataFrame) -> pl.DataFrame:
        rows = [
            {
                "interval": "calibrated x residual quantiles",
                **interval_metrics(df, "base_p10", "base_p50", "base_p90"),
            },
            {"interval": "v2 raw", **interval_metrics(df, "raw_p10", "raw_p50", "raw_p90")},
            {"interval": "v2 conformal", **interval_metrics(df, "eta_p10", "eta_p50", "eta_p90")},
        ]  # fmt: skip
        return pl.DataFrame(rows)

    slices = []
    for sname, cond in SLICES.items():
        part = test.filter(cond)
        row = {"slice": sname, "n": part.height}
        row.update({c.removeprefix("pred_"): metrics(part, c)["mae_h"] for c in cols})
        im = interval_metrics(part, "eta_p10", "eta_p50", "eta_p90")
        bm = interval_metrics(part, "base_p10", "base_p50", "base_p90")
        row["v2_coverage"] = im["coverage_p10_p90"]
        row["v2_width_h"] = im["median_width_h"]
        row["base_coverage"] = bm["coverage_p10_p90"]
        row["base_width_h"] = bm["median_width_h"]
        slices.append(row)

    mv = test.filter(~pl.col("waiting_now"))
    y = mv["stop_next"].cast(pl.Int8).to_numpy()
    p = mv["p_stop"].to_numpy()
    base_rate = float(split.train.filter(~pl.col("waiting_now"))["stop_next"].mean())
    rel = (
        mv.with_columns(bin=(pl.col("p_stop") * 5).floor().clip(upper_bound=4) / 5)
        .group_by("bin")
        .agg(pl.len().alias("n"), pl.col("p_stop").mean().alias("predicted"),
             pl.col("stop_next").cast(pl.Float64).mean().alias("observed"))
        .sort("bin")
    )  # fmt: skip
    classifier = {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "brier_base_rate": float(brier_score_loss(y, np.full(len(y), base_rate))),
        "test_rate": float(y.mean()),
        "train_rate": base_rate,
    }

    seen, unseen = test.filter("seen_ship"), test.filter(~pl.col("seen_ship"))
    result = {
        "name": name,
        "test_start": test_start.isoformat(),
        "feature_set": feature_set,
        "test_voyages": test["voyage_id"].n_unique(),
        "point": point_table(test).to_dicts(),
        "point_unseen_ships": point_table(unseen).to_dicts(),
        "point_seen_ships": point_table(seen).to_dicts(),
        "intervals": interval_table(test).to_dicts(),
        "intervals_unseen_ships": interval_table(unseen).to_dicts(),
        "slices": slices,
        "stop_classifier": classifier,
        "reliability": rel.to_dicts(),
        "routing_share_waiting": float(test["waiting_now"].mean()),
        "conformal_widening_log": {str(k): v for k, v in v2.widen.items()},
        "conformal_widening_before": {str(k): v for k, v in v2_raw_widen.items()},
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / f"{name}.json").write_text(json.dumps(result, indent=2, default=str))
    keep = ["model", "mae_h", "median_ae_h", "voyage_mae_h", "median_bias_h"]
    md = [
        f"# {name}: decomposed probabilistic ETA, features: {feature_set}",
        f"Train: arrivals before {test_start:%Y-%m-%d}; test: from {test_start:%Y-%m-%d}, "
        f"{result['test_voyages']} voyages. Hours. v1 is retrained on the same split.",
        "## Point forecast (v2 = median of the predicted distribution)",
        _table(point_table(test).select(keep)),
        "## Ships never seen in training", _table(point_table(unseen).select(keep)),
        "## Intervals (P10-P90, target coverage 0.80)", _table(interval_table(test)),
        "## Intervals, ships never seen", _table(interval_table(unseen)),
        "## By state at prediction time (MAE; P10-P90 coverage and median width, v2 vs baseline)",
        _table(pl.DataFrame(slices)),
        "## Will the ship wait before arriving? (moving ships)",
        _table(pl.DataFrame([classifier])),
        "Reliability (predicted vs observed rate, 5 bins):", _table(rel),
    ]  # fmt: skip
    (REPORTS_DIR / f"{name}.md").write_text("\n\n".join(md) + "\n")
    return result
