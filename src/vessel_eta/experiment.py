"""End-to-end ETA experiment: features -> split -> baselines + LightGBM -> report."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl

from vessel_eta.config import PROCESSED_DIR, ROOT
from vessel_eta.features import build_features
from vessel_eta.geo import Port
from vessel_eta.modeling import (
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
    voyages = pl.read_parquet(processed / f"voyages_{port.name}.parquet").filter("labelled")
    points = pl.read_parquet(processed / f"eta_points_{port.name}.parquet")
    state = pl.read_parquet(processed / f"port_state_{port.name}.parquet")
    return build_features(points, voyages, state, port), voyages


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
