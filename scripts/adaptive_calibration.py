"""Interval calibration over time: static vs monthly re-calibration vs daily quantile tracking.

1. Tune the tracker step on Oct-Dec 2023 with a model trained on Jan-Sep 2023.
2. Test on all of 2024 with a model trained on 2023 (never refit):
   - static:   conformal shifts from held-out 2023 ships, fixed;
   - monthly:  shifts re-estimated at each month start from last month's arrivals;
   - tracking: shifts updated daily from the arrivals of that day (calibration.py).

uv run python scripts/adaptive_calibration.py
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl

from vessel_eta.calibration import track_shifts
from vessel_eta.config import ROOT
from vessel_eta.decompose import DecomposedETA, apply_widen, conformal_widen
from vessel_eta.experiment import _table, interval_metrics
from vessel_eta.features import BASE_FEATURES
from vessel_eta.geo import load_port
from vessel_eta.modeling import make_split
from vessel_eta.production import _labelled_features

UTC = dt.UTC
NO_WIDEN = {True: (0.0, 0.0), False: (0.0, 0.0)}
STEPS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def fit_raw(feats, voyages, train_end, test_end):
    """Model trained on arrivals < train_end; raw predictions on [train_end, test_end)."""
    split = make_split(feats, voyages, train_end, test_end=test_end)
    model = DecomposedETA.fit(split.train, split.valid, BASE_FEATURES)
    static = dict(model.calibrate(split.valid).widen)
    model.widen = NO_WIDEN
    raw = model.predict(split.test).filter(pl.col("ts") >= train_end)
    return raw, static


def score(df: pl.DataFrame) -> dict[str, float]:
    return interval_metrics(df, "eta_p10", "eta_p50", "eta_p90")


def monthly(raw: pl.DataFrame, static) -> pl.DataFrame:
    raw = raw.with_columns(_m=pl.col("ts").dt.month(), _am=pl.col("arrival_ts").dt.month())
    parts = []
    for m in sorted(raw["_m"].unique()):
        w = static if m == raw["_m"].min() else conformal_widen(
            raw.filter(pl.col("_am") == m - 1), fallback=static
        )  # fmt: skip
        parts.append(apply_widen(raw.filter(pl.col("_m") == m), w))
    return pl.concat(parts).drop("_m", "_am")


def main() -> None:
    port = load_port()
    feats, voyages = _labelled_features(port)

    # 1) tune the step on Oct-Dec 2023
    raw23, static23 = fit_raw(
        feats, voyages, dt.datetime(2023, 10, 1, tzinfo=UTC), dt.datetime(2024, 1, 1, tzinfo=UTC)
    )
    tuning = [{"step": s, **score(track_shifts(raw23, static23, s)[0])} for s in STEPS]
    best = min(tuning, key=lambda r: r["pinball_h"])["step"]
    print(_table(pl.DataFrame(tuning)), "\nchosen step:", best, flush=True)

    # 2) test on 2024
    raw24, static24 = fit_raw(feats, voyages, dt.datetime(2024, 1, 1, tzinfo=UTC), None)
    methods = {
        "static": apply_widen(raw24, static24),
        "monthly re-calibration": monthly(raw24, static24),
        f"daily tracking (step {best})": track_shifts(raw24, static24, best)[0],
    }
    rows, by_month = [], []
    for name, df in methods.items():
        for branch, cond in [("all", pl.lit(True)), ("under way", ~pl.col("waiting_now")),
                             ("waiting", pl.col("waiting_now"))]:  # fmt: skip
            rows.append({"method": name, "rows": branch, **score(df.filter(cond))})
        df = df.with_columns(_m=pl.col("ts").dt.month())
        cov = {m: score(df.filter(pl.col("_m") == m))["coverage_p10_p90"] for m in range(1, 13)}
        by_month.append({"method": name, **{f"{m:02d}": v for m, v in cov.items()},
                         "worst month": min(cov.values()),
                         "spread": float(np.std(list(cov.values())))})  # fmt: skip
    t_test, t_month = _table(pl.DataFrame(rows)), _table(pl.DataFrame(by_month))
    print(t_test)
    print(t_month)
    out = ROOT / "reports" / "adaptive_calibration"
    out.with_suffix(".md").write_text(
        "# Interval calibration over time (P10-P90, target 0.80)\n\n"
        "## Step chosen on Oct-Dec 2023 (model trained Jan-Sep 2023)\n\n"
        + _table(pl.DataFrame(tuning))
        + f"\n\nchosen step: {best}\n\n"
        "## Test: 2024, model trained on 2023, never refit\n\n"
        + t_test
        + "\n\n## Coverage by month of 2024\n\n"
        + t_month
        + "\n"
    )
    out.with_suffix(".json").write_text(
        json.dumps({"tuning": tuning, "step": best, "test": rows, "by_month": by_month}, indent=2)
    )


if __name__ == "__main__":
    main()
