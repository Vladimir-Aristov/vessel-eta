"""Out-of-time test on 2024 with static vs monthly re-calibrated intervals.

The model (v2, base features) is trained on 2023 only and never refit. Intervals:
- static: conformal shifts estimated once, on held-out 2023 ships;
- rolling: at the start of each month of 2024 the shifts are re-estimated on the
  arrivals of the previous month (their outcomes are known by then); January uses
  the static shifts.

uv run python scripts/rolling_conformal.py
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl

from vessel_eta.config import PROCESSED_DIR, ROOT
from vessel_eta.decompose import (
    DecomposedETA,
    add_targets,
    apply_widen,
    conformal_widen,
    waiting_episodes,
)
from vessel_eta.experiment import _table, interval_metrics, load_features
from vessel_eta.features import BASE_FEATURES
from vessel_eta.geo import load_port
from vessel_eta.modeling import make_split

TEST_START = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
NO_WIDEN = {True: (0.0, 0.0), False: (0.0, 0.0)}


def main() -> None:
    port = load_port()
    feats, voyages = load_features(port)
    points = pl.read_parquet(PROCESSED_DIR / f"eta_points_{port.name}.parquet")
    feats = add_targets(feats, waiting_episodes(points))
    split = make_split(feats, voyages, TEST_START)
    model = DecomposedETA.fit(split.train, split.valid, BASE_FEATURES)
    static = dict(model.calibrate(split.valid).widen)
    model.widen = NO_WIDEN
    raw = model.predict(split.test).with_columns(
        month=pl.col("ts").dt.month(), arr_month=pl.col("arrival_ts").dt.month()
    )

    parts = []
    shifts = []
    for m in range(1, 13):
        rows = raw.filter((pl.col("ts").dt.year() == 2024) & (pl.col("month") == m))
        if m == 1:
            w = static
        else:
            calib = raw.filter(pl.col("arr_month") == m - 1)  # outcomes known at month start
            w = conformal_widen(calib, fallback=static)
        shifts.append({"month": m, **{f"{'wait' if k else 'move'}_{s}": v[i]
                       for k, v in w.items() for i, s in enumerate(("lo", "hi"))}})  # fmt: skip
        parts.append(apply_widen(rows, w))
    rolling = pl.concat(parts)
    static_pred = apply_widen(raw.filter(pl.col("ts").dt.year() == 2024), static)

    rows = []
    for name, df in [
        ("static (2023 held-out ships)", static_pred),
        ("rolling (previous month)", rolling),
    ]:
        for branch, cond in [("all", pl.lit(True)), ("moving", ~pl.col("waiting_now")),
                             ("waiting", pl.col("waiting_now"))]:  # fmt: skip
            m = interval_metrics(df.filter(cond), "eta_p10", "eta_p50", "eta_p90")
            rows.append({"intervals": name, "rows": branch, **m})

    def cov(df: pl.DataFrame, m: int) -> float:
        part = df.filter(pl.col("month") == m)
        return interval_metrics(part, "eta_p10", "eta_p50", "eta_p90")["coverage_p10_p90"]

    monthly = [
        {"month": m, "static": cov(static_pred, m), "rolling": cov(rolling, m)}
        for m in range(1, 13)
    ]
    t1, t2 = _table(pl.DataFrame(rows)), _table(pl.DataFrame(monthly))
    out = ROOT / "reports" / "rolling_conformal.md"
    out.write_text(
        "# 2024 out-of-time: static vs monthly re-calibrated intervals (P10-P90, target 0.80)\n\n"
        "Model trained on 2023 only, never refit.\n\n"
        + t1
        + "\n\n## Coverage by month\n\n"
        + t2
        + "\n"
    )
    (ROOT / "reports" / "rolling_conformal.json").write_text(
        json.dumps({"summary": rows, "monthly": monthly, "shifts": shifts}, indent=2)
    )
    print(t1)
    print(t2)
    print(np.round(pl.DataFrame(shifts).to_numpy(), 3))


if __name__ == "__main__":
    main()
