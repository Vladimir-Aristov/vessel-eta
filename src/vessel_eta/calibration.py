"""Online calibration of the P10 / P90 interval: quantile tracking with delayed feedback.

The static conformal step (`decompose.conformal_widen`) estimates log-space shifts once;
in 2024 they went stale (README, "Out-of-time test"). Here the shifts follow the data:

    every day d, for each branch (waiting / under way) and each side:
        predictions made on day d use the current shift
        then, for the predictions whose voyage *arrived* on day d (outcome now known):
            miss = share of outcomes outside that side of the interval
            shift += step * (miss - 0.10)

i.e. the online quantile tracker of conformal PID control (Angelopoulos et al., 2023)
applied to the conformity scores log(y) - log(P90) and log(P10) - log(y). A prediction
never sees an outcome that arrived on or after its own day.
"""

from __future__ import annotations

import numpy as np
import polars as pl

TARGET_MISS = 0.10  # per side -> 80% interval


def track_shifts(
    raw: pl.DataFrame,
    initial: dict[bool, tuple[float, float]],
    step: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Apply daily-updated shifts to raw (un-widened) predictions.

    raw: rows with ts, arrival_ts, waiting_now, eta_min, eta_p10, eta_p50, eta_p90.
    Returns (rows with widened eta_p10 / eta_p90, the daily shift history).
    """
    df = raw.with_row_index("_i").with_columns(
        _day=pl.col("ts").dt.date(),
        _arr=pl.col("arrival_ts").dt.date(),
    )
    n = df.height
    y = np.log1p(df["eta_min"].to_numpy())
    lo = np.log1p(df["eta_p10"].to_numpy())
    hi = np.log1p(df["eta_p90"].to_numpy())
    branch = df["waiting_now"].to_numpy().astype(bool)
    pred_day = df["_day"].to_numpy()
    arr_day = df["_arr"].to_numpy()

    shift = {b: list(initial[b]) for b in (True, False)}
    used_lo = np.empty(n)
    used_hi = np.empty(n)
    by_pred = _index_by(pred_day)
    by_arr = _index_by(arr_day)
    history = []
    for day in sorted(set(by_pred) | set(by_arr)):
        rows = by_pred.get(day)
        if rows is not None:  # predictions made today use the shifts known so far
            for b in (True, False):
                r = rows[branch[rows] == b]
                used_lo[r], used_hi[r] = shift[b]
        rows = by_arr.get(day)
        if rows is not None:  # outcomes revealed at the end of today
            for b in (True, False):
                r = rows[branch[rows] == b]
                if not len(r):
                    continue
                miss_lo = np.mean(y[r] < lo[r] - shift[b][0])
                miss_hi = np.mean(y[r] > hi[r] + shift[b][1])
                shift[b][0] += step * (miss_lo - TARGET_MISS)
                shift[b][1] += step * (miss_hi - TARGET_MISS)
        history.append({
            "day": day,
            "waiting_lo": shift[True][0], "waiting_hi": shift[True][1],
            "moving_lo": shift[False][0], "moving_hi": shift[False][1],
        })  # fmt: skip

    p50 = df["eta_p50"].to_numpy()
    out = df.with_columns(
        eta_p10=pl.Series(np.minimum(np.expm1(lo - used_lo).clip(min=0), p50)),
        eta_p90=pl.Series(np.maximum(np.expm1(hi + used_hi), p50)),
    ).drop("_i", "_day", "_arr")
    return out, pl.DataFrame(history)


def _index_by(days: np.ndarray) -> dict:
    order = np.argsort(days, kind="stable")
    sorted_days = days[order]
    cuts = np.flatnonzero(sorted_days[1:] != sorted_days[:-1]) + 1
    return {g[0]: order[s:e] for g, s, e in zip(
        np.split(sorted_days, cuts), np.r_[0, cuts], np.r_[cuts, len(days)], strict=True
    )}  # fmt: skip
