import datetime as dt

import numpy as np
import polars as pl

from vessel_eta.calibration import track_shifts

T0 = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)


def _raw(n_days=120, per_day=200, seed=0):
    """Each day: predictions whose voyages arrive the next day; raw interval too narrow."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        ts = T0 + dt.timedelta(days=d, hours=1)
        y = np.exp(rng.normal(5.0, 1.0, per_day))  # true minutes, log-normal
        for k in range(per_day):
            rows.append({
                "ts": ts, "arrival_ts": ts + dt.timedelta(days=1),
                "waiting_now": bool(k % 2), "eta_min": float(y[k]),
                "eta_p10": float(np.exp(4.8)), "eta_p50": float(np.exp(5.0)),
                "eta_p90": float(np.exp(5.2)),
            })  # fmt: skip
    return pl.DataFrame(rows)


def test_tracker_converges_to_ten_percent_each_side():
    raw = _raw()
    out, hist = track_shifts(raw, {True: (0.0, 0.0), False: (0.0, 0.0)}, step=0.5)
    late = out.filter(pl.col("ts") > T0 + dt.timedelta(days=60))  # after burn-in
    y = late["eta_min"].to_numpy()
    assert abs(np.mean(y < late["eta_p10"].to_numpy()) - 0.10) < 0.03
    assert abs(np.mean(y > late["eta_p90"].to_numpy()) - 0.10) < 0.03
    assert hist["moving_hi"][-1] > 0.5  # widened a lot from the too-narrow start


def test_tracker_never_uses_outcomes_from_the_future():
    raw = _raw(n_days=30, per_day=50)
    base, _ = track_shifts(raw, {True: (0.0, 0.0), False: (0.0, 0.0)}, step=0.5)
    # change every outcome revealed on day 20 or later
    cut = (T0 + dt.timedelta(days=20)).date()
    changed = raw.with_columns(
        eta_min=pl.when(pl.col("arrival_ts").dt.date() >= cut)
        .then(pl.col("eta_min") * 50)
        .otherwise(pl.col("eta_min"))
    )
    alt, _ = track_shifts(changed, {True: (0.0, 0.0), False: (0.0, 0.0)}, step=0.5)
    before = pl.col("ts").dt.date() <= cut  # predictions made up to the day of the change
    for c in ("eta_p10", "eta_p90"):
        np.testing.assert_allclose(
            base.filter(before)[c].to_numpy(), alt.filter(before)[c].to_numpy()
        )
    after = pl.col("ts").dt.date() > cut
    assert not np.allclose(
        base.filter(after)["eta_p90"].to_numpy(), alt.filter(after)["eta_p90"].to_numpy()
    )
