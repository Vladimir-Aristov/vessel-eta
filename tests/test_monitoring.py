import numpy as np

from vessel_eta.monitoring import PSI_ALERT, PSI_WARN, _edges, _shares, psi


def _ref(values):
    e = _edges(values)
    return {"edges": e, "shares": _shares(values, e), "null_share": 0.0}


def test_psi_is_near_zero_for_the_same_distribution():
    rng = np.random.default_rng(0)
    ref = _ref(rng.normal(0, 1, 20_000))
    assert psi(ref, rng.normal(0, 1, 20_000)) < 0.01


def test_psi_flags_a_shift():
    rng = np.random.default_rng(1)
    ref = _ref(rng.normal(0, 1, 20_000))
    assert PSI_WARN < psi(ref, rng.normal(0.4, 1, 20_000)) < PSI_ALERT * 2
    assert psi(ref, rng.normal(1.5, 1, 20_000)) > PSI_ALERT


def test_shares_sum_to_one_and_cover_outliers():
    v = np.arange(100.0)
    e = _edges(v)
    s = _shares(np.r_[v, -1e9, 1e9], e)
    assert abs(sum(s) - 1) < 1e-9 and len(s) == len(e) + 1


def test_drift_limit_respects_normal_month_to_month_variation():
    from vessel_eta.monitoring import drift_limit

    assert drift_limit({"normal_monthly_psi": 0.05}) == PSI_ALERT  # never below the convention
    assert drift_limit({"normal_monthly_psi": 1.8}) == 1.8  # slow inputs: months differ a lot
    assert drift_limit({}) == PSI_ALERT  # old artifacts without the statistic


def test_reference_profile_measures_normal_monthly_psi():
    import datetime as dt

    import polars as pl

    from vessel_eta.monitoring import MONITORED, reference_profile

    rng = np.random.default_rng(0)
    n = 24_000
    month = np.repeat(np.arange(12), n // 12)
    df = pl.DataFrame({c: rng.normal(0, 1, n) for c in MONITORED}).with_columns(
        # a slowly varying input: each month sits at its own level
        port_waiting=pl.Series(month * 1.0 + rng.normal(0, 0.1, n)),
        arrival_ts=pl.Series([dt.datetime(2023, m + 1, 15, tzinfo=dt.UTC) for m in month]),
    )
    ref = reference_profile(df, np.exp(rng.normal(5, 1, n)))
    assert ref["sog"]["normal_monthly_psi"] < 0.05
    assert ref["port_waiting"]["normal_monthly_psi"] > 1.0
