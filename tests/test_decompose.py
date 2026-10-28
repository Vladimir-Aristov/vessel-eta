import datetime as dt

import numpy as np
import polars as pl
import pytest
from synthetic import DEG_LAT_PER_MIN_AT_10KN, GATE, PORT, T0, approach, stay, track

from vessel_eta.decompose import (
    QUANTILES,
    add_targets,
    compose_moving,
    sample_from_quantiles,
    waiting_episodes,
)
from vessel_eta.experiment import interval_metrics
from vessel_eta.voyages import build_voyages

INSIDE = 33.75


def voyage_points(rows):
    _, points = build_voyages(track(1, rows).sort("mmsi", "ts"), PORT)
    return points


def into_port(t0, start_lat, minutes):
    rows = [
        (t0 + i, start_lat + i * DEG_LAT_PER_MIN_AT_10KN, GATE[0], 10.0) for i in range(minutes)
    ]
    return rows + stay(t0 + minutes, 30, INSIDE)


def minutes_to_gate(lat):
    return int((GATE[1] - lat) / DEG_LAT_PER_MIN_AT_10KN) + 1


def test_anchor_stop_is_one_episode():
    rows = approach(0, 60, 33.40)  # steam for 1 h
    lat = rows[-1][1]
    rows += stay(60, 180, lat)  # 3 h at anchor
    rows += into_port(240, lat, minutes_to_gate(lat))
    eps = waiting_episodes(voyage_points(rows))
    assert eps.height == 1
    start, end = eps.row(0)[1:]
    # centred 30-min median: boundaries within a few minutes of the real stop
    assert abs((start - (T0 + dt.timedelta(minutes=60))).total_seconds()) <= 3 * 60
    assert abs((end - (T0 + dt.timedelta(minutes=239))).total_seconds()) <= 3 * 60


def test_pilot_slowdown_is_not_waiting():
    rows = approach(0, 60, 33.40)
    lat = rows[-1][1]
    rows += [(60 + i, lat, GATE[0], 1.0) for i in range(40)]  # 40 min crawl < 1 h
    rows += into_port(100, lat, minutes_to_gate(lat))
    assert waiting_episodes(voyage_points(rows)).height == 0


def test_close_stops_merge_into_one_episode():
    rows = approach(0, 30, 33.40)
    lat = rows[-1][1]
    rows += stay(30, 120, lat)  # 2 h
    rows += [(150 + i, lat + i * 0.0002, GATE[0], 4.0) for i in range(120)]  # 2 h shift
    lat2 = rows[-1][1]
    rows += stay(270, 120, lat2)  # 2 h more
    rows += into_port(390, lat2, minutes_to_gate(lat2))
    assert waiting_episodes(voyage_points(rows)).height == 1


def test_targets_before_during_after_episode():
    rows = approach(0, 60, 33.40)
    lat = rows[-1][1]
    rows += stay(60, 180, lat)
    rows += into_port(240, lat, minutes_to_gate(lat))
    points = voyage_points(rows)
    eps = waiting_episodes(points)
    feats = points.with_columns(sog_mean_30m=pl.col("sog")).select(
        "voyage_id", "ts", "sog_mean_30m", "eta_min"
    )
    t = add_targets(feats, eps)
    ep_start = eps["ep_start"][0]

    before = t.filter(pl.col("ts") == T0 + dt.timedelta(minutes=10)).row(0, named=True)
    assert before["stop_next"] and not before["waiting_now"]
    assert before["t_next_min"] == pytest.approx(
        (ep_start - (T0 + dt.timedelta(minutes=10))).total_seconds() / 60
    )
    assert before["after_stop_min"] == pytest.approx(before["eta_min"] - before["t_next_min"])

    during = t.filter(pl.col("ts") == T0 + dt.timedelta(minutes=150)).row(0, named=True)
    assert during["waiting_now"] and during["stop_next"] and during["t_next_min"] == 0

    after = t.filter(pl.col("ts") == T0 + dt.timedelta(minutes=250)).row(0, named=True)
    assert not after["stop_next"]
    assert after["t_next_min"] == pytest.approx(after["eta_min"])
    assert after["after_stop_min"] is None


def test_sampling_hits_the_knots():
    q_log = np.log1p(np.array([[10, 20, 30, 40, 50, 60, 70]], dtype=float))
    s = sample_from_quantiles(q_log, QUANTILES)
    np.testing.assert_allclose(s[0], [10, 20, 30, 40, 50, 60, 70], rtol=1e-9)
    mid = sample_from_quantiles(q_log, np.array([0.375]))  # halfway between 0.25 and 0.5 knots
    assert 30 < mid[0, 0] < 40


def test_composition_limits():
    t_q = np.log1p(np.tile(np.linspace(50, 150, len(QUANTILES)), (2, 1)))
    w_q = np.log1p(np.tile(np.linspace(1000, 3000, len(QUANTILES)), (2, 1)))
    out = compose_moving(np.array([0.0, 1.0]), t_q, w_q, n_samples=4000)
    never, always = out
    assert 50 <= never[0] <= never[1] <= never[2] <= 150  # only travel time
    assert always[1] == pytest.approx(100 + 2000, rel=0.05)  # medians add (symmetric knots)
    assert np.all(np.diff(out, axis=1) >= 0)


def test_interval_metrics():
    df = pl.DataFrame(
        {
            "eta_min": [60.0, 120, 600],
            "lo": [30.0, 130, 100],
            "mid": [60.0, 140, 200],
            "hi": [90.0, 200, 300],
        }
    )
    m = interval_metrics(df, "lo", "mid", "hi")
    assert m["coverage_p10_p90"] == pytest.approx(1 / 3)
    assert m["below_p10"] == pytest.approx(1 / 3) and m["above_p90"] == pytest.approx(1 / 3)


def test_conformal_widen_targets_ten_percent_each_side():
    from vessel_eta.decompose import apply_widen, conformal_widen

    rng = np.random.default_rng(0)
    n = 4000
    truth = np.exp(rng.normal(5, 1, n))
    # deliberately too-narrow raw interval around the right median
    pred = pl.DataFrame({
        "waiting_now": rng.random(n) < 0.5,
        "eta_min": truth,
        "eta_p50": np.full(n, np.exp(5.0)),
        "eta_p10": np.full(n, np.exp(4.8)),
        "eta_p90": np.full(n, np.exp(5.2)),
    })  # fmt: skip
    widened = apply_widen(pred, conformal_widen(pred))
    y = widened["eta_min"].to_numpy()
    assert abs(np.mean(y < widened["eta_p10"].to_numpy()) - 0.1) < 0.02
    assert abs(np.mean(y > widened["eta_p90"].to_numpy()) - 0.1) < 0.02
    assert (widened["eta_p10"] <= widened["eta_p50"]).all()
    assert (widened["eta_p90"] >= widened["eta_p50"]).all()
