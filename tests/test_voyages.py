import datetime as dt

import polars as pl
import pytest
from synthetic import DEG_LAT_PER_MIN_AT_10KN, GATE, PORT, T0, approach, stay, track

from vessel_eta.voyages import build_voyages, summarize

INSIDE = 33.75


def run(df):
    return build_voyages(df.sort("mmsi", "ts"), PORT)


def arrival_rows(start_min=0, minutes=120, stay_min=30):
    """Steam north from outside, cross the gate line, then berth inside."""
    start_lat = GATE[1] - (minutes - 0.5) * DEG_LAT_PER_MIN_AT_10KN  # last point just outside
    rows = approach(start_min, minutes, start_lat)
    return rows + stay(start_min + minutes, stay_min, INSIDE)


def test_simple_arrival():
    voyages, labels = run(track(1, arrival_rows()))
    assert voyages.height == 1
    v = voyages.row(0, named=True)
    assert v["arrival_ts"] == T0 + dt.timedelta(minutes=120)
    assert v["entry_gate"] == "gate" and not v["from_port"] and v["labelled"]
    assert labels.height == 120
    eta = labels["eta_min"].to_list()
    assert eta[0] == pytest.approx(120) and eta[-1] == pytest.approx(1)
    assert all(a > b for a, b in zip(eta, eta[1:], strict=False))  # strictly decreasing
    assert labels["gate_km"][-1] < 0.2


def test_boundary_jitter_is_not_an_arrival():
    rows = approach(0, 60, 33.55)
    rows += [(60 + i, 33.701, -118.20, 1.0) for i in range(5)]  # 5 min inside < 10 min
    rows += [(65 + i, 33.69, -118.20, 5.0) for i in range(60)]
    voyages, labels = run(track(1, rows))
    assert voyages.height == 0 and labels.height == 0


def test_passing_ship_gets_no_labels():
    rows = [(i, 33.60, -118.40 + i * 0.003, 10.0) for i in range(120)]  # sails east, south of port
    voyages, labels = run(track(1, rows))
    assert voyages.height == 0 and labels.height == 0


def test_only_last_segment_before_long_gap_is_labelled():
    early = approach(0, 30, 33.30)  # seen, then silent for 10 h
    late = arrival_rows(start_min=30 + 600, minutes=60)
    voyages, labels = run(track(1, early + late))
    assert voyages.height == 1
    assert labels.height == 60
    assert voyages["start_ts"][0] == T0 + dt.timedelta(minutes=630)


def test_unsure_arrival_time_is_not_labelled():
    rows = approach(0, 60, 33.50)
    rows += stay(60 + 40, 30, INSIDE)  # 40-min hole right at the entrance
    voyages, labels = run(track(1, rows))
    assert voyages.height == 1
    assert not voyages["labelled"][0]
    assert voyages["entry_gap_min"][0] == pytest.approx(41)
    assert labels.height == 0


def test_departure_and_return_is_flagged_from_port():
    rows = stay(0, 60, INSIDE)  # in port at the start
    rows += [(60 + i, 33.69 - i * 0.001, -118.20, 8.0) for i in range(30)]  # leaves
    rows += arrival_rows(start_min=90, minutes=60)  # comes back
    voyages, _ = run(track(1, rows))
    assert voyages.height == 1
    assert voyages["from_port"][0]


def test_two_ships_two_voyages_and_summary():
    df = pl.concat([track(1, arrival_rows()), track(2, arrival_rows(start_min=500, minutes=90))])
    voyages, labels = run(df)
    assert voyages.height == 2 and labels["voyage_id"].n_unique() == 2
    s = summarize(voyages, labels)
    assert s["arrivals_labelled"] == 2 and s["ships"] == 2 and s["labelled_points"] == 210
