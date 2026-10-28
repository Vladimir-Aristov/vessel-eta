import dataclasses
import datetime as dt

import numpy as np
import polars as pl
import pytest
from synthetic import DEG_LAT_PER_MIN_AT_10KN, GATE, PORT, T0, approach, stay, track

from vessel_eta.features import FEATURES, build_features, port_state
from vessel_eta.modeling import make_split
from vessel_eta.voyages import build_voyages

# feature code names the two LA/LB gates
PORT2 = dataclasses.replace(
    PORT, gates={"angels_gate": GATE, "queens_gate": (GATE[0] + 0.05, GATE[1])}
)
INSIDE = 33.75


def ship(mmsi, t0):
    """Approach, wait at anchor ~3 h, slow approach, enter port, berth."""
    rows = approach(t0, 90, 33.30)
    lat = rows[-1][1]
    rows += stay(t0 + 90, 180, lat)  # anchored
    rows += [
        (t0 + 270 + i, lat + i * DEG_LAT_PER_MIN_AT_10KN * 0.6, GATE[0], 6.0) for i in range(80)
    ]
    last = rows[-1][1]
    n = int((GATE[1] - last) / (DEG_LAT_PER_MIN_AT_10KN * 0.6)) + 1
    rows += [
        (t0 + 350 + i, last + i * DEG_LAT_PER_MIN_AT_10KN * 0.6, GATE[0], 6.0) for i in range(n + 2)
    ]
    rows += stay(t0 + 352 + n, 40, INSIDE)
    return track(mmsi, rows)


@pytest.fixture(scope="module")
def data():
    tracks = pl.concat([ship(1, 0), ship(2, 120), ship(3, 400)]).sort("mmsi", "ts")
    voyages, points = build_voyages(tracks, PORT2)
    assert voyages.height == 3 and voyages["labelled"].all()
    return tracks, voyages, points


def features_at_cutoff(tracks, voyages, points, cutoff):
    pts = points.filter(pl.col("ts") <= cutoff)
    state = port_state(tracks.filter(pl.col("ts") <= cutoff), PORT2)
    return build_features(pts, voyages, state, PORT2, every_min=None)


# dense cut-offs: some fall mid-bucket while a ship starts waiting (catches same-bucket leaks)
@pytest.mark.parametrize("minutes", range(15, 720, 10))
def test_features_never_look_ahead(data, minutes):
    tracks, voyages, points = data
    full = build_features(points, voyages, port_state(tracks, PORT2), PORT2, every_min=None)
    cutoff = T0 + dt.timedelta(minutes=minutes)
    cut = features_at_cutoff(tracks, voyages, points, cutoff)
    assert cut.height > 0
    joined = cut.join(full, on=["voyage_id", "ts"], suffix="_full")
    assert joined.height == cut.height
    for f in FEATURES:
        a = joined[f].cast(pl.Float64).to_numpy()
        b = joined[f + "_full"].cast(pl.Float64).to_numpy()
        np.testing.assert_allclose(a, b, rtol=1e-6, equal_nan=True, err_msg=f)


def test_features_are_sane(data):
    tracks, voyages, points = data
    f = build_features(points, voyages, port_state(tracks, PORT2), PORT2, every_min=None)
    assert f["port_waiting"].max() >= 1  # someone was anchored while others approached
    anchored = f.filter(pl.col("stationary_share_2h") == 1)
    assert anchored.height > 0 and (anchored["stationary_min_total"] > 0).all()
    moving_in = f.filter(pl.col("sog") > 5)
    assert (moving_in["closing_kmh_1h"].drop_nulls() > 0).mean() > 0.9  # heading to the gate


def test_downsampling_keeps_one_row_per_bucket(data):
    tracks, voyages, points = data
    f = build_features(points, voyages, port_state(tracks, PORT2), PORT2, every_min=5)
    per_bucket = f.group_by("voyage_id", pl.col("ts").dt.truncate("5m")).len()
    assert per_bucket["len"].max() == 1


def test_split_separates_time_and_ships(data):
    tracks, voyages, points = data
    f = build_features(points, voyages, port_state(tracks, PORT2), PORT2, every_min=None)
    start = voyages["arrival_ts"].sort()[2]  # last ship's arrival goes to test
    s = make_split(f, voyages, start, valid_frac=0.5, seed=0)
    assert (s.train["arrival_ts"] < start).all() and (s.test["arrival_ts"] >= start).all()
    assert not set(s.train["mmsi"]) & set(s.valid["mmsi"])
    assert s.train.height and s.valid.height and s.test.height
    assert not s.test["seen_ship"].any()  # ship 3 was never in the training period
