import dataclasses
import datetime as dt

import numpy as np
import polars as pl
import pytest
from synthetic import DEG_LAT_PER_MIN_AT_10KN, GATE, PORT, T0, approach, stay, track

from vessel_eta.features import ALL_FEATURES as FEATURES
from vessel_eta.features import build_features, port_state
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


def test_portwatch_features_are_lagged_week_sums():
    from vessel_eta.data.portwatch import COLUMNS, LAG_DAYS, lagged_features

    days = pl.date_range(dt.date(2023, 1, 1), dt.date(2023, 1, 31), "1d", eager=True)
    daily = pl.DataFrame({"day": days, **{c: list(range(1, 32)) for c in COLUMNS}})
    daily = daily.filter(pl.col("day") != dt.date(2023, 1, 20))  # a missing day
    f = lagged_features(daily).sort("day")
    target = dt.date(2023, 1, 1) + dt.timedelta(days=6 + LAG_DAYS)  # first full window, lagged
    row = f.filter(pl.col("day") == target).row(0, named=True)
    assert row["pw_container_calls_7d"] == sum(range(1, 8))  # Jan 1..7, known on Jan 7 + lag
    # windows touching the missing day are null, not under-counted
    hole = f.filter(pl.col("day") == dt.date(2023, 1, 20) + dt.timedelta(days=LAG_DAYS))
    assert hole["pw_import_7d"].is_null().all()


POLA_TEXT = """HISTORICAL CONTAINER
VESSEL ACTIVITY
Date POLA Vessels at Anchor POLA Vessels at Berth POLA Vessels Departed Average Days at Berth Average Days at ANC + Berth
1/3/2023 0 14 6 4.83 4.83
1/4/2023 2 12 1 5 7
1/5/2023 3 11 0 0 0
Prepared by the Wharfinger Division
"""


def test_pola_parse_and_lag():
    from vessel_eta.data.pola import features, parse_text

    daily = parse_text(POLA_TEXT)
    assert daily.height == 3 and daily["anchor"].to_list() == [0, 2, 3]
    f = features(daily)
    jan4 = f.filter(pl.col("known_from") == dt.date(2023, 1, 5)).row(0, named=True)
    assert jan4["pola_anchor"] == 2  # the Jan 4 report is used from Jan 5, not on Jan 4
    assert jan4["pola_wait_days_5r"] == pytest.approx((0 + 2) / 2)
    jan5 = f.filter(pl.col("known_from") == dt.date(2023, 1, 6)).row(0, named=True)
    assert jan5["pola_wait_days_5r"] == pytest.approx(1.0)  # no departures: wait undefined, skipped


def test_pola_rejects_unparseable_rows():
    from vessel_eta.data.pola import parse_text

    with pytest.raises(ValueError):
        parse_text("1/3/2023 0 14 six 4.83 4.83")


def test_pola_join_never_uses_same_day_report(data):
    tracks, voyages, points = data
    from vessel_eta.data.pola import parse_text

    day = T0.date()
    text = f"{day:%-m/%-d/%Y} 7 10 1 2 3\n{(day + dt.timedelta(days=1)):%-m/%-d/%Y} 9 10 1 2 3"
    f = build_features(
        points, voyages, port_state(tracks, PORT2), PORT2, None, pola_daily=parse_text(text)
    )
    same_day = f.filter(pl.col("ts").dt.date() == day)
    assert same_day["pola_anchor"].is_null().all()
    next_day = f.filter(pl.col("ts").dt.date() == day + dt.timedelta(days=1))
    if next_day.height:
        assert (next_day["pola_anchor"] == 7).all()
