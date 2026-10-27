import polars as pl
from synthetic import PORT, approach, stay, track

from vessel_eta.tracks import clean_tracks


def run(df):
    return clean_tracks(df, PORT.tracks)


def test_clean_track_is_untouched():
    df = track(1, approach(0, 60, 33.50))
    out, st = run(df)
    assert out.height == 60
    assert st.points_dropped_spikes == st.points_dropped_ghosts == 0


def test_single_spike_is_dropped():
    rows = approach(0, 60, 33.50)
    rows[30] = (30, 34.50, -119.50, 10.0)  # ~150 km away for one minute
    out, st = run(track(1, rows))
    assert out.height == 59
    assert st.points_dropped_spikes == 1
    assert 34.50 not in out["lat"].to_list()


def test_interleaved_ghost_is_dropped():
    main = approach(0, 200, 33.40)
    ghost = [(m + 0.5, 33.75, -118.25, 0.0) for m in range(0, 200, 10)]  # 20 pts in port
    out, st = run(track(1, main + ghost))
    assert out.height == 200
    assert st.points_dropped_ghosts == 20
    assert st.mmsi_with_ghosts == 1


def test_stale_replay_is_rejected_by_sog_check():
    # anchored ship (SOG 0) "moves" 3 km in 4 min: 24 kn is under the 40 kn cap,
    # but a ship reporting 0-7 kn cannot do it
    rows = stay(0, 30, 33.65)
    rows.append((30 + 4, 33.65 - 0.027, -118.20, 6.7))
    rows += stay(30 + 9, 30, 33.65)
    out, _ = run(track(1, rows))
    assert out.height == 60
    assert out["lat"].min() == 33.65


def test_old_spike_cannot_steal_points_after_a_gap():
    rows = approach(0, 30, 33.50)  # main track up to ~33.58
    rows.append((31, 33.62, -118.20, 10.0))  # spike ahead (too far for 1 min)
    # 3 h of silence, then the ship passes right through the spike's location
    rows += approach(211, 60, 33.60)
    out, st = run(track(1, rows))
    assert out.height == 90
    assert st.points_dropped_spikes == 1


def test_ships_are_independent():
    df = pl.concat([track(1, approach(0, 30, 33.5)), track(2, approach(0, 30, 33.0))])
    out, st = run(df)
    assert out.height == 60 and st.mmsi == 2
