import dataclasses
import datetime as dt

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from synthetic import DEG_LAT_PER_MIN_AT_10KN, GATE, PORT, T0, approach, stay, track

from vessel_eta import artifact
from vessel_eta.decompose import DecomposedETA, add_targets, waiting_episodes
from vessel_eta.features import BASE_FEATURES, build_features, port_state
from vessel_eta.serving.app import create_app
from vessel_eta.serving.predictor import AlreadyInPort, Static, request_features, request_table
from vessel_eta.voyages import build_voyages

PORT2 = dataclasses.replace(
    PORT, gates={"angels_gate": GATE, "queens_gate": (GATE[0] + 0.05, GATE[1])}
)
INSIDE = 33.75


def ship(mmsi, t0, anchor_min, speed=10.0):
    step = DEG_LAT_PER_MIN_AT_10KN * speed / 10
    rows = [(t0 + i, 33.30 + i * step, GATE[0], speed) for i in range(60)]
    lat = rows[-1][1]
    if anchor_min:
        rows += stay(t0 + 60, anchor_min, lat)
    t = t0 + 60 + anchor_min
    n = int((GATE[1] - lat) / step) + 2
    rows += [(t + i, lat + i * step, GATE[0], speed) for i in range(n)]
    rows += stay(t + n, 30, INSIDE)
    return track(mmsi, rows)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    ships = [
        ship(200000000 + k, 600 * k, anchor_min=[0, 180, 0, 300, 120, 0, 240, 0][k],
             speed=[8, 10, 12, 9, 11, 13, 10, 12][k])
        for k in range(8)
    ]  # fmt: skip
    tracks = pl.concat(ships).sort("mmsi", "ts")
    voyages, points = build_voyages(tracks, PORT2)
    state = port_state(tracks, PORT2)
    feats = add_targets(
        build_features(points, voyages, state, PORT2, every_min=None), waiting_episodes(points)
    )
    valid_ships = [200000001, 200000004]
    tr = feats.filter(~pl.col("mmsi").is_in(valid_ships))
    va = feats.filter(pl.col("mmsi").is_in(valid_ships))
    model = DecomposedETA.fit(tr, va, BASE_FEATURES).calibrate(va)
    path = artifact.save(model, tmp_path_factory.mktemp("model"), {"version": "test"})
    return model, path, feats, points, voyages


def test_artifact_round_trip(trained):
    model, path, feats, *_ = trained
    loaded, meta = artifact.load(path)
    a, b = model.predict(feats), loaded.predict(feats)
    for c in ("eta_p10", "eta_p50", "eta_p90"):
        np.testing.assert_allclose(a[c].to_numpy(), b[c].to_numpy(), rtol=1e-6)
    assert meta["version"] == "test" and loaded.widen == model.widen


@pytest.mark.parametrize("cut_min", [20, 90, 200])
def test_serving_features_equal_training_features(trained, cut_min):
    """No train/serve skew: same points in, same feature row out."""
    _, _, _, points, voyages = trained
    vid = voyages.filter(pl.col("mmsi") == 200000001)["voyage_id"][0]
    pts = points.filter(pl.col("voyage_id") == vid).sort("ts")
    cutoff = pts["ts"][0] + dt.timedelta(minutes=cut_min)
    upto = pts.filter(pl.col("ts") <= cutoff)
    empty_state = pl.DataFrame(schema={"bucket": pl.Datetime("ms", "UTC"), "port_waiting": pl.Int64,
                                       "port_inbound": pl.Int64, "port_in_port": pl.Int64})  # fmt: skip
    train_row = build_features(upto, voyages, empty_state, PORT2, every_min=None).tail(1)

    raw = upto.select("ts", "lat", "lon", "sog", "cog", "heading", "status")
    static = Static(mmsi=200000001, vessel_type=70, length=200.0, width=30.0, draft=10.0)
    table, _ = request_table(raw, static, PORT2)
    serve_row = request_features(table, PORT2)
    for f in BASE_FEATURES:
        a = train_row[f].cast(pl.Float64).to_numpy()
        b = serve_row[f].cast(pl.Float64).to_numpy()
        np.testing.assert_allclose(a, b, rtol=1e-5, equal_nan=True, err_msg=f)


def test_ship_in_port_is_refused():
    raw = pl.DataFrame({"ts": [T0, T0 + dt.timedelta(minutes=1)], "lat": [33.69, INSIDE],
                        "lon": [GATE[0]] * 2, "sog": [5.0, 0.0], "cog": [0.0, 0.0],
                        "heading": [None, None], "status": [0, 5]})  # fmt: skip
    with pytest.raises(AlreadyInPort):
        request_table(raw, Static(mmsi=200000001, vessel_type=70), PORT2)


@pytest.fixture(scope="module")
def client(trained):
    _, path, *_ = trained
    with TestClient(create_app(path)) as c:
        yield c


def _request(rows, mmsi=200000001):
    return {
        "ship": {"mmsi": mmsi, "vessel_type": 70, "length": 200, "draft": 10},
        "track": [
            {"ts": (T0 + dt.timedelta(minutes=m)).isoformat(), "lat": lat, "lon": lon,
             "sog": sog, "cog": 0.0, "status": 0}
            for m, lat, lon, sog in rows
        ],
    }  # fmt: skip


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_version"] == "test"


def test_eta_for_a_ship_under_way(client):
    r = client.post("/v1/eta", json=_request(approach(0, 90, 33.20)))
    assert r.status_code == 200, r.text
    body = r.json()
    q = body["eta_minutes"]
    assert 0 <= q["p10"] <= q["p50"] <= q["p90"]
    assert body["waiting_now"] is False and 0 <= body["p_wait_before_arrival"] <= 1
    assert body["eta"]["p50"] >= body["as_of"]
    assert body["points_used"] == body["points_received"] == 90


def test_eta_for_a_waiting_ship(client):
    r = client.post("/v1/eta", json=_request(approach(0, 30, 33.40) + stay(30, 120, 33.48)))
    assert r.status_code == 200, r.text
    assert r.json()["waiting_now"] is True and r.json()["p_wait_before_arrival"] is None


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda b: b["track"].reverse(), "time order"),
        (lambda b: b["ship"].update(mmsi=993692028), "mmsi"),
        (lambda b: b["track"][-1].update(sog=None), "sog"),
        (lambda b: b["track"][0].update(lat=95), "lat"),
        (lambda b: b["ship"].update(vessel_type=37), "vessel_type"),
    ],
)
def test_bad_requests_are_rejected(client, mutate, fragment):
    body = _request(approach(0, 30, 33.40))
    mutate(body)
    r = client.post("/v1/eta", json=body)
    assert r.status_code == 422 and fragment in r.text


def test_arrived_ship_gets_a_clear_error(client):
    # steams in through the gate (the app uses the real LA/LB polygon) and stops inside
    rows = approach(0, 45, 33.61)
    rows += stay(45, 5, rows[-1][1])
    r = client.post("/v1/eta", json=_request(rows))
    assert r.status_code == 422 and "arrived" in r.text


def test_teleport_into_port_is_treated_as_a_position_spike(client):
    # a jump of 30 km in a minute is not physical: cleaning drops it, the ship is still out
    r = client.post("/v1/eta", json=_request(approach(0, 30, 33.40) + stay(30, 5, 33.75)))
    assert r.status_code == 200 and r.json()["points_used"] == 30


def test_metrics_count_requests_and_predictions(client):
    client.post("/v1/eta", json=_request(approach(0, 90, 33.20)))
    client.post("/v1/eta", json={"bad": "request"})
    text = client.get("/metrics/").text
    assert 'vessel_eta_requests_total{endpoint="/v1/eta",status="200"}' in text
    assert 'vessel_eta_requests_total{endpoint="/v1/eta",status="422"}' in text
    assert "vessel_eta_predicted_eta_hours_count" in text
    assert 'vessel_eta_model_info{trained_until="",version="test"} 1.0' in text


def test_monthly_report_end_to_end(trained):
    from vessel_eta.modeling import baseline_predictions
    from vessel_eta.monitoring import evaluate_month, reference_profile

    model, _, feats, _, voyages = trained
    rows = feats.join(voyages.select("voyage_id", "arrival_ts"), on="voyage_id")
    rows = baseline_predictions(rows, rows)
    ref = reference_profile(rows, model.predict(rows)["eta_p50"].to_numpy())
    rep = evaluate_month(model, ref, rows, "2023-01")
    assert rep.status in {"OK", "WARN", "ALERT"}
    assert 0 <= rep.quality["coverage"] <= 1 and rep.drift["sog"] < 0.05  # same data: no drift
    assert rep.new_widen is not None and "Monitoring report 2023-01" in rep.markdown()
