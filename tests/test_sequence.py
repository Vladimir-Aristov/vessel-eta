import datetime as dt

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")  # optional extra: uv sync --extra nn

from vessel_eta.features import BASE_FEATURES  # noqa: E402
from vessel_eta.sequence import STEP_COLS, SequenceTravelModel, step_table, window_index  # noqa: E402


def test_window_stays_inside_voyage_and_looks_back_only():
    vids = np.array(["a", "a", "a", "b", "b", "b", "b"])
    rows = np.array([0, 2, 3, 6])
    idx, mask = window_index(vids, rows, window=3)
    # row 0: only itself
    assert mask[0].tolist() == [False, False, True] and idx[0, -1] == 0
    # row 2: rows 0,1,2 of voyage a
    assert idx[1].tolist() == [0, 1, 2] and mask[1].all()
    # row 3: first row of voyage b -> nothing from voyage a
    assert mask[2].tolist() == [False, False, True]
    # row 6: rows 4,5,6 (all voyage b, all in the past)
    assert idx[3].tolist() == [4, 5, 6] and mask[3].all()
    assert (np.where(mask, idx, -1) <= rows[:, None]).all()


def _toy_table(n_voyages=40, steps=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    t0 = dt.datetime(2023, 1, 1, tzinfo=dt.UTC)
    for v in range(n_voyages):
        speed = rng.uniform(6, 14)
        for i in range(steps):
            remaining = (steps - i) * 5
            row = {f: 0.0 for f in BASE_FEATURES}
            row.update(
                voyage_id=f"v{v}", mmsi=v, ts=t0 + dt.timedelta(minutes=5 * i),
                lat=33.0 + i * 0.01, lon=-118.2, sog=speed, cog_sin=0.0, cog_cos=1.0,
                stationary=False, gate_km=remaining * speed * 0.03, vessel_type=70, status=0,
                t_next_min=float(remaining),
            )  # fmt: skip
            rows.append(row)
    return pl.DataFrame(rows)


def test_step_table_columns():
    df = step_table(_toy_table(2, 5).sort("voyage_id", "ts"))
    assert all(c in df.columns for c in STEP_COLS)
    assert df["_dt"][0] == 0  # first row of a voyage has no gap


def test_model_learns_and_quantiles_are_monotone():
    tr, va = _toy_table(40, 30, 0), _toy_table(10, 30, 1).with_columns(pl.col("voyage_id") + "x")
    m = SequenceTravelModel(BASE_FEATURES, use_sequence=True, epochs=8, batch_size=256, lr=3e-3)
    m.fit(tr, tr, va, va, "t_next_min")
    q = m.predict_log(va, full=va)
    assert q.shape == (va.height, 7)
    assert np.all(np.diff(q, axis=1) >= 0)
    med = np.expm1(q[:, 3])
    mae = np.mean(np.abs(med - va["t_next_min"].to_numpy()))
    assert mae < 0.5 * np.mean(np.abs(va["t_next_min"].to_numpy() - va["t_next_min"].mean()))
    assert m.history[-1]["valid_pinball_log"] <= m.history[0]["valid_pinball_log"]
