"""Travel-time part (T) of the decomposed ETA: LightGBM vs MLP vs GRU.

Same split as the full-year study (train arrivals Jan-Sep 2023, early stopping on
held-out ships, test Oct-Dec), same rows (ships under way), same target
(minutes to the next event: start of a wait, or arrival), same 7 quantiles.

uv run python scripts/travel_models.py
"""

from __future__ import annotations

import datetime as dt
import json
import time

import numpy as np
import polars as pl

from vessel_eta.config import PROCESSED_DIR, ROOT
from vessel_eta.decompose import QUANTILES, QuantileModel, add_targets, waiting_episodes
from vessel_eta.experiment import _table, load_features
from vessel_eta.features import BASE_FEATURES
from vessel_eta.geo import load_port
from vessel_eta.modeling import make_split
from vessel_eta.sequence import SequenceTravelModel

TEST_START = dt.datetime(2023, 10, 1, tzinfo=dt.UTC)
TEST_END = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
TARGET = "t_next_min"


def evaluate(rows: pl.DataFrame, q_log: np.ndarray) -> dict[str, float]:
    y = np.log1p(rows[TARGET].to_numpy())
    d = y[:, None] - q_log
    pinball = float(np.mean(np.maximum(QUANTILES * d, (QUANTILES - 1) * d)))
    med = np.expm1(q_log[:, 3])
    p10, p90 = np.expm1(q_log[:, 1]), np.expm1(q_log[:, 5])
    t = rows[TARGET].to_numpy()
    near = (rows["gate_km"] < 30).to_numpy()
    ae = np.abs(med - t) / 60
    return {
        "pinball_log": pinball,
        "mae_h": float(ae.mean()),
        "median_ae_h": float(np.median(ae)),
        "mae_h_near": float(ae[near].mean()),
        "mae_h_far": float(ae[~near].mean()),
        "coverage_p10_p90": float(np.mean((t >= p10) & (t <= p90))),
    }


def main() -> None:
    port = load_port()
    feats, voyages = load_features(port)
    points = pl.read_parquet(PROCESSED_DIR / f"eta_points_{port.name}.parquet")
    feats = add_targets(feats, waiting_episodes(points))
    split = make_split(feats, voyages, TEST_START, test_end=TEST_END)
    moving = ~pl.col("waiting_now")
    tr, va, te = (d.filter(moving) for d in (split.train, split.valid, split.test))
    print(f"moving rows: train {tr.height:,}  valid {va.height:,}  test {te.height:,}", flush=True)

    rows = []
    t = time.time()
    lgbm = QuantileModel.fit(tr, va, TARGET, BASE_FEATURES)
    rows.append({"model": "LightGBM (tabular)", "fit_s": time.time() - t,
                 **evaluate(te, lgbm.predict_log(te))})  # fmt: skip
    print(rows[-1], flush=True)

    histories, valid_q, test_q = {}, {}, {}
    valid_q["lgbm"], test_q["lgbm"] = lgbm.predict_log(va), lgbm.predict_log(te)
    variants = [
        ("MLP (tabular)", dict(use_sequence=False)),
        ("GRU (2 h track) + tabular", dict(use_sequence=True)),
        ("GRU + tabular, regularised", dict(use_sequence=True, dropout=0.2, weight_decay=1e-4,
                                            lr=5e-4, patience=4)),
    ]  # fmt: skip
    for name, kw in variants:
        t = time.time()
        m = SequenceTravelModel(BASE_FEATURES, **kw).fit(split.train, tr, split.valid, va, TARGET)
        valid_q[name], test_q[name] = (
            m.predict_log(va, full=split.valid),
            m.predict_log(te, full=split.test),
        )
        rows.append({"model": name, "fit_s": time.time() - t,
                     "valid_pinball_log": evaluate(va, valid_q[name])["pinball_log"],
                     **evaluate(te, test_q[name])})  # fmt: skip
        histories[name] = m.history
        print(rows[-1], flush=True)
    rows[0]["valid_pinball_log"] = evaluate(va, valid_q["lgbm"])["pinball_log"]

    # ensemble: average LightGBM and the best GRU (chosen on valid) in log space,
    # weight chosen on the held-out ships of the training period, then test once
    gru_names = [n for n, _ in variants if n.startswith("GRU")]
    best_gru = min(gru_names, key=lambda n: evaluate(va, valid_q[n])["pinball_log"])
    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    scores = {w: evaluate(va, w * valid_q[best_gru] + (1 - w) * valid_q["lgbm"])["pinball_log"]
              for w in weights}  # fmt: skip
    w = min(scores, key=scores.get)
    rows.append({"model": f"LightGBM + {best_gru} (w={w})", "fit_s": float("nan"),
                 "valid_pinball_log": scores[w],
                 **evaluate(te, w * test_q[best_gru] + (1 - w) * test_q["lgbm"])})  # fmt: skip
    print("ensemble weights on valid:", scores, flush=True)
    print(rows[-1], flush=True)

    table = _table(pl.DataFrame(rows))
    out = ROOT / "reports" / "travel_models.md"
    out.write_text(
        "# Travel time to the next event (moving ships), test Oct-Dec 2023\n\n"
        "pinball on log1p(minutes) over 7 quantiles; MAE of the median in hours; "
        "near = within 30 km of a gate.\n\n" + table + "\n"
    )
    (ROOT / "reports" / "travel_models.json").write_text(
        json.dumps({"results": rows, "histories": histories}, indent=2)
    )
    print(table)


if __name__ == "__main__":
    main()
