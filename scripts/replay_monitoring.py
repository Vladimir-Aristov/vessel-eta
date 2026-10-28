"""Replay a year of operations: a model trained on 2023, monitored month by month in 2024.

Every month: score the closed month (quality, PSI drift, gates), write the report, then
re-calibrate the interval shifts for the next month — exactly what
`vessel-eta monitor --month YYYY-MM --apply` does on the shipped artifact. The model's
trees are never refit, so the replay also shows when retraining would have been due.

uv run python scripts/replay_monitoring.py [--method window|tracking]
"""

from __future__ import annotations

import argparse
import datetime as dt

import polars as pl

from vessel_eta import artifact
from vessel_eta.config import ROOT
from vessel_eta.experiment import _table
from vessel_eta.geo import load_port
from vessel_eta.monitoring import MONITORED, evaluate_month, widen_after
from vessel_eta.production import _labelled_features, month_rows, train_production

OUT = ROOT / "reports" / "monitoring" / "replay_2024"
MODEL_DIR = ROOT / "artifacts" / "replay_2023"


def main(method: str) -> None:
    port = load_port()
    meta_path = MODEL_DIR / "meta.json"
    if not meta_path.exists() or "normal_monthly_psi" not in meta_path.read_text():
        train_production(
            port, MODEL_DIR, version="v2-replay", train_end=dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
        )
    model, meta = artifact.load(MODEL_DIR)
    feats, voyages = _labelled_features(port)
    OUT.mkdir(parents=True, exist_ok=True)

    summary = []
    for m in range(1, 13):
        month = f"2024-{m:02d}"
        rep = evaluate_month(
            model, meta["reference"], month_rows(feats, voyages, month), month, method
        )
        (OUT / f"{month}.md").write_text(rep.markdown())
        inputs = {k: v for k, v in rep.drift.items() if k in MONITORED}
        worst = max(inputs, key=lambda k: inputs[k])
        summary.append({
            "month": month,
            "voyages": rep.voyages,
            "model MAE h": rep.quality["model_mae_h"],
            "physics MAE h": rep.quality["naive_mae_h"],
            "coverage": rep.quality["coverage"],
            "worst input PSI": f"{worst} {inputs[worst]:.2f}",
            "prediction PSI": rep.drift["prediction_log_eta"],
            "status": rep.status,
            "drifted inputs": ", ".join(
                a.removeprefix("watch drift: ") for a in rep.actions if a.startswith("watch")
            ) or "-",
        })  # fmt: skip
        model.widen = widen_after(rep)  # what --apply writes, used from next month on
        print(summary[-1], flush=True)

    table = _table(pl.DataFrame(summary))
    (OUT.parent / "replay_2024.md").write_text(
        "# A year of monitoring, replayed (model trained on 2023, never refit)\n\n"
        f"Interval re-calibration each month: {method}. Per-month reports: replay_2024/.\n\n"
        + table
        + "\n"
    )
    print(table)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="window", choices=["window", "tracking"])
    main(ap.parse_args().method)
