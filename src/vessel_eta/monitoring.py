"""Monthly model monitoring: quality, input drift, gates, re-calibration.

For a closed month with known outcomes:

1. quality  — MAE vs the distance/speed baseline, P10-P90 coverage, pinball;
2. drift    — PSI of key features and of the predictions against the training
              reference stored in the artifact (`reference_profile`);
3. gates    — quality: the model must beat physics and coverage stay in [0.70, 0.90];
              drift: no input beyond max(0.25, its own normal month-to-month PSI);
4. action   — re-calibrate the interval shifts from the month's outcomes (always
              safe: only the shifts change); recommend retraining when quality or
              drift gates fail. Default: conformal shifts from the month's arrivals
              ("window"); daily quantile tracking ("tracking") lost to it on 2024
              (reports/adaptive_calibration.md).

PSI: sum over decile bins of (cur - ref) * ln(cur / ref); < 0.1 stable,
0.1-0.25 moderate shift, > 0.25 large shift (the usual credit-risk convention).

That convention assumes a month should look like the whole reference. For slowly
varying inputs it does not: the port queue sits at one level for weeks, so a single
month of the *training* data itself scores PSI up to 2.0 against the training year.
Each feature's alert threshold is therefore max(0.25, the 95th percentile of its
monthly PSI inside the training data), stored with the reference.

Status: ALERT when a quality gate fails, WARN when only drift gates fail, else OK.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from vessel_eta.calibration import track_shifts
from vessel_eta.decompose import DecomposedETA, conformal_widen
from vessel_eta.modeling import metrics
from vessel_eta.scalars import as_float

MONITORED = [
    "sog", "gate_km", "draft", "length", "closing_kmh_1h", "stationary_share_6h",
    "min_since_start", "port_waiting", "port_inbound",
]  # fmt: skip
PSI_WARN, PSI_ALERT = 0.1, 0.25
COVERAGE_OK = (0.70, 0.90)
EPS = 1e-4


def _edges(values: np.ndarray) -> list[float]:
    qs = np.quantile(values, np.linspace(0, 1, 11)[1:-1])
    return sorted(set(np.round(qs, 6).tolist()))


def _shares(values: np.ndarray, edges: list[float]) -> list[float]:
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    total = int(counts.sum()) or 1
    return (counts / total).tolist()


def reference_profile(df: pl.DataFrame, pred_p50_min: np.ndarray) -> dict:
    """Decile bins of monitored features and of log predicted ETA, from training rows.

    With an `arrival_ts` column, also each feature's normal month-to-month PSI inside
    the training data (95th percentile), used as its drift threshold.
    """
    prof: dict[str, dict] = {}
    months = (
        df.with_columns(_m=pl.col("arrival_ts").dt.strftime("%Y-%m")).partition_by("_m")
        if "arrival_ts" in df.columns
        else []
    )
    for c in MONITORED:
        v = df[c].drop_nulls().cast(pl.Float64).to_numpy()
        e = _edges(v)
        prof[c] = {
            "edges": e,
            "shares": _shares(v, e),
            "null_share": df[c].null_count() / df.height,
        }
        monthly = [psi(prof[c], m[c].drop_nulls().cast(pl.Float64).to_numpy()) for m in months]
        prof[c]["normal_monthly_psi"] = float(np.quantile(monthly, 0.95)) if monthly else 0.0
    lp = np.log1p(pred_p50_min)
    e = _edges(lp)
    prof["prediction_log_eta"] = {"edges": e, "shares": _shares(lp, e), "null_share": 0.0}
    return prof


def psi(ref: dict, values: np.ndarray) -> float:
    cur = np.asarray(_shares(values, ref["edges"])) + EPS
    base = np.asarray(ref["shares"]) + EPS
    return float(np.sum((cur - base) * np.log(cur / base)))


@dataclass
class MonthReport:
    month: str
    voyages: int
    rows: int
    quality: dict
    coverage_by_branch: dict
    drift: dict
    gates: dict
    status: str
    actions: list[str] = field(default_factory=list)
    new_widen: dict | None = None
    limits: dict = field(default_factory=dict)  # per-feature drift thresholds

    def markdown(self) -> str:
        q, g = self.quality, self.gates
        drift_sorted = sorted(self.drift.items(), key=lambda kv: -kv[1])
        lines = [
            f"# Monitoring report {self.month}: **{self.status}**",
            "",
            f"{self.voyages} voyages arrived, {self.rows:,} predictions scored.",
            "",
            "| quality | value |",
            "|---|---|",
            f"| model MAE (h) | {q['model_mae_h']:.1f} |",
            f"| distance/speed MAE (h) | {q['naive_mae_h']:.1f} |",
            f"| P10-P90 coverage (target 0.80) | {q['coverage']:.3f} |",
            f"| — under way / waiting | {self.coverage_by_branch['under_way']:.3f} / "
            f"{self.coverage_by_branch['waiting']:.3f} |",
            f"| pinball (h) | {q['pinball_h']:.2f} |",
            "",
            "| input / output drift | PSI | its alert limit | over the limit |",
            "|---|---|---|---|",
            *[self._drift_row(k, v) for k, v in drift_sorted],
            "",
            "| gate | passed |",
            "|---|---|",
            *[f"| {k} | {'yes' if v else '**no**'} |" for k, v in g.items()],
            "",
            "Actions: " + ("; ".join(self.actions) if self.actions else "none"),
            "",
        ]  # fmt: skip
        return "\n".join(lines)

    def _drift_row(self, name: str, value: float) -> str:
        if name not in self.limits:
            return f"| {name} | {value:.3f} | — (watched, not gated) | — |"
        limit = self.limits[name]
        over = "**yes**" if value > limit else "no"
        return f"| {name} | {value:.3f} | {limit:.2f} | {over} |"


def drift_limit(ref: dict) -> float:
    """Alert threshold for one feature: max(0.25, its normal monthly PSI in training)."""
    return max(PSI_ALERT, float(ref.get("normal_monthly_psi", 0.0)))


def evaluate_month(
    model: DecomposedETA,
    reference: dict,
    month_rows: pl.DataFrame,
    month: str,
    recalibration: str = "window",
    step: float = 0.01,
) -> MonthReport:
    """month_rows: feature rows of voyages that arrived in the month, with eta_min,
    arrival_ts, waiting_now and pred_naive (distance/speed baseline)."""
    pred = model.predict(month_rows)
    cov = lambda d: as_float(  # noqa: E731
        ((d["eta_min"] >= d["eta_p10"]) & (d["eta_min"] <= d["eta_p90"])).mean()
    )
    y = pred["eta_min"].to_numpy() / 60
    qs = {0.1: pred["eta_p10"], 0.5: pred["eta_p50"], 0.9: pred["eta_p90"]}
    pinball = np.mean([
        np.mean(np.maximum(a * (y - p.to_numpy() / 60), (a - 1) * (y - p.to_numpy() / 60)))
        for a, p in qs.items()
    ])  # fmt: skip
    quality = {
        "model_mae_h": metrics(pred.with_columns(pl.col("eta_p50").alias("_m")), "_m")["mae_h"],
        "naive_mae_h": metrics(pred, "pred_naive")["mae_h"],
        "coverage": cov(pred),
        "pinball_h": float(pinball),
    }
    by_branch = {
        "under_way": cov(pred.filter(~pl.col("waiting_now"))),
        "waiting": cov(pred.filter(pl.col("waiting_now"))),
    }
    drift = {c: psi(reference[c], month_rows[c].drop_nulls().cast(pl.Float64).to_numpy())
             for c in MONITORED}  # fmt: skip
    drift["prediction_log_eta"] = psi(
        reference["prediction_log_eta"], np.log1p(pred["eta_p50"].to_numpy())
    )
    limits = {c: drift_limit(reference[c]) for c in MONITORED}
    drifted = [c for c in MONITORED if drift[c] > limits[c]]
    quality_gates = {
        "beats distance/speed": quality["model_mae_h"] <= quality["naive_mae_h"],
        f"coverage in {COVERAGE_OK}": COVERAGE_OK[0] <= quality["coverage"] <= COVERAGE_OK[1],
    }
    gates = {**quality_gates, "inputs within their normal drift": not drifted}
    actions: list[str] = []
    if not all(quality_gates.values()):
        actions.append("retrain recommended")
    if drifted:
        drift_note = ", ".join(f"{c} {drift[c]:.2f} > {limits[c]:.2f}" for c in drifted)
        actions.append(f"watch drift: {drift_note}")
    # re-calibrate the shifts from this month's outcomes, for next month
    current = dict(model.widen)
    model.widen = {True: (0.0, 0.0), False: (0.0, 0.0)}
    raw = model.predict(month_rows)
    model.widen = current
    if recalibration == "tracking":
        _, hist = track_shifts(raw, current, step)
        last = hist.row(-1, named=True)
        new = {True: (last["waiting_lo"], last["waiting_hi"]),
               False: (last["moving_lo"], last["moving_hi"])}  # fmt: skip
    else:
        new = conformal_widen(raw, fallback=current)
    actions.append(f"interval shifts re-estimated ({recalibration})")
    status = "ALERT" if not all(quality_gates.values()) else "WARN" if drifted else "OK"
    return MonthReport(
        month=month,
        voyages=month_rows["voyage_id"].n_unique(),
        rows=month_rows.height,
        quality=quality,
        coverage_by_branch=by_branch,
        drift=drift,
        gates=gates,
        status=status,
        actions=actions,
        new_widen={"waiting": list(new[True]), "moving": list(new[False])},
        limits=limits,
    )


def widen_after(report: MonthReport) -> dict[bool, tuple[float, float]]:
    assert report.new_widen is not None
    (wl, wh), (ml, mh) = report.new_widen["waiting"], report.new_widen["moving"]
    return {True: (wl, wh), False: (ml, mh)}


def apply_month(model: DecomposedETA, report: MonthReport) -> None:
    model.widen = widen_after(report)
