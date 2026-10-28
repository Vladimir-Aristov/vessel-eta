"""Decomposed, probabilistic ETA.

v1 showed the error is the queue: a ship either goes straight in or waits
(at anchor, drifting offshore, or at another port in the region) for hours to days,
and one regressor over that mixture predicts neither well. So:

    moving ship:  ETA = T + B * W
        T  time to the next event (start of a waiting episode, or arrival)   quantile model
        B  will there be a waiting episode before arrival?                    classifier
        W  time from the start of that episode to arrival                     quantile model
    waiting ship: ETA = A                                                      quantile model

Which branch answers is decided from what is visible now (`waiting_now`: mean SOG over
the last 30 min below 2 kn). The ETA distribution of a moving ship is the mixture,
built by sampling; the point forecast is its median.

Waiting episodes (labels only, may use the future): runs where the centred 30-min median SOG
is below 2 kn for at least 1 h, merged when less than 6 h apart, kept if >= 2 h.
On Jan-Mar 2023 that gives 344 episodes on 287 of 880 voyages; median 27 h, p90 147 h.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import polars as pl

from vessel_eta.features import CATEGORICAL, FEATURES
from vessel_eta.modeling import EARLY_STOPPING, LGB_PARAMS, MAX_ROUNDS

STOP_SOG_KN = 2.0
MIN_RUN_H = 1.0
MERGE_GAP_H = 6.0
MIN_EPISODE_H = 2.0
WAITING_NOW_KN = 2.0  # routing threshold on the trailing 30-min mean SOG

QUANTILES = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
OUT_Q = (0.1, 0.5, 0.9)


# ------------------------------------------------------------------ labels


def waiting_episodes(points: pl.DataFrame) -> pl.DataFrame:
    """voyage_id, ep_start, ep_end for every waiting episode (full-resolution points)."""
    pts = points.sort("voyage_id", "ts").select("voyage_id", "ts", "sog")
    trailing = pts.with_columns(
        sog_med=pl.col("sog").fill_null(0).rolling_median_by("ts", "30m").over("voyage_id")
    ).select("voyage_id", pl.col("ts").alias("_t"), "sog_med")
    # centred 30-min median (labels may look ahead): the trailing median 15 min later
    df = (
        pts.with_columns(_t_look=pl.col("ts") + pl.duration(minutes=15))
        .sort("_t_look")
        .join_asof(
            trailing.sort("_t"),
            left_on="_t_look",
            right_on="_t",
            by="voyage_id",
            strategy="backward",
            check_sortedness=False,
        )  # fmt: skip
        .sort("voyage_id", "ts")
        .with_columns(slow=pl.col("sog_med") < STOP_SOG_KN)
        .with_columns(run=pl.col("slow").rle_id().over("voyage_id"))
    )
    runs = (
        df.filter("slow")
        .group_by("voyage_id", "run")
        .agg(pl.col("ts").min().alias("t0"), pl.col("ts").max().alias("t1"))
        .filter((pl.col("t1") - pl.col("t0")) >= pl.duration(hours=MIN_RUN_H))
        .sort("voyage_id", "t0")
    )
    gap = pl.col("t0") - pl.col("t1").shift().over("voyage_id")
    return (
        runs.with_columns(
            ep=(gap > pl.duration(hours=MERGE_GAP_H)).fill_null(True).cum_sum().over("voyage_id")
        )
        .group_by("voyage_id", "ep")
        .agg(pl.col("t0").min().alias("ep_start"), pl.col("t1").max().alias("ep_end"))
        .filter((pl.col("ep_end") - pl.col("ep_start")) >= pl.duration(hours=MIN_EPISODE_H))
        .select("voyage_id", "ep_start", "ep_end")
        .sort("voyage_id", "ep_start")
    )


def add_targets(feats: pl.DataFrame, episodes: pl.DataFrame) -> pl.DataFrame:
    """Add waiting_now (routing, causal) and the decomposed targets (labels).

    stop_next      a waiting episode that has not ended yet lies ahead
    t_next_min     minutes to its start (0 if it has started), else to arrival
    after_stop_min minutes from its start to arrival (only when stop_next)
    """
    eps = episodes.sort("ep_end")
    out = (
        feats.sort("ts")
        .join_asof(
            eps,
            left_on="ts",
            right_on="ep_end",
            by="voyage_id",
            strategy="forward",
            check_sortedness=False,
        )  # fmt: skip
        .sort("voyage_id", "ts")
        .with_columns(
            waiting_now=pl.col("sog_mean_30m") < WAITING_NOW_KN,
            stop_next=pl.col("ep_end").is_not_null(),
            _to_start=((pl.col("ep_start") - pl.col("ts")).dt.total_seconds() / 60).clip(
                lower_bound=0
            ),
        )
        .with_columns(
            t_next_min=pl.when("stop_next").then("_to_start").otherwise("eta_min").cast(pl.Float64)
        )
        .with_columns(
            after_stop_min=pl.when("stop_next").then(pl.col("eta_min") - pl.col("t_next_min"))
        )
        .drop("_to_start", "ep_start", "ep_end")
    )
    return out


# ------------------------------------------------------------------ models


def _x(df: pl.DataFrame) -> np.ndarray:
    return df.select(pl.col(FEATURES).cast(pl.Float64)).to_numpy()


_CAT = [FEATURES.index(c) for c in CATEGORICAL]


def _train(params, xtr, ytr, xva, yva) -> lgb.Booster:
    dtr = lgb.Dataset(xtr, ytr, feature_name=FEATURES, categorical_feature=_CAT)
    dva = lgb.Dataset(xva, yva, reference=dtr)
    return lgb.train(
        params, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dva],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )  # fmt: skip


@dataclass
class QuantileModel:
    """One LightGBM per quantile on log1p(minutes); predictions sorted (no crossing)."""

    boosters: list[lgb.Booster]

    @classmethod
    def fit(cls, train: pl.DataFrame, valid: pl.DataFrame, target: str) -> QuantileModel:
        xtr, xva = _x(train), _x(valid)
        ytr = np.log1p(train[target].to_numpy().astype(np.float64))
        yva = np.log1p(valid[target].to_numpy().astype(np.float64))
        boosters = []
        for q in QUANTILES:
            params = {
                **LGB_PARAMS,
                "objective": "quantile",
                "alpha": float(q),
                "metric": "quantile",
            }
            boosters.append(_train(params, xtr, ytr, xva, yva))
        return cls(boosters)

    def predict_log(self, df: pl.DataFrame) -> np.ndarray:
        x = _x(df)
        p = np.stack([b.predict(x, num_iteration=b.best_iteration) for b in self.boosters], axis=1)
        return np.sort(p, axis=1)


@dataclass
class StopClassifier:
    booster: lgb.Booster

    @classmethod
    def fit(cls, train: pl.DataFrame, valid: pl.DataFrame) -> StopClassifier:
        params = {**LGB_PARAMS, "objective": "binary", "metric": "binary_logloss"}
        y = lambda d: d["stop_next"].cast(pl.Int8).to_numpy()  # noqa: E731
        return cls(_train(params, _x(train), y(train), _x(valid), y(valid)))

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return self.booster.predict(_x(df), num_iteration=self.booster.best_iteration)


# ------------------------------------------------------------------ composition


def sample_from_quantiles(q_log: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse-CDF sampling, piecewise linear in log space between QUANTILES knots.

    Beyond the outer knots the value is held flat, so tails are slightly too thin.
    q_log: (n_rows, n_knots); u: (n_samples,) shared across rows -> (n_rows, n_samples) minutes.
    """
    u = np.clip(u, QUANTILES[0], QUANTILES[-1])
    idx = np.clip(np.searchsorted(QUANTILES, u, side="right") - 1, 0, len(QUANTILES) - 2)
    w = (u - QUANTILES[idx]) / (QUANTILES[idx + 1] - QUANTILES[idx])
    lo, hi = q_log[:, idx], q_log[:, idx + 1]
    return np.expm1(lo + w * (hi - lo)).clip(min=0)


def compose_moving(
    p_stop: np.ndarray, t_q: np.ndarray, w_q: np.ndarray, n_samples: int = 400, seed: int = 0
) -> np.ndarray:
    """Quantiles OUT_Q of T + B*W, with T, B, W independent given the features."""
    rng = np.random.default_rng(seed)
    out = np.empty((len(p_stop), len(OUT_Q)))
    for s in range(0, len(p_stop), 20_000):  # chunk rows to bound memory
        sl = slice(s, s + 20_000)
        t = sample_from_quantiles(t_q[sl], rng.random(n_samples))
        w = sample_from_quantiles(w_q[sl], rng.random(n_samples))
        b = rng.random((t.shape[0], n_samples)) < p_stop[sl, None]
        out[sl] = np.quantile(t + b * w, OUT_Q, axis=1).T
    return out


@dataclass
class DecomposedETA:
    stop: StopClassifier
    travel: QuantileModel  # T
    after_stop: QuantileModel  # W
    waiting: QuantileModel  # A
    # conformal widening in log space per branch: {waiting_now: (lower, upper)}
    widen: dict[bool, tuple[float, float]] = field(
        default_factory=lambda: {True: (0, 0), False: (0, 0)}
    )

    @classmethod
    def fit(cls, train: pl.DataFrame, valid: pl.DataFrame) -> DecomposedETA:
        mv_tr, mv_va = train.filter(~pl.col("waiting_now")), valid.filter(~pl.col("waiting_now"))
        st_tr, st_va = mv_tr.filter("stop_next"), mv_va.filter("stop_next")
        wt_tr, wt_va = train.filter("waiting_now"), valid.filter("waiting_now")
        return cls(
            stop=StopClassifier.fit(mv_tr, mv_va),
            travel=QuantileModel.fit(mv_tr, mv_va, "t_next_min"),
            after_stop=QuantileModel.fit(st_tr, st_va, "after_stop_min"),
            waiting=QuantileModel.fit(wt_tr, wt_va, "eta_min"),
        )

    def calibrate(self, calib: pl.DataFrame) -> DecomposedETA:
        """Conformalised quantiles (CQR, asymmetric, in log space, per branch).

        On ships not used for fitting, find how far P10 / P90 must move so that 10% of
        outcomes fall below / above them. Quantile regression with heavy regularisation
        and flat tails under-covers otherwise (76% in-period CV for a nominal 80%).
        """
        raw = self.widen
        self.widen = {True: (0, 0), False: (0, 0)}
        pred = self.predict(calib)
        widen = {}
        for waiting in (True, False):
            part = pred.filter(pl.col("waiting_now") == waiting)
            if part.height < 50:
                widen[waiting] = raw[waiting]
                continue
            y = np.log1p(part["eta_min"].to_numpy())
            lo = np.log1p(part["eta_p10"].to_numpy())
            hi = np.log1p(part["eta_p90"].to_numpy())
            widen[waiting] = (float(np.quantile(lo - y, 0.9)), float(np.quantile(y - hi, 0.9)))
        self.widen = widen
        return self

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Adds p_stop (moving rows), eta_p10 / eta_p50 / eta_p90 (minutes)."""
        df = df.with_row_index("_i")
        moving = df.filter(~pl.col("waiting_now"))
        waiting = df.filter("waiting_now")
        q = np.empty((df.height, len(OUT_Q)))
        p = np.full(df.height, np.nan)
        if moving.height:
            pm = self.stop.predict(moving)
            idx = moving["_i"].to_numpy()
            p[idx] = pm
            q[idx] = compose_moving(
                pm, self.travel.predict_log(moving), self.after_stop.predict_log(moving)
            )
        if waiting.height:
            knots = [int(np.flatnonzero(np.isclose(QUANTILES, a))[0]) for a in OUT_Q]
            q[waiting["_i"].to_numpy()] = np.expm1(self.waiting.predict_log(waiting)[:, knots])
        is_waiting = df["waiting_now"].to_numpy()
        d_lo = np.where(is_waiting, self.widen[True][0], self.widen[False][0])
        d_hi = np.where(is_waiting, self.widen[True][1], self.widen[False][1])
        q[:, 0] = np.expm1(np.log1p(q[:, 0]) - d_lo).clip(min=0)
        q[:, 2] = np.expm1(np.log1p(q[:, 2]) + d_hi)
        q[:, 0] = np.minimum(q[:, 0], q[:, 1])  # a negative widening must not cross the median
        q[:, 2] = np.maximum(q[:, 2], q[:, 1])
        return df.drop("_i").with_columns(
            p_stop=pl.Series(p),
            eta_p10=pl.Series(q[:, 0]),
            eta_p50=pl.Series(q[:, 1]),
            eta_p90=pl.Series(q[:, 2]),
        )
