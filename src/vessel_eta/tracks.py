"""Split each MMSI's points into physically consistent tracks.

Raw AIS for one MMSI is not always one ship:
- single-point position spikes (a report 100+ km off the track, then back);
- two ships broadcasting the same MMSI, so points alternate between them;
- stale messages re-received hours later with a new timestamp (a ship at anchor
  "jumps" 3 km to where it was earlier, then back).

All show up as moves the ship could not have made: faster than `max_speed_kn`,
or much faster than the speed over ground it reports (`sog_slack_kn`).

Points are assigned greedily in time order: a point joins the longest open
sub-track it could have reached under those limits (or within `jitter_km`),
otherwise it starts a new sub-track.
Short sub-tracks are closed after an hour of silence so they cannot pick up
the real track's points later, when a long gap makes everything "reachable".
Only the longest sub-track per MMSI is kept (see `clean_tracks`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from vessel_eta.geo import EARTH_RADIUS_KM, KM_PER_NM, TrackParams

MAX_SUBTRACKS = 1000
STALE_SPIKE_S = 3600  # a short sub-track silent this long is closed


@dataclass
class TrackStats:
    points_in: int
    points_dropped_spikes: int  # in sub-tracks shorter than min_points
    points_dropped_ghosts: int  # in secondary sub-tracks that are long enough to be real
    mmsi: int
    mmsi_with_ghosts: int


def _assign(mmsi, t_s, lat, lon, sog, p: TrackParams) -> np.ndarray:
    """Sub-track index per point. Inputs sorted by (mmsi, t)."""
    n = len(mmsi)
    sub = np.empty(n, dtype=np.int32)
    rad_lat = np.radians(lat)
    rad_lon = np.radians(lon)
    kn_to_kms = KM_PER_NM / 3600.0
    max_kms = p.max_speed_kn * kn_to_kms
    sog = np.where(np.isnan(sog), p.max_speed_kn, sog)  # unknown SOG: only the hard cap applies
    cur = None
    # open sub-tracks: [lat_rad, lon_rad, t_s, k, n_points, last_sog]
    open_tracks: list[list[float]] = []
    next_k = 0
    for i in range(n):
        if mmsi[i] != cur:
            cur = mmsi[i]
            open_tracks = []
            next_k = 0
        la, lo, t, v = rad_lat[i], rad_lon[i], t_s[i], sog[i]
        # A short sub-track that has gone quiet was a spike: stop offering it, otherwise
        # after a long gap it is "reachable" and could steal points from the real track.
        open_tracks = [
            tr for tr in open_tracks if tr[4] >= p.min_points or t - tr[2] <= STALE_SPIKE_S
        ]
        best, best_key = None, None
        for tr in open_tracks:
            dlat, dlon = la - tr[0], lo - tr[1]
            a = math.sin(dlat / 2) ** 2 + math.cos(la) * math.cos(tr[0]) * math.sin(dlon / 2) ** 2
            d = 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(a, 1.0)))
            limit = min(max_kms, (max(v, tr[5]) + p.sog_slack_kn) * kn_to_kms)
            if d <= p.jitter_km or d <= limit * (t - tr[2]):
                key = (-tr[4], d)  # longest reachable track wins, then nearest
                if best_key is None or key < best_key:
                    best, best_key = tr, key
        if best is None:
            best = [la, lo, t, min(next_k, MAX_SUBTRACKS - 1), 0, v]
            next_k += 1
            open_tracks.append(best)
        best[0], best[1], best[2], best[5] = la, lo, t, v
        best[4] += 1
        sub[i] = int(best[3])
    return sub


def clean_tracks(df: pl.DataFrame, params: TrackParams) -> tuple[pl.DataFrame, TrackStats]:
    """Keep only each MMSI's main physically consistent track. Sorted by (mmsi, ts).

    In January 2023 every secondary sub-track was small (median 26 points, at most
    ~10% of the vessel's points) and scattered over weeks: ghost positions, not a
    second ship. So only the longest sub-track is kept and MMSI stays the ship key.
    """
    df = df.sort("mmsi", "ts")
    sub = _assign(
        df["mmsi"].to_numpy(),
        df["ts"].dt.epoch("s").to_numpy().astype(np.float64),
        df["lat"].to_numpy(),
        df["lon"].to_numpy(),
        df["sog"].cast(pl.Float64).fill_null(float("nan")).to_numpy(),
        params,
    )
    df = df.with_columns(_sub=pl.Series(sub, dtype=pl.Int32))
    sizes = df.group_by("mmsi", "_sub").agg(pl.len().alias("n"))
    main = sizes.sort("n", "_sub", descending=[True, False]).unique("mmsi", keep="first")
    out = df.join(main.select("mmsi", "_sub"), on=["mmsi", "_sub"], how="semi").drop("_sub")
    secondary = sizes.join(main.select("mmsi", "_sub"), on=["mmsi", "_sub"], how="anti")
    is_spike = secondary["n"] < params.min_points
    stats = TrackStats(
        points_in=df.height,
        points_dropped_spikes=int(secondary.filter(is_spike)["n"].sum()),
        points_dropped_ghosts=int(secondary.filter(~is_spike)["n"].sum()),
        mmsi=main.height,
        mmsi_with_ghosts=secondary.filter(~is_spike)["mmsi"].n_unique(),
    )
    return out.sort("mmsi", "ts"), stats
