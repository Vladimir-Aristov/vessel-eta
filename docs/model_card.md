# Model card: vessel-eta v2

## Summary

Probabilistic estimated time of arrival (ETA) for cargo ships and tankers approaching the
ports of **Los Angeles and Long Beach**, from the ship's own AIS track. Output: median ETA,
an 80% interval (P10–P90), and the probability the ship waits (anchor, drift, another
port) before arriving.

| | |
|---|---|
| Version | v2, artifact `artifacts/eta_v2` |
| Type | 3 × 7 quantile LightGBM models + 1 LightGBM classifier, conformal interval shifts |
| Trained on | 7,582 arrivals of 2,514 ships, 2023-01-01 … 2024-11-30 (NOAA AIS) |
| Interval calibration | conformal shifts on held-out ships, re-calibrated on November 2024 |
| Serving | FastAPI, ~9 ms per request; `POST /v1/eta` |

## Intended use

- Decision support where an *uncertain* arrival time is useful: terminal and berth
  planning, trucking and drayage scheduling, logistics visibility.
- Ships of AIS type 70–89 (cargo, tanker) **outside** the LA/LB breakwater, with the
  current approach track available.

**Not for**: navigation or safety decisions; enforcement or any decision about a
specific crew or operator; other ports or ship types (not trained, not validated);
ships already inside the port.

## Data

- NOAA Office for Coastal Management / USCG *Nationwide AIS*, 2023–2024 (CC0), clipped to
  32–35° N, 121.5–117° W; 378M clean positions.
- Cleaning: AIS sentinels, non-ship MMSIs, duplicates, malformed rows; per-ship physical
  consistency (40 kn cap, implied speed vs reported SOG) removes spikes, ghost positions
  and stale replays (0.9% of points).
- Label: arrival = first sustained position behind the breakwater (polygon validated
  against moored and anchored positions). Arrivals whose entry time is unsure (>15 min
  silence at the gate) are not used for training: 474 of 8,411.
- AIS contains ship identifiers (MMSI, name) broadcast publicly by law; no personal data
  about people is used.

## Evaluation

All numbers are on data the model did not see; full details in the README.

| test | model MAE | distance/speed MAE | 80% interval holds |
|---|---|---|---|
| all of 2024, model trained on 2023 | 70.5 h | 85.8 h | 0.75 static, 0.78 with monthly re-calibration |
| December 2024, shipped artifact | 53.9 h | 68.9 h | 0.74 |
| — ships under way, < 30 km from the gate | 11.2 h | 13.5 h | 0.82 |
| — ships at anchor | 91.7 h | 120.6 h | 0.70 |

Ships never seen in training: voyage MAE 15.0 h vs 10.7 h for returning ships (2024).

## Limitations and known failure modes

- **Waiting time is the main error.** Whether and how long a ship queues depends on berth
  schedules that are not in AIS; for ships at anchor the error is days, not hours. The
  interval says so (median width ~7 days for waiting ships) — use it.
- **Intervals drift.** Calibrated once, coverage fell from 0.80 to 0.75 a year later
  (longer waits in 2024). Re-calibrate monthly (`vessel-eta monitor --apply`); the gates
  flag months outside 0.70–0.90.
- **The destination is assumed.** The model answers "when will this ship arrive at LA/LB,
  given that it is going there"; it does not decide whether a passing ship will call.
- **Input contract.** The client must send the whole current approach (since entering the
  region or since a > 6 h silence). A short snippet of a ship that has waited for days
  looks like a ship that just stopped.
- **Tails.** Quantiles are modelled up to P05/P95; the composed distribution is thin
  beyond them.
- **Unseen ships and new routes** are harder (see above); a new liner service changes the
  mix.

## Monitoring

`vessel-eta monitor --month YYYY-MM` scores a closed month: MAE vs distance/speed, interval
coverage, PSI drift of key inputs and of the predictions against the training reference
stored in `meta.json`. Gates: beats physics; coverage in [0.70, 0.90]; no input PSI > 0.25.
Status OK / WARN / ALERT; `--apply` re-estimates only the interval shifts. The API also
exports prediction histograms on `/metrics` for live monitoring before outcomes are known.
