# Design notes

How the system fits together and why it is built this way. Results and numbers live in
the README; this page is about decisions.

## Problem framing

ETA for a ship approaching LA/Long Beach, as a distribution, from public AIS only.
Arrival = passing the breakwater. Labels come free from the data (the moment a ship
enters); the hard part turned out to be the queue, not the sea passage.

## Data flow

```
NOAA daily files (zip CSV 2015-23, GeoParquet 2024)
  │  download.py   async, resumable (.part + Range), 12 connections (server throttles per connection)
  ▼
ingest.py          stream CSV out of the zip in 16 MB blocks, one schema for both formats,
  │                clip to region, AIS sentinels -> null, junk rows dropped and counted
  ▼  data/interim/ais/date=YYYY-MM-DD/part-0.parquet  (+ _manifest.jsonl quality log)
tracks.py          per-MMSI physical-consistency gating -> one clean track per ship
voyages.py         breakwater geofence, debounced arrivals, approach segments, eta_min labels
  ▼  data/processed/{voyages,eta_points,port_state}_la_lb.parquet
features.py        causal point features + port congestion (previous 30 min)
decompose.py       routing, waiting episodes, T / B / W / A models, conformal shifts
  ▼  artifacts/eta_v2  (gzipped LightGBM text + meta.json with drift reference)
serving/           FastAPI: request -> same build_features -> distribution; /metrics
monitoring.py      monthly quality, PSI drift, gates, re-calibration
```

## Decisions

| decision | alternatives | why |
|---|---|---|
| Polars + Parquet, streaming ingest | pandas, Spark, a database | 6.1 billion raw rows on a laptop: per-day streaming keeps memory flat (~1.3 GB) and Spark would add a cluster for a single-node problem. |
| Clean tracks by physical consistency | drop rows by thresholds; trust MMSI | One MMSI was often not one ship (spikes, ghosts, replays). Row filters cannot see a 26-kn "jump" by a ship that reports 0 kn; a per-track check can. |
| Breakwater polygon validated on data | port boundary from a map; distance threshold | "Arrived" had to match how ships behave: moored positions inside, anchorage circles never cut. Checked numerically, not drawn by eye. |
| Time split + held-out *ships* | random row split | Rows within a voyage are near-duplicates and regular callers repeat; random splits would grade memorisation. |
| Decomposed ETA (travel, will-it-wait, wait) | one regressor over everything | The target is a mixture (straight in vs queue for days); one regressor predicted neither well (v1). |
| LightGBM | GRU over the raw track, MLP | Measured: the track adds signal over an MLP, but boosting on good features stays ahead; the ensemble gain was ~0.3%. |
| Quantiles + conformal shifts | Gaussian / log-normal head, plain quantile regression | Quantile regression under-covered (76%); conformal shifts fix coverage without assuming a distribution, and can be updated without refitting. |
| Serve through the training feature code | re-implement features for serving | The only way to make train/serve skew impossible; a test compares the two paths. |
| Gzipped LightGBM text artifact | pickle / joblib, ONNX | Portable, inspectable, no code execution on load; ONNX adds a conversion for no benefit here. |
| Re-calibrate intervals, gate retraining | retrain on a schedule | A year later the model was still the best estimator but its intervals were stale; re-calibration is cheap and label-driven, retraining is only needed when quality or input drift gates fail. |

## Leakage defences

- Features use only the ship's points up to the prediction time and port counts from the
  previous 30-minute bucket; external data is lagged (PortWatch 8 days, Port of LA +1 day).
- `tests/test_features.py` truncates tracks at 71 cut-offs and requires identical features;
  it was mutation-tested with two planted leaks (one only caught after densifying cut-offs).
- A feature that encoded the dataset's start date (`from_port`) was found by its drift
  across months and redefined.
- Every hyper-parameter and calibration choice is made on training-period data; test
  periods are scored once.

## What is deliberately not here

- No stream processing (Kafka etc.): the data is historical and daily-batched; the API is
  stateless and takes the track per request.
- No feature store or model registry service: one model, one artifact directory with
  provenance in `meta.json` and recalibration history.
- No paid data. Berth line-ups (Marine Exchange) are the single most valuable missing
  input; public queue data (IMF PortWatch, Port of LA summary) was tested and added nothing.
