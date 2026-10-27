# vessel-eta

Predicting when a ship will arrive at port, and spotting ships that behave strangely,
from raw AIS transponder data published by NOAA.

> **Status:** data pipeline done. Modelling is next — see [Roadmap](#roadmap).

## Why this problem

Every ship over 300 GT broadcasts its position, speed and heading over AIS every few seconds.
NOAA archives these broadcasts for all US waters: billions of points per year, free (CC0).

- **ETA prediction** is a real product problem (ports, terminals, logistics, insurers)
  with labels that come for free: the moment a ship actually arrived is in the data.
- **Behavioural anomalies** — transponders switched off, ships loitering or meeting at sea,
  a declared vessel type that does not match how the ship moves — matter for maritime
  safety and enforcement, and have almost no labelled data.

The first region is the approaches to **Los Angeles / Long Beach**, the busiest container
port complex in the US.

## Data

| Years     | Source                         | Raw format                   | Size / day |
|-----------|--------------------------------|------------------------------|------------|
| 2015–2023 | `coast.noaa.gov` AISDataHandler | zipped CSV                   | ~320 MB zip / ~900 MB CSV |
| 2024      | MarineCadastre Azure blob      | GeoParquet (WKB points)      | ~230 MB    |

Both formats are normalised into one schema and written as daily Hive partitions:

```
data/interim/ais/date=2023-01-01/part-0.parquet   # one region, one day, zstd
data/interim/ais/_manifest.jsonl                  # row counts and drop reasons per day
```

| column | type | notes |
|---|---|---|
| `mmsi` | int32 | ship identity (only ship stations, first digit 2–7) |
| `ts` | datetime[ms, UTC] | |
| `lat`, `lon` | float64 | WGS84 |
| `sog`, `cog` | float32 | knots, degrees; AIS "not available" sentinels → null |
| `heading` | int16 | 511 → null |
| `vessel_name`, `imo`, `call_sign` | str | empty → null |
| `vessel_type`, `status`, `cargo` | int16 | AIS codes |
| `length`, `width`, `draft` | float32 | metres; 0 → null |
| `transceiver` | str | class A / B |

Cleaning rules (from ITU-R M.1371) and the reasons for every dropped row are in
[`ingest.py`](src/vessel_eta/data/ingest.py) and logged per day in the manifest.

### Pipeline design notes

- **Streams the CSV out of the zip** in 16 MB blocks and clips to the region per block,
  so a day never touches disk uncompressed. One day (8M rows, ~900 MB of CSV) is
  processed in ~4 s at ~1.3 GB peak memory, and the range length does not change that.
- **Resumable parallel downloads**: `.part` files + HTTP Range, size checked against
  `Content-Length` before the atomic rename, exponential backoff on 5xx/timeouts only.
- **Idempotent**: re-running a command skips finished days; interrupted runs continue.
- `fetch` deletes raw files after each batch, so a year of data needs about
  `concurrency × 320 MB` of scratch disk instead of ~115 GB. One region-day is ~3.5 MB.

## Quickstart

```bash
uv sync
uv run vessel-eta fetch                     # range + region from configs/data.yaml
uv run vessel-eta fetch --start 2023-02-01 --end 2023-02-07
uv run vessel-eta report                    # data quality summary
uv run pytest
```

NOAA's server is slow (~0.5–1 MB/s per connection); `--concurrency` helps.

## Roadmap

1. ~~**Data pipeline**~~ — download, normalise, clean, partition.
2. **Trajectories** — split each ship's points into voyages; detect port calls
   (arrival = entering a berth/anchorage polygon and staying); build the ETA labels.
3. **Baselines** — distance / current speed, then gradient boosting on hand-made features;
   validation split by time *and* by vessel to avoid leakage.
4. **Sequence model** — GRU / transformer over the recent track; quantile outputs for an
   ETA interval, not just a point.
5. **Anomalies** — AIS gaps, loitering, rendezvous; vessel-type classifier from motion only.
6. **Serving** — FastAPI endpoint, Docker, a map demo, drift monitoring.

## Layout

```
configs/data.yaml          date range, region bbox
src/vessel_eta/
  config.py                paths, config loader
  cli.py                   `vessel-eta` commands
  data/sources.py          URL + format per day
  data/download.py         async resumable downloader
  data/ingest.py           raw -> clean parquet
tests/                     offline tests (fake HTTP server, synthetic files)
```

Data: NOAA Office for Coastal Management / US Coast Guard Navigation Center,
*Nationwide Automatic Identification System*, CC0 1.0.
