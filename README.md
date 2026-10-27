# vessel-eta

Predicting when a ship will arrive at port, and spotting ships that behave strangely,
from raw AIS transponder data published by NOAA.

> **Status:** data pipeline and ETA labels done (January 2023: 310 arrivals, 388k labelled
> points). Baselines are next — see [Roadmap](#roadmap).

![Labelled approaches to LA / Long Beach](docs/img/approaches_map.png)

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

## From points to ETA labels

`uv run vessel-eta labels` turns the clean points of cargo ships and tankers
(AIS types 70–89) into voyages and labelled approach points:

```
data/processed/voyages_la_lb.parquet      # one row per port arrival
data/processed/eta_points_la_lb.parquet   # approach points with eta_min (the target)
data/processed/labels_la_lb.json          # summary below
```

### 1. One MMSI is not always one ship

Even after row-level cleaning, 294 of 418 cargo/tanker MMSIs had at least one
position jump no ship could make. Looking at the rows, three different causes:

| what | example | how it is caught |
|---|---|---|
| single-point spike | POLARIS VOYAGER reported 110+ km off its track for one minute | faster than 40 kn |
| ghost positions | SYNERGY KEELUNG "stationary inside the port" every few minutes while steaming at sea | faster than 40 kn |
| stale message re-received hours later | an anchored ship "jumps" 3 km to where it was 6 h earlier, then back | implied speed > reported SOG + 10 kn |

[`tracks.py`](src/vessel_eta/tracks.py) assigns each point, in time order, to the
longest sub-track it could physically have come from, and keeps only the main one.
It drops 0.6% of points (2,925 spikes + 7,317 ghost/replay points). The third
case was only found because an early version produced a ship that "arrived" in
port twice in five hours without ever leaving.

### 2. What "arrived" means — the port geofence

Arrival = passing the breakwater through Angels Gate or Queens Gate. Ships waiting
at the outer anchorages have not arrived; ships moved to the inner anchorages have.
The polygon ([`configs/port.yaml`](configs/port.yaml)) follows the breakwater and was
checked against the data rather than drawn by eye:

- 99.998% of moored cargo positions (AIS status 5) fall inside;
- no anchorage swing circle is cut by the edge: of 269 at-anchor stays, the 3 with points
  on both sides are ships moved between an outer and an inner anchorage without changing
  their AIS status (two clusters 1.4–6.9 km apart, one after the other);
- 99% of labelled arrivals are first seen inside within 0.56 km of a gate.

### 3. Voyages and labels

[`voyages.py`](src/vessel_eta/voyages.py), per ship:

- inside-port visits shorter than 10 min are boundary jitter, not arrivals;
- the approach before an arrival is cut at the last silence longer than 6 h, so a ship
  that passed by on day 3 and called on day 25 does not get a 22-day "ETA";
- if the ship is silent for more than 15 min right at the entrance, the arrival time
  is unsure: the voyage is kept but not labelled (31 of 341 arrivals);
- passing traffic and ships still at sea when the data ends get no labels.

| January 2023 | |
|---|---|
| arrivals detected | 341 |
| labelled arrivals (229 cargo, 81 tanker) | 310 |
| — of them returning after leaving port (shifts, anchoring) | 53 |
| ships | 265 |
| labelled points | 387,896 (56% within 24 h of arrival) |
| approach length, median / p90 | 16.6 h / 83 h |

![Approach length](docs/img/approach_hours.png)

The long tail is waiting: 36 labelled voyages spend more than half their approach
stationary at the outer anchorages. Predicting *that* wait is the hard part of ETA.

**The bar to beat.** Straight-line distance to the nearest gate divided by current
speed, on underway points (SOG > 5 kn) within 24 h of arrival: **MAE 150 min**, and it
is biased — the median prediction is 93 min too early (ships slow down for the pilot
and follow the traffic lanes, not a straight line).

## Quickstart

```bash
uv sync
uv run vessel-eta fetch                     # range + region from configs/data.yaml
uv run vessel-eta fetch --start 2023-02-01 --end 2023-02-07
uv run vessel-eta report                    # data quality summary
uv run vessel-eta labels                    # tracks -> voyages -> ETA labels
uv run python scripts/plot_labels.py        # README figures
uv run pytest
```

NOAA's server is slow (~0.5–1 MB/s per connection); `--concurrency` helps.

## Roadmap

1. ~~**Data pipeline**~~ — download, normalise, clean, partition.
2. ~~**Trajectories and labels**~~ — clean tracks, port geofence, arrivals, ETA labels.
3. **Baselines** — gradient boosting on hand-made features against the distance/speed
   baseline above; validation split by time *and* by vessel to avoid leakage.
4. **Sequence model** — GRU / transformer over the recent track; quantile outputs for an
   ETA interval, not just a point.
5. **Anomalies** — AIS gaps, loitering, rendezvous; vessel-type classifier from motion only.
6. **Serving** — FastAPI endpoint, Docker, a map demo, drift monitoring.

## Layout

```
configs/data.yaml          date range, region bbox
configs/port.yaml          port polygon, gates, track/voyage thresholds
src/vessel_eta/
  config.py                paths, config loader
  cli.py                   `vessel-eta` commands
  data/sources.py          URL + format per day
  data/download.py         async resumable downloader
  data/ingest.py           raw -> clean parquet
  geo.py                   distances, port geofence
  tracks.py                spikes / ghosts / replays -> one clean track per ship
  voyages.py               arrivals, approach segments, ETA labels
scripts/plot_labels.py     README figures
tests/                     offline tests (fake HTTP server, synthetic files and tracks)
```

Data: NOAA Office for Coastal Management / US Coast Guard Navigation Center,
*Nationwide Automatic Identification System*, CC0 1.0.
