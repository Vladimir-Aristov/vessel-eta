# vessel-eta

Predicting when a ship will arrive at port, and spotting ships that behave strangely,
from raw AIS transponder data published by NOAA.

> **Status:** data pipeline, ETA labels and a first model done (Jan–Mar 2023: 880 arrivals,
> 1.13M labelled points). Headline: on moving ships the model beats physics by ~5–7%; the
> error is dominated by *anchorage waiting*, which AIS alone barely predicts — see
> [First model](#first-model-and-what-it-tells-us).

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

Even after row-level cleaning, 294 of the 418 cargo/tanker MMSIs seen in January had
at least one position jump no ship could make. Looking at the rows, three different causes:

| what | example | how it is caught |
|---|---|---|
| single-point spike | POLARIS VOYAGER reported 110+ km off its track for one minute | faster than 40 kn |
| ghost positions | SYNERGY KEELUNG "stationary inside the port" every few minutes while steaming at sea | faster than 40 kn |
| stale message re-received hours later | an anchored ship "jumps" 3 km to where it was 6 h earlier, then back | implied speed > reported SOG + 10 kn |

[`tracks.py`](src/vessel_eta/tracks.py) assigns each point, in time order, to the
longest sub-track it could physically have come from, and keeps only the main one.
Over Jan–Mar it drops 0.9% of points (7,434 spikes + 32,109 ghost/replay points;
ghosts on 370 of 836 ships). The third
case was only found because an early version produced a ship that "arrived" in
port twice in five hours without ever leaving.

### 2. What "arrived" means — the port geofence

Arrival = passing the breakwater through Angels Gate or Queens Gate. Ships waiting
at the outer anchorages have not arrived; ships moved to the inner anchorages have.
The polygon ([`configs/port.yaml`](configs/port.yaml)) follows the breakwater and was
checked against the data (Jan–Mar 2023, cleaned tracks) rather than drawn by eye:

- 99.999% of moored cargo positions (AIS status 5) fall inside;
- no anchorage swing circle is cut by the edge: of 848 at-anchor stays, the 6 with points
  on both sides are ships moved between an outer and an inner anchorage without changing
  their AIS status (two clusters 1.4–9.3 km apart, one after the other);
- labelled arrivals are first seen inside a median 0.27 km (p99 1.2 km) from a gate.

### 3. Voyages and labels

[`voyages.py`](src/vessel_eta/voyages.py), per ship:

- inside-port visits shorter than 10 min are boundary jitter, not arrivals;
- the approach before an arrival is cut at the last silence longer than 6 h, so a ship
  that passed by on day 3 and called on day 25 does not get a 22-day "ETA";
- if the ship is silent for more than 15 min right at the entrance, the arrival time
  is unsure: the voyage is kept but not labelled (90 of 970 arrivals);
- passing traffic and ships still at sea when the data ends get no labels.

| Jan–Mar 2023 | |
|---|---|
| arrivals detected | 970 |
| labelled arrivals (623 cargo, 257 tanker) | 880 |
| — approach starts at a departure from this port (berth shift, anchoring trip) | 107 |
| ships | 597 |
| labelled points | 1,134,014 (53% within 24 h of arrival) |
| approach length, median / p90 | 16.0 h / 81 h |

`from_port` is true only when the labelled approach begins at the port exit. An earlier
version set it whenever the ship had *any* earlier port stay in the data, so liner ships
returning from Hawaii a week later counted as "from port" — and the share grew with the
length of the data window (17% in January, 52% in March): a feature that would have
leaked the dataset's start date into the model.

![Approach length](docs/img/approach_hours.png)

The orange spike at ~7.5 h is almost entirely one ship: CATALINA PROVIDER, the
Santa Catalina Island supply vessel, with 51 of the 880 labelled arrivals. Its labels are
correct, but a model scored on random rows would look better than it is by learning one
easy, repetitive route — so validation must split by vessel, and metrics are reported
per voyage as well as per point.

The long tail is waiting: 102 labelled voyages spend more than half their approach
stationary at the outer anchorages. Predicting *that* wait is the hard part of ETA.

**The bar to beat.** Straight-line distance to the nearest gate divided by current
speed, on underway points (SOG > 5 kn) within 24 h of arrival (445k points):
**MAE 151 min**, and it is biased — the median prediction is 92 min too early (ships slow down for the pilot
and follow the traffic lanes, not a straight line).

## First model and what it tells us

`uv run vessel-eta train` → [`reports/eta_v1.md`](reports/eta_v1.md).

**Features** ([`features.py`](src/vessel_eta/features.py)): distance and bearing to the
gates, speed/course now and over the last 30 min–6 h, closing speed over the last hour,
time spent stationary so far, ship size/draft, hour and weekday, and port-wide counts
(ships waiting at anchor, inbound, in port) from the previous 30 minutes. Every feature
uses only data up to the prediction time; `tests/test_features.py` truncates tracks at 71
cut-offs and checks nothing changes. Two deliberate leaks (time measured from the
voyage's *end*; port state from the *current* half hour) were planted to make sure the
test catches them — the second was only caught after the cut-offs were made dense.

**Validation**: train on arrivals in Jan–Feb, test on March; early stopping on 20% of
*ships* held out from training; configuration chosen by 5-fold CV grouped by ship on
Jan–Feb ([`reports/cv_ablation.md`](reports/cv_ablation.md)); March scored once. One row
per 5 minutes per voyage (neighbouring minutes are near-duplicates, and the effective
sample is ~500 voyages, not 200k rows — hence small, heavily regularised trees).

**Test, March 2023** — 310 voyages; absolute error in hours:

| model | point MAE | median AE | voyage MAE | median bias |
|---|---|---|---|---|
| constant median | 103.9 | 26.0 | 27.1 | -18.5 |
| distance / speed | 97.1 | 33.8 | 16.1 | -28.0 |
| calibrated distance / speed | 93.3 | 34.3 | 19.7 | -0.6 |
| LightGBM | 99.4 | 27.8 | 16.0 | -13.1 |

By what you can see when predicting (MAE, hours):

| ship state | rows | median true ETA | distance / speed | calibrated | LightGBM |
|---|---|---|---|---|---|
| moving, < 30 km from gate | 9,219 | 1.6 | 12.5 | 13.6 | **11.9** |
| moving, ≥ 30 km | 38,134 | 11.6 | 28.8 | 28.7 | **26.8** |
| slow (0.5–5 kn) | 26,772 | 94.2 | 75.1 | **74.6** | 85.7 |
| stationary | 77,858 | 75.1 | 148.2 | **140.8** | 150.1 |

What this says:

1. **The model does not beat simple physics overall.** It wins on moving ships (5–7%),
   loses to a two-parameter calibration on ships that are slow or at anchor, and ties
   distance/speed per voyage. CV on Jan–Feb tells the same story, so this is not bad luck
   with March.
2. **The error is the queue, not the voyage.** Even moving ships within 30 km of the gate,
   whose median true ETA is 1.6 h, have a ~12 h MAE: some of them are heading for an
   anchorage and will wait for days. Whether a ship goes straight to a berth is decided by
   berth schedules that are not in AIS. "Distance/speed is off by only 0.5 h when the true
   ETA is under 2 h" is a selection effect — that slice contains only ships that went
   straight in (see the true-horizon table in the report).
3. **Unseen ships are much harder** (voyage MAE 19.6 h vs 11.6 h for ships seen in
   training), and all models under-predict them: March's new arrivals waited longer than
   anything in Jan–Feb.
4. Ship size and draft matter (removing them hurts in CV), and the strongest features are
   the physical ones: closing speed, current speed, distance/speed.

So the next model should not be a bigger regressor over the same mixture, but a
decomposition: *travel time to the next stop* (predictable from motion) + *will it anchor
first* (classifier) + *how long it waits* (survival-style, with port congestion), with
quantile outputs so the answer can be "2 h if it goes in, 1–4 days if it queues".

## Quickstart

```bash
uv sync
uv run vessel-eta fetch                     # range + region from configs/data.yaml
uv run vessel-eta fetch --start 2023-02-01 --end 2023-02-07
uv run vessel-eta report                    # data quality summary
uv run vessel-eta labels                    # tracks -> voyages -> ETA labels
uv run python scripts/plot_labels.py        # README figures
uv run python scripts/cv_ablation.py        # model selection (grouped CV, Jan-Feb)
uv run vessel-eta train                     # baselines + LightGBM, March test -> reports/
uv run pytest
```

NOAA's server is slow (~0.5–1 MB/s per connection); `--concurrency` helps.

## Roadmap

1. ~~**Data pipeline**~~ — download, normalise, clean, partition.
2. ~~**Trajectories and labels**~~ — clean tracks, port geofence, arrivals, ETA labels.
3. ~~**Baselines**~~ — causal features, time + ship validation, LightGBM vs physics.
4. **Decomposed ETA** — time to next stop + anchor-or-not + waiting time, quantile outputs;
   then a sequence model (GRU / transformer) over the recent track for the travel part.
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
  features.py              causal point features + port congestion
  modeling.py              splits, baselines, metrics, LightGBM, grouped CV
  experiment.py            `train`: end-to-end run -> reports/
scripts/plot_labels.py     README figures
scripts/cv_ablation.py     model selection table
reports/                   eval reports (markdown + json)
tests/                     offline tests (fake HTTP server, synthetic files and tracks)
```

Data: NOAA Office for Coastal Management / US Coast Guard Navigation Center,
*Nationwide Automatic Identification System*, CC0 1.0.
