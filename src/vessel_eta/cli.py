"""Command line entry point: `uv run vessel-eta --help`."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from vessel_eta.config import (
    DEFAULT_CONFIG,
    INTERIM_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    DataConfig,
    load_config,
)
from vessel_eta.data.download import Status, download, download_iter
from vessel_eta.data.ingest import ingest_day, partition_exists
from vessel_eta.data.sources import DailyFile, daily_file, date_range
from vessel_eta.features import port_state
from vessel_eta.geo import DEFAULT_PORT_CONFIG, load_port
from vessel_eta.tracks import clean_tracks
from vessel_eta.voyages import build_voyages, summarize

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()
log = logging.getLogger("vessel_eta")

MANIFEST = "_manifest.jsonl"
LABEL_INPUT_COLS = [
    "mmsi", "ts", "lat", "lon", "sog", "cog", "heading", "status",
    "vessel_type", "length", "width", "draft", "vessel_name",
]  # fmt: skip

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Data config YAML")]
StartOpt = Annotated[
    dt.datetime | None, typer.Option(formats=["%Y-%m-%d"], help="Overrides config")
]
EndOpt = Annotated[dt.datetime | None, typer.Option(formats=["%Y-%m-%d"], help="Overrides config")]


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    """Vessel ETA: NOAA AIS data pipeline and models."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is noise


def _days(cfg: DataConfig, start: dt.datetime | None, end: dt.datetime | None) -> list[DailyFile]:
    s = start.date() if start else cfg.start
    e = end.date() if end else cfg.end
    return [daily_file(d) for d in date_range(s, e)]


def _ingest_and_log(f: DailyFile, raw: Path, out_dir: Path, cfg: DataConfig) -> None:
    stats = ingest_day(f, raw, out_dir, cfg.bbox)
    with (out_dir / MANIFEST).open("a") as fh:
        fh.write(stats.to_json() + "\n")
    log.info(
        "%s: %s raw -> %s clean rows, %s vessels",
        f.day,
        f"{stats.rows_raw:,}",
        f"{stats.rows_out:,}",
        stats.vessels,
    )


@app.command("download")
def download_cmd(
    config: ConfigOpt = DEFAULT_CONFIG,
    start: StartOpt = None,
    end: EndOpt = None,
    concurrency: int = 12,  # NOAA throttles each connection (~0.75 MB/s), not the client
    out_dir: Path = RAW_DIR,
) -> None:
    """Only download raw daily files (resumable)."""
    cfg = load_config(config)
    results = download(_days(cfg, start, end), out_dir, concurrency=concurrency)
    _print_download_summary(results)
    if any(r.status is Status.FAILED for r in results):
        raise typer.Exit(1)


@app.command("ingest")
def ingest_cmd(
    config: ConfigOpt = DEFAULT_CONFIG,
    start: StartOpt = None,
    end: EndOpt = None,
    raw_dir: Path = RAW_DIR,
    out_dir: Path = INTERIM_DIR,
    force: bool = False,
) -> None:
    """Convert already-downloaded raw files into clean parquet partitions."""
    cfg = load_config(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in _days(cfg, start, end):
        raw = raw_dir / f.filename
        if not raw.exists():
            log.warning("%s: raw file missing, skipping", f.day)
        elif partition_exists(out_dir, f) and not force:
            log.info("%s: already ingested", f.day)
        else:
            _ingest_and_log(f, raw, out_dir, cfg)


@app.command("fetch")
def fetch_cmd(
    config: ConfigOpt = DEFAULT_CONFIG,
    start: StartOpt = None,
    end: EndOpt = None,
    concurrency: int = 12,  # NOAA throttles each connection (~0.75 MB/s), not the client
    raw_dir: Path = RAW_DIR,
    out_dir: Path = INTERIM_DIR,
) -> None:
    """Download + ingest as a pipeline, deleting raw files unless keep_raw is set.

    Each day is ingested as soon as its file arrives while the other downloads keep
    going, so peak disk use is about `concurrency` raw days (~300 MB each).
    """
    cfg = load_config(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [f for f in _days(cfg, start, end) if not partition_exists(out_dir, f)]
    log.info(
        "%d days to fetch (%d already ingested)", len(todo), len(_days(cfg, start, end)) - len(todo)
    )

    async def pipeline() -> int:
        failed = 0
        async for r in download_iter(todo, raw_dir, concurrency=concurrency):
            if r.status is Status.FAILED:
                failed += 1
                log.error("%s: download failed: %s", r.file.day, r.error)
                continue
            if r.status is Status.MISSING:
                log.warning("%s: not published on the server", r.file.day)
                continue
            try:  # CPU-bound: run off the event loop so downloads keep streaming
                await asyncio.to_thread(_ingest_and_log, r.file, r.path, out_dir, cfg)
            except Exception:  # one bad file must not stop the whole range
                failed += 1
                log.exception("%s: ingest failed, raw file kept for inspection", r.file.day)
                continue
            if not cfg.keep_raw:
                r.path.unlink()
        return failed

    failed = asyncio.run(pipeline())
    if failed:
        log.error("%d days failed; rerun the same command to retry them", failed)
        raise typer.Exit(1)


@app.command("labels")
def labels_cmd(
    port_config: Annotated[Path, typer.Option("--port", help="Port YAML")] = DEFAULT_PORT_CONFIG,
    in_dir: Path = INTERIM_DIR,
    out_dir: Path = PROCESSED_DIR,
) -> None:
    """Clean tracks, detect port arrivals, write voyages + ETA-labelled points."""
    port = load_port(port_config)
    lo, hi = port.vessel_types
    points = (
        pl.scan_parquet(in_dir / "**" / "*.parquet", hive_partitioning=False)
        .filter(pl.col("vessel_type").is_between(lo, hi))
        .select(LABEL_INPUT_COLS)  # skip IMO / call sign / transceiver: ~30% less memory
        .collect()
    )
    log.info(
        "%s points of %s ships with type %d-%d",
        f"{points.height:,}",
        points["mmsi"].n_unique(),
        lo,
        hi,
    )

    tracks, tstats = clean_tracks(points, port.tracks)
    log.info(
        "tracks: dropped %s spike + %s ghost points",
        f"{tstats.points_dropped_spikes:,}",
        f"{tstats.points_dropped_ghosts:,}",
    )
    voyages, labels = build_voyages(tracks, port)
    state = port_state(tracks, port)
    summary = {"port": port.name, "tracks": asdict(tstats), **summarize(voyages, labels)}

    out_dir.mkdir(parents=True, exist_ok=True)
    voyages.write_parquet(out_dir / f"voyages_{port.name}.parquet")
    labels.write_parquet(out_dir / f"eta_points_{port.name}.parquet", compression="zstd")
    state.write_parquet(out_dir / f"port_state_{port.name}.parquet")
    (out_dir / f"labels_{port.name}.json").write_text(json.dumps(summary, indent=2, default=str))

    t = Table(title=f"ETA labels: {port.name}")
    t.add_column("metric")
    t.add_column("value", justify="right")
    for k, v in summary.items():
        if k not in ("port", "tracks"):
            t.add_row(k, f"{v:,.1f}" if isinstance(v, float) else f"{v:,}")
    console.print(t)


@app.command("train")
def train_cmd(
    test_start: Annotated[
        dt.datetime,
        typer.Option(formats=["%Y-%m-%d"], help="Arrivals from this day are the test set"),
    ] = dt.datetime(2023, 3, 1),
    version: Annotated[
        str, typer.Option(help="v1: single regressor; v2: decomposed + quantiles")
    ] = "v1",
    features: Annotated[str, typer.Option(help="v2 only: base | queue | all")] = "base",
    train_end: Annotated[
        dt.datetime | None,
        typer.Option(formats=["%Y-%m-%d"], help="v2 only: train on arrivals before this day"),
    ] = None,
    name: Annotated[str | None, typer.Option(help="report name (default eta_<version>)")] = None,
    port_config: Annotated[Path, typer.Option("--port", help="Port YAML")] = DEFAULT_PORT_CONFIG,
) -> None:
    """Baselines + model; writes reports/<name>.md|json."""
    from vessel_eta.experiment import REPORTS_DIR, run, run_v2  # lightgbm import is slow-ish

    name = name or f"eta_{version}"
    start = test_start.replace(tzinfo=dt.UTC)
    if version == "v2":
        te = train_end.replace(tzinfo=dt.UTC) if train_end else None
        res = run_v2(load_port(port_config), start, name, features, te)
        console.print(Path(REPORTS_DIR / f"{name}.md").read_text())
        return
    res = run(load_port(port_config), start, name)
    t = Table(title=f"{name}: test ({res['sizes']['test']['voyages']} voyages), hours")
    for c in ("model", "mae_h", "median_ae_h", "voyage_mae_h", "median_bias_h"):
        t.add_column(c, justify="right")
    for r in res["test"]:
        t.add_row(
            r["model"],
            *(f"{r[c]:.1f}" for c in ("mae_h", "median_ae_h", "voyage_mae_h", "median_bias_h")),
        )
    console.print(t)
    console.print(f"full report: {REPORTS_DIR / (name + '.md')}")


@app.command("portwatch")
def portwatch_cmd(
    config: ConfigOpt = DEFAULT_CONFIG,
    start: StartOpt = None,
    end: EndOpt = None,
) -> None:
    """Download IMF PortWatch daily data for LA/Long Beach (range + 30 days before it)."""
    from vessel_eta.data import portwatch

    cfg = load_config(config)
    s = (start.date() if start else cfg.start) - dt.timedelta(days=30)  # room for the lag
    e = end.date() if end else cfg.end
    daily = portwatch.load(portwatch.fetch(s, e))
    log.info("PortWatch: %d days, %s .. %s", daily.height, daily["day"].min(), daily["day"].max())


@app.command("report")
def report_cmd(out_dir: Path = INTERIM_DIR) -> None:
    """Data quality summary of the ingested partitions."""
    manifest = out_dir / MANIFEST
    if not manifest.exists():
        console.print("Nothing ingested yet.")
        raise typer.Exit(1)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    # last entry per day wins (re-ingests append)
    df = pl.DataFrame(list({r["day"]: r for r in rows}.values())).sort("day")
    dropped = df["dropped"].struct.unnest()

    t = Table(title=f"AIS partitions in {out_dir}")
    t.add_column("metric")
    t.add_column("value", justify="right")
    t.add_row("days", f"{df.height}  ({df['day'].min()} .. {df['day'].max()})")
    t.add_row("raw rows (all US)", f"{df['rows_raw'].sum():,}")
    t.add_row("rows in region", f"{df['rows_in_region'].sum():,}")
    for c in dropped.columns:
        t.add_row(f"  dropped: {c}", f"{dropped[c].sum():,}")
    t.add_row("clean rows", f"{df['rows_out'].sum():,}")
    t.add_row("vessels per day (median)", f"{int(df['vessels'].median()):,}")
    console.print(t)


def _print_download_summary(results) -> None:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
        if r.status is Status.FAILED:
            log.error("%s: %s", r.file.day, r.error)
    console.print(", ".join(f"{k}: {v}" for k, v in counts.items()))


if __name__ == "__main__":
    app()
