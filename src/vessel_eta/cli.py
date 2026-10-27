"""Command line entry point: `uv run vessel-eta --help`."""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from vessel_eta.config import DEFAULT_CONFIG, INTERIM_DIR, RAW_DIR, DataConfig, load_config
from vessel_eta.data.download import Status, download
from vessel_eta.data.ingest import ingest_day, partition_exists
from vessel_eta.data.sources import DailyFile, daily_file, date_range

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()
log = logging.getLogger("vessel_eta")

MANIFEST = "_manifest.jsonl"

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
    concurrency: int = 4,
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
    concurrency: int = 4,
    raw_dir: Path = RAW_DIR,
    out_dir: Path = INTERIM_DIR,
) -> None:
    """Download + ingest in batches, deleting raw files unless keep_raw is set.

    Peak disk use is about `concurrency` raw days (~300 MB each), not the whole range.
    """
    cfg = load_config(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [f for f in _days(cfg, start, end) if not partition_exists(out_dir, f)]
    log.info(
        "%d days to fetch (%d already ingested)", len(todo), len(_days(cfg, start, end)) - len(todo)
    )
    failed = 0
    for i in range(0, len(todo), concurrency):
        batch = todo[i : i + concurrency]
        for r in download(batch, raw_dir, concurrency=concurrency):
            if r.status is Status.FAILED:
                failed += 1
                log.error("%s: download failed: %s", r.file.day, r.error)
            elif r.status is Status.MISSING:
                log.warning("%s: not published on the server", r.file.day)
            else:
                _ingest_and_log(r.file, r.path, out_dir, cfg)
                if not cfg.keep_raw:
                    r.path.unlink()
    if failed:
        log.error("%d days failed; rerun the same command to retry them", failed)
        raise typer.Exit(1)


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
