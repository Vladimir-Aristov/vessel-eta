"""Port of Los Angeles "Historical Container Vessel Activity" yearly summary (PDF).

Published by the POLA Wharfinger Division at
https://portoflosangeles.org/business/supply-chain/ships (one PDF per calendar year).
The server refuses non-browser downloads, so the file is saved by hand to
data/external/ and not redistributed here.

One row per business day, Port of Los Angeles container ships only (not Long Beach,
not tankers): ships at anchor, at berth, departed, and for the departed ships the
average days at berth and at anchor + berth. In 2023 the anchor count is 0 on 159
of 246 days — the 2021-22 queue was gone — so this is a weak, sparse signal; it is
used as a separate feature group so its value can be measured on its own.

A report for day d is assumed known from day d+1 (time of publication unknown).
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import polars as pl

from vessel_eta.config import DATA_DIR

PDF_GLOB = "Port-of-Los-Angeles-Container-Vessel-Activity-Summary-*.pdf"
ROW = re.compile(r"^(\d{1,2}/\d{1,2}/\d{4})\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$")
FEATURES = ["pola_anchor", "pola_anchor_5r", "pola_berth", "pola_wait_days_5r"]


def parse_text(text: str) -> pl.DataFrame:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not re.match(r"^\d{1,2}/\d{1,2}/\d{4}", line):
            continue
        m = ROW.match(line)
        if not m:
            raise ValueError(f"unparsed POLA row: {line!r}")
        day, anchor, berth, departed, d_berth, d_total = m.groups()
        rows.append(
            {
                "day": dt.datetime.strptime(day, "%m/%d/%Y").date(),
                "anchor": int(anchor),
                "berth": int(berth),
                "departed": int(departed),
                "days_berth": float(d_berth),
                "days_total": float(d_total),
            }
        )
    return pl.DataFrame(rows)


def load(directory: Path = DATA_DIR / "external") -> pl.DataFrame | None:
    """All saved yearly PDFs -> daily table, or None if none are saved."""
    from pypdf import PdfReader

    paths = sorted(directory.glob(PDF_GLOB))
    if not paths:
        return None
    parts = []
    for p in paths:
        text = "\n".join(page.extract_text() for page in PdfReader(p).pages)
        parts.append(parse_text(text))
    return pl.concat(parts).unique("day").sort("day")


def features(daily: pl.DataFrame) -> pl.DataFrame:
    """known_from (= report day + 1) -> POLA features, last value and 5-report means."""
    # wait at anchor of the ships that departed that day; undefined when none departed
    wait = pl.when(pl.col("departed") > 0).then(pl.col("days_total") - pl.col("days_berth"))
    return (
        daily.sort("day")
        .with_columns(_wait=wait)
        .select(
            known_from=pl.col("day") + pl.duration(days=1),
            pola_anchor=pl.col("anchor").cast(pl.Float64),
            pola_anchor_5r=pl.col("anchor").cast(pl.Float64).rolling_mean(5, min_samples=1),
            pola_berth=pl.col("berth").cast(pl.Float64),
            pola_wait_days_5r=pl.col("_wait").rolling_mean(5, min_samples=1),
        )
    )
