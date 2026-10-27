"""Where NOAA/MarineCadastre publishes daily AIS files, and in what format.

The archive changed format over the years:
- 2015-2023: one zipped CSV per day on coast.noaa.gov
- 2024:      one GeoParquet file per day on Azure blob storage
Older years are file geodatabases and are not supported.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum


class RawFormat(StrEnum):
    ZIP_CSV = "zip_csv"
    GEOPARQUET = "geoparquet"


@dataclass(frozen=True)
class DailyFile:
    day: dt.date
    url: str
    fmt: RawFormat

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


FIRST_SUPPORTED = dt.date(2015, 1, 1)
LAST_SUPPORTED = dt.date(2024, 12, 31)

_CSV_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{y}/AIS_{y}_{m:02d}_{d:02d}.zip"
_PARQUET_URL = (
    "https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/ais{y}/ais-{y}-{m:02d}-{d:02d}.parquet"
)


def daily_file(day: dt.date) -> DailyFile:
    if not FIRST_SUPPORTED <= day <= LAST_SUPPORTED:
        raise ValueError(f"{day} is outside the supported range {FIRST_SUPPORTED}..{LAST_SUPPORTED}")
    if day.year <= 2023:
        url = _CSV_URL.format(y=day.year, m=day.month, d=day.day)
        return DailyFile(day, url, RawFormat.ZIP_CSV)
    url = _PARQUET_URL.format(y=day.year, m=day.month, d=day.day)
    return DailyFile(day, url, RawFormat.GEOPARQUET)


def date_range(start: dt.date, end: dt.date) -> list[dt.date]:
    """Inclusive list of days from start to end."""
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
