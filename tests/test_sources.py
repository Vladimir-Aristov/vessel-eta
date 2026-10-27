import datetime as dt

import pytest

from vessel_eta.data.sources import RawFormat, daily_file, date_range


def test_csv_years_use_noaa_zip():
    f = daily_file(dt.date(2023, 1, 5))
    assert f.fmt is RawFormat.ZIP_CSV
    assert f.url == "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/AIS_2023_01_05.zip"
    assert f.filename == "AIS_2023_01_05.zip"


def test_2024_uses_geoparquet():
    f = daily_file(dt.date(2024, 12, 31))
    assert f.fmt is RawFormat.GEOPARQUET
    assert f.url.endswith("/ais2024/ais-2024-12-31.parquet")


@pytest.mark.parametrize("day", [dt.date(2014, 12, 31), dt.date(2025, 1, 1)])
def test_unsupported_years_raise(day):
    with pytest.raises(ValueError):
        daily_file(day)


def test_date_range_is_inclusive():
    days = date_range(dt.date(2023, 2, 27), dt.date(2023, 3, 1))
    assert days == [dt.date(2023, 2, 27), dt.date(2023, 2, 28), dt.date(2023, 3, 1)]
    with pytest.raises(ValueError):
        date_range(dt.date(2023, 3, 1), dt.date(2023, 2, 1))
