import datetime as dt
import struct
import zipfile

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vessel_eta.config import BBox
from vessel_eta.data.ingest import SCHEMA, ingest_day
from vessel_eta.data.sources import daily_file

BBOX = BBox(min_lon=-121.5, min_lat=32.0, max_lon=-117.0, max_lat=35.0)

CSV_HEADER = (
    "MMSI,BaseDateTime,LAT,LON,SOG,COG,Heading,VesselName,IMO,CallSign,"
    "VesselType,Status,Length,Width,Draft,Cargo,TransceiverClass"
)
CSV_ROWS = [
    # valid cargo ship in the region; heading written as float like real NOAA files
    "352001704,2023-01-01T00:00:04,33.70,-118.20,14.7,97.5,97.0,ENEOS EXPLORER,IMO9935492,3E2723,80,0,230,36,11.7,80,A",
    # same vessel, sentinels: SOG 102.3, COG 360, heading 511, empty strings, zero dims
    "352001704,2023-01-01T00:01:04,33.71,-118.21,102.3,360.0,511.0,,,,80,0,0,0,0.0,80,A",
    # exact duplicate (mmsi, ts) -> dropped
    "352001704,2023-01-01T00:01:04,33.71,-118.21,10.0,90.0,90.0,X,,,80,0,0,0,0.0,80,A",
    # outside region -> clipped before cleaning
    "368926035,2023-01-01T00:00:05,38.65,-90.17,0.1,360.0,511.0,KIMMSWICK,,AENA,33,15,,,,33,A",
    # AtoN / base-station MMSI in region -> dropped
    "993692028,2023-01-01T00:00:06,33.72,-118.22,0.0,0.0,511.0,BUOY,,,,,,,,,",
    # second ship, earlier timestamp -> output must be sorted by (mmsi, ts)
    "211000000,2023-01-01T00:05:00,34.00,-119.00,5.0,45.0,44,TUG ONE,,CALL,52,0,30,10,3.5,52,B",
]


@pytest.fixture
def csv_zip(tmp_path):
    path = tmp_path / "AIS_2023_01_01.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("AIS_2023_01_01.csv", "\n".join([CSV_HEADER, *CSV_ROWS]) + "\n")
    return path


def _read_out(out_dir, day):
    return pl.read_parquet(out_dir / f"date={day}" / "part-0.parquet")


def test_csv_ingest_cleans_and_clips(csv_zip, tmp_path):
    out = tmp_path / "out"
    stats = ingest_day(daily_file(dt.date(2023, 1, 1)), csv_zip, out, BBOX)
    df = _read_out(out, "2023-01-01")

    assert df.schema == pl.Schema(SCHEMA)
    assert stats.rows_raw == 6
    assert stats.rows_in_region == 5
    assert stats.dropped["non_ship_mmsi"] == 1
    assert stats.dropped["duplicate"] == 1
    assert stats.rows_out == df.height == 3
    assert stats.vessels == 2

    assert df["mmsi"].to_list() == [211000000, 352001704, 352001704]  # sorted
    first, second = df.filter(pl.col("mmsi") == 352001704).iter_rows(named=True)
    assert first["heading"] == 97  # "97.0" parsed, not lost
    assert first["ts"] == dt.datetime(2023, 1, 1, 0, 0, 4, tzinfo=dt.UTC)
    assert second["sog"] is None and second["cog"] is None and second["heading"] is None
    assert second["vessel_name"] is None and second["length"] is None and second["draft"] is None


def _wkb_point(lon, lat):
    return struct.pack("<BIdd", 1, 1, lon, lat)


def test_geoparquet_ingest_decodes_points(tmp_path):
    raw = tmp_path / "ais-2024-01-01.parquet"
    table = pa.table(
        {
            "mmsi": pa.array([352001704, 352001704, 368926035], pa.int32()),
            "base_date_time": pa.array(
                [
                    dt.datetime(2024, 1, 1, 0, 0),
                    dt.datetime(2024, 1, 1, 0, 1),
                    dt.datetime(2024, 1, 1),
                ],
                pa.timestamp("ms"),
            ),
            "sog": pa.array([12.0, None, 1.0], pa.float32()),
            "heading": pa.array([90, None, 10], pa.int32()),
            "vessel_name": ["A", "A", "B"],
            "transceiver": ["A", "A", "A"],
            "geometry": [
                _wkb_point(-118.2, 33.7),
                _wkb_point(-118.3, 33.8),
                _wkb_point(-90.0, 38.0),
            ],
        }
    )
    pq.write_table(table, raw)

    out = tmp_path / "out"
    stats = ingest_day(daily_file(dt.date(2024, 1, 1)), raw, out, BBOX)
    df = _read_out(out, "2024-01-01")

    assert stats.rows_raw == 3 and stats.rows_out == 2
    assert df["lon"].to_list() == pytest.approx([-118.2, -118.3])
    assert df["lat"].to_list() == pytest.approx([33.7, 33.8])
    assert df["ts"].dtype == pl.Datetime("ms", "UTC")
    assert df["cog"].null_count() == 2  # column missing in source -> all null, still present


def test_no_bbox_keeps_everything(csv_zip, tmp_path):
    stats = ingest_day(daily_file(dt.date(2023, 1, 1)), csv_zip, tmp_path / "out", None)
    assert stats.rows_in_region == 6
