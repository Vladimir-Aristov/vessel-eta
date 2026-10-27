"""Turn one raw daily AIS file into a clean, region-clipped parquet partition.

Both raw formats (zipped CSV for 2015-2023, GeoParquet for 2024) end up in one
canonical schema. The CSV is streamed straight out of the zip in blocks, so a
~900 MB uncompressed day never has to be extracted to disk or held in memory.

Cleaning rules (AIS spec, ITU-R M.1371):
- position: lat in [-90, 90], lon in [-180, 180]; 91/181 mean "not available"; (0, 0) is junk
- SOG 102.3 = not available; COG 360 = not available; heading 511 = not available
- MMSI must be a ship station (first digit 2-7); base stations, SAR aircraft,
  AtoN and test MMSIs are dropped
- duplicate (mmsi, timestamp) rows are dropped
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv

from vessel_eta.config import BBox
from vessel_eta.data.sources import DailyFile, RawFormat

SCHEMA: dict[str, pl.DataType] = {
    "mmsi": pl.Int32(),
    "ts": pl.Datetime("ms", "UTC"),
    "lat": pl.Float64(),
    "lon": pl.Float64(),
    "sog": pl.Float32(),
    "cog": pl.Float32(),
    "heading": pl.Int16(),
    "vessel_name": pl.String(),
    "imo": pl.String(),
    "call_sign": pl.String(),
    "vessel_type": pl.Int16(),
    "status": pl.Int16(),
    "length": pl.Float32(),
    "width": pl.Float32(),
    "draft": pl.Float32(),
    "cargo": pl.Int16(),
    "transceiver": pl.String(),
}

_CSV_RENAME = {
    "MMSI": "mmsi",
    "BaseDateTime": "ts",
    "LAT": "lat",
    "LON": "lon",
    "SOG": "sog",
    "COG": "cog",
    "Heading": "heading",
    "VesselName": "vessel_name",
    "IMO": "imo",
    "CallSign": "call_sign",
    "VesselType": "vessel_type",
    "Status": "status",
    "Length": "length",
    "Width": "width",
    "Draft": "draft",
    "Cargo": "cargo",
    "TransceiverClass": "transceiver",
}

_STRING_COLS = ("vessel_name", "imo", "call_sign", "transceiver")
CSV_BLOCK_BYTES = 16 << 20  # 64 MB: same speed, 2.2 GB peak RSS vs 1.3 GB


@dataclass
class IngestStats:
    day: str
    rows_raw: int = 0
    rows_in_region: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    rows_out: int = 0
    vessels: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))


# ---------------------------------------------------------------- readers


def _to_canonical(df: pl.DataFrame) -> pl.DataFrame:
    """Add missing columns as nulls, cast everything to SCHEMA, fix column order."""
    cols = []
    for name, dtype in SCHEMA.items():
        if name not in df.columns:
            cols.append(pl.lit(None, dtype=dtype).alias(name))
        elif name == "ts" and df.schema[name] == pl.String:
            cols.append(pl.col(name).str.to_datetime("%Y-%m-%dT%H:%M:%S", time_unit="ms", time_zone="UTC"))
        elif name == "ts":
            cols.append(pl.col(name).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"))
        elif df.schema[name] == pl.String and dtype.is_numeric():
            # CSV ints can be written as "511.0", so parse as float first
            cols.append(pl.col(name).cast(pl.Float64, strict=False).cast(dtype, strict=False))
        else:
            cols.append(pl.col(name).cast(dtype, strict=False))
    return df.select(cols)


def _region_filter(bbox: BBox | None) -> pl.Expr:
    if bbox is None:
        return pl.lit(True)
    return pl.col("lon").is_between(bbox.min_lon, bbox.max_lon) & pl.col("lat").is_between(
        bbox.min_lat, bbox.max_lat
    )


def _iter_csv_blocks(path: Path) -> Iterator[pl.DataFrame]:
    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"{path.name}: expected one CSV inside, found {members}")
        with zf.open(members[0]) as fh:
            # Read every column as string; casting happens once in _to_canonical.
            # (Type inference on the first block is unsafe: IMO/CallSign can look numeric.)
            header = fh.readline().decode().strip().split(",")
            reader = pacsv.open_csv(
                fh,
                read_options=pacsv.ReadOptions(column_names=header, block_size=CSV_BLOCK_BYTES),
                convert_options=pacsv.ConvertOptions(
                    column_types={c: pa.string() for c in header},
                    strings_can_be_null=True,
                ),
            )
            for batch in reader:
                yield pl.from_arrow(pa.Table.from_batches([batch])).rename(
                    _CSV_RENAME, strict=False
                )


def _read_geoparquet(path: Path) -> pl.LazyFrame:
    lf = pl.scan_parquet(path)
    # NOAA tags the column as a geoarrow.wkb extension type; plain writers give Binary
    geom = pl.col("geometry")
    if lf.collect_schema()["geometry"] != pl.Binary:
        geom = geom.ext.storage()

    def coord(offset: int) -> pl.Expr:
        # WKB point, little endian: 1 byte order + 4 bytes type + x (lon) + y (lat)
        return geom.bin.slice(offset, 8).bin.reinterpret(dtype=pl.Float64, endianness="little")

    return lf.select(
        pl.exclude("geometry", "base_date_time"),
        pl.col("base_date_time").alias("ts"),
        coord(5).alias("lon"),
        coord(13).alias("lat"),
    )


# ---------------------------------------------------------------- cleaning


def clean(df: pl.DataFrame, stats: IngestStats) -> pl.DataFrame:
    """Apply the cleaning rules from the module docstring, recording what was dropped."""

    def drop(frame: pl.DataFrame, keep: pl.Expr, reason: str) -> pl.DataFrame:
        kept = frame.filter(keep)
        stats.dropped[reason] = stats.dropped.get(reason, 0) + frame.height - kept.height
        return kept

    df = df.with_columns(
        # sentinels -> null
        pl.when(pl.col("sog") < 102.25).then(pl.col("sog")).alias("sog"),
        pl.when(pl.col("cog") < 360).then(pl.col("cog")).alias("cog"),
        pl.when(pl.col("heading") < 360).then(pl.col("heading")).alias("heading"),
        # empty strings -> null
        *[pl.col(c).str.strip_chars().replace("", None) for c in _STRING_COLS],
        pl.when(pl.col("length") > 0).then(pl.col("length")).alias("length"),
        pl.when(pl.col("width") > 0).then(pl.col("width")).alias("width"),
        pl.when(pl.col("draft") > 0).then(pl.col("draft")).alias("draft"),
    )
    df = drop(df, pl.col("mmsi").is_between(200_000_000, 799_999_999), "non_ship_mmsi")
    df = drop(
        df,
        pl.col("lat").is_between(-90, 90)
        & pl.col("lon").is_between(-180, 180)
        & ~((pl.col("lat") == 0) & (pl.col("lon") == 0)),
        "bad_position",
    )
    df = drop(df, pl.col("ts").is_not_null(), "bad_timestamp")
    before = df.height
    df = df.unique(subset=["mmsi", "ts"], keep="first", maintain_order=True)
    stats.dropped["duplicate"] = before - df.height
    return df.sort("mmsi", "ts")


# ---------------------------------------------------------------- entry point


def read_raw(f: DailyFile, path: Path, bbox: BBox | None, stats: IngestStats) -> pl.DataFrame:
    """Read one raw file into the canonical schema, clipped to bbox (not yet cleaned)."""
    if f.fmt is RawFormat.GEOPARQUET:
        lf = _read_geoparquet(path)
        stats.rows_raw = lf.select(pl.len()).collect().item()
        return _to_canonical(lf.filter(_region_filter(bbox)).collect())

    parts = []
    for block in _iter_csv_blocks(path):
        stats.rows_raw += block.height
        block = _to_canonical(block)
        parts.append(block.filter(_region_filter(bbox)))
    return pl.concat(parts) if parts else pl.DataFrame(schema=SCHEMA)


def ingest_day(f: DailyFile, raw_path: Path, out_dir: Path, bbox: BBox | None) -> IngestStats:
    """raw file -> out_dir/date=YYYY-MM-DD/part-0.parquet. Returns quality stats."""
    stats = IngestStats(day=f.day.isoformat())
    df = read_raw(f, raw_path, bbox, stats)
    stats.rows_in_region = df.height
    df = clean(df, stats)
    stats.rows_out = df.height
    stats.vessels = df["mmsi"].n_unique()

    part_dir = out_dir / f"date={f.day.isoformat()}"
    part_dir.mkdir(parents=True, exist_ok=True)
    tmp = part_dir / "part-0.parquet.tmp"
    df.write_parquet(tmp, compression="zstd", statistics=True)
    tmp.rename(part_dir / "part-0.parquet")
    return stats


def partition_exists(out_dir: Path, f: DailyFile) -> bool:
    return (out_dir / f"date={f.day.isoformat()}" / "part-0.parquet").exists()
