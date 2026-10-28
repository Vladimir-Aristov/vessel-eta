"""Project paths and the data config loaded from configs/data.yaml."""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

# Project root (configs/, data/, artifacts/). Set VESSEL_ETA_ROOT when the package is
# installed rather than run from the source tree (e.g. in the Docker image).
ROOT = Path(os.environ.get("VESSEL_ETA_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "ais"
INTERIM_DIR = DATA_DIR / "interim" / "ais"
PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_CONFIG = ROOT / "configs" / "data.yaml"


@dataclass(frozen=True)
class BBox:
    """Lon/lat bounding box in WGS84."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def __post_init__(self) -> None:
        if not (self.min_lon < self.max_lon and self.min_lat < self.max_lat):
            raise ValueError(f"invalid bbox {self}")


@dataclass(frozen=True)
class DataConfig:
    start: dt.date
    end: dt.date
    region_name: str | None
    bbox: BBox | None
    keep_raw: bool


def load_config(path: Path = DEFAULT_CONFIG) -> DataConfig:
    raw = yaml.safe_load(path.read_text())
    region = raw.get("region") or {}
    bbox = BBox(**region["bbox"]) if region.get("bbox") else None
    return DataConfig(
        start=raw["start"],
        end=raw["end"],
        region_name=region.get("name"),
        bbox=bbox,
        keep_raw=bool(raw.get("keep_raw", False)),
    )
