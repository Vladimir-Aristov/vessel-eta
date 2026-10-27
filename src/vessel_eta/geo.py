"""Distances and the port geofence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import shapely
import yaml

from vessel_eta.config import ROOT

EARTH_RADIUS_KM = 6371.0088
KM_PER_NM = 1.852
DEFAULT_PORT_CONFIG = ROOT / "configs" / "port.yaml"


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km; works on scalars or numpy arrays (degrees)."""
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(a, dtype=np.float64)) for a in (lat1, lon1, lat2, lon2)
    )
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass(frozen=True)
class TrackParams:
    max_speed_kn: float = 40.0
    sog_slack_kn: float = 10.0
    jitter_km: float = 0.5
    min_points: int = 10


@dataclass(frozen=True)
class VoyageParams:
    min_port_stay_min: float = 10.0
    max_gap_h: float = 6.0
    max_entry_gap_min: float = 15.0


@dataclass(frozen=True)
class Port:
    name: str
    polygon: shapely.Polygon
    gates: dict[str, tuple[float, float]]  # name -> (lon, lat)
    vessel_types: tuple[int, int]
    tracks: TrackParams = field(default_factory=TrackParams)
    voyages: VoyageParams = field(default_factory=VoyageParams)

    def contains(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        return shapely.contains_xy(self.polygon, np.asarray(lon), np.asarray(lat))

    def gate_distances_km(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """(n_points, n_gates) distance matrix, gates in config order."""
        lon, lat = np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
        return np.stack(
            [haversine_km(lat, lon, glat, glon) for glon, glat in self.gates.values()], axis=1
        )


def load_port(path: Path = DEFAULT_PORT_CONFIG) -> Port:
    raw = yaml.safe_load(path.read_text())
    polygon = shapely.Polygon(raw["polygon"])
    if not polygon.is_valid:
        raise ValueError(f"{path}: port polygon is not valid")
    lo, hi = raw["vessel_types"]
    return Port(
        name=raw["name"],
        polygon=polygon,
        gates={k: tuple(v) for k, v in raw["gates"].items()},
        vessel_types=(lo, hi),
        tracks=TrackParams(**raw.get("tracks", {})),
        voyages=VoyageParams(**raw.get("voyages", {})),
    )
