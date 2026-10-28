"""Save / load the decomposed ETA model as a portable artifact.

Layout (all plain text, no pickle):

    <dir>/meta.json          features, quantiles, conformal shifts, version, provenance
    <dir>/stop.txt.gz        LightGBM classifier: will the ship wait before arriving
    <dir>/travel_qNN.txt.gz  LightGBM quantile models, one per quantile (NN = 05 .. 95)
    <dir>/after_stop_qNN.txt.gz
    <dir>/waiting_qNN.txt.gz

LightGBM's text format, gzipped (~3x smaller; still plain text inside).

Boosters are saved truncated at their early-stopping iteration, so a loaded model
predicts with all of its trees.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from vessel_eta.config import ROOT
from vessel_eta.decompose import QUANTILES, DecomposedETA, QuantileModel, StopClassifier

DEFAULT_DIR = ROOT / "artifacts" / "eta_v2"
FORMAT = 1
PARTS = ("travel", "after_stop", "waiting")


def _qname(part: str, q: float) -> str:
    return f"{part}_q{round(q * 100):02d}.txt.gz"


def _write(booster: lgb.Booster, path: Path) -> None:
    text = booster.model_to_string(num_iteration=booster.best_iteration or None)
    path.write_bytes(gzip.compress(text.encode(), mtime=0))  # mtime=0: reproducible bytes


def save(model: DecomposedETA, out: Path, info: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    _write(model.stop.booster, out / "stop.txt.gz")
    for part in PARTS:
        qm: QuantileModel = getattr(model, part)
        for q, booster in zip(QUANTILES, qm.boosters, strict=True):
            _write(booster, out / _qname(part, q))
    meta = {
        "format": FORMAT,
        "features": model.stop.features,
        "quantiles": QUANTILES.tolist(),
        "widen": {"waiting": list(model.widen[True]), "moving": list(model.widen[False])},
        "saved_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        **info,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


def load(path: Path = DEFAULT_DIR) -> tuple[DecomposedETA, dict]:
    meta = json.loads((path / "meta.json").read_text())
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path}: unsupported artifact format {meta.get('format')}")
    if not np.allclose(meta["quantiles"], QUANTILES):
        raise ValueError(f"{path}: artifact quantiles differ from the code's")
    features = meta["features"]

    def booster(name: str) -> lgb.Booster:
        return lgb.Booster(model_str=gzip.decompress((path / name).read_bytes()).decode())

    parts = {
        part: QuantileModel([booster(_qname(part, q)) for q in QUANTILES], features)
        for part in PARTS
    }
    model = DecomposedETA(
        stop=StopClassifier(booster("stop.txt.gz"), features),
        **parts,
        widen={True: tuple(meta["widen"]["waiting"]), False: tuple(meta["widen"]["moving"])},
    )
    return model, meta


def save_widen(path: Path, widen: dict[bool, tuple[float, float]], info: dict) -> dict:
    """Replace only the conformal shifts (re-calibration), keeping the boosters."""
    meta = json.loads((path / "meta.json").read_text())
    meta["widen"] = {"waiting": list(widen[True]), "moving": list(widen[False])}
    meta.setdefault("recalibrations", []).append(
        {"at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), **info}
    )
    (path / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta
