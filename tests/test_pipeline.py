"""End to end through the CLI on synthetic data in a temp dir:
interim partitions -> labels -> export-model -> recalibrate -> monitor -> report."""

import json

import polars as pl
import pytest
import yaml
from synthetic import DEG_LAT_PER_MIN_AT_10KN, GATE, stay, track
from typer.testing import CliRunner

from vessel_eta import artifact
from vessel_eta.cli import app
from vessel_eta.data.ingest import SCHEMA

runner = CliRunner()


def ship(mmsi, t0_min, anchor_min, speed):
    step = DEG_LAT_PER_MIN_AT_10KN * speed / 10
    rows = [(t0_min + i, 33.30 + i * step, GATE[0], speed) for i in range(60)]
    lat = rows[-1][1]
    if anchor_min:
        rows += stay(t0_min + 60, anchor_min, lat)
    t = t0_min + 60 + anchor_min
    n = int((GATE[1] - lat) / step) + 2
    rows += [(t + i, lat + i * step, GATE[0], speed) for i in range(n)]
    rows += stay(t + n, 30, 33.75)
    return track(mmsi, rows)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("ws")
    ships = []
    for k in range(16):  # 8 in January, 8 in February
        start = (0 if k < 8 else 35 * 24 * 60) + 600 * (k % 8)
        ships.append(ship(200000000 + k, start, [0, 180, 0, 300][k % 4], 8 + k % 5))
    pts = pl.concat(ships)
    pts = pts.with_columns(
        *[pl.lit(None, dtype).alias(c) for c, dtype in SCHEMA.items() if c not in pts.columns]
    ).select([pl.col(c).cast(t) for c, t in SCHEMA.items()])
    interim = root / "interim"
    for (day,), part in pts.group_by(pl.col("ts").dt.date()):
        d = interim / f"date={day}"
        d.mkdir(parents=True)
        part.write_parquet(d / "part-0.parquet")
    (interim / "_manifest.jsonl").write_text(
        json.dumps({"day": "2023-01-01", "rows_raw": 10, "rows_in_region": 9,
                    "dropped": {"duplicate": 1}, "rows_out": 8, "vessels": 3}) + "\n"
    )  # fmt: skip
    port = root / "port.yaml"
    port.write_text(yaml.safe_dump({
        "name": "synthetic",
        "polygon": [[-118.30, 33.70], [-118.10, 33.70], [-118.10, 33.80], [-118.30, 33.80]],
        "gates": {"angels_gate": list(GATE), "queens_gate": [GATE[0] + 0.05, GATE[1]]},
        "vessel_types": [70, 89],
    }))  # fmt: skip
    return root, port


def invoke(*args):
    res = runner.invoke(app, [str(a) for a in args])
    assert res.exit_code in (0, 2), res.output + repr(res.exception)  # 2 = monitor ALERT
    return res


def test_full_pipeline(workspace):
    root, port = workspace
    processed, art, reports = root / "processed", root / "artifact", root / "reports"

    invoke("labels", "--port", port, "--in-dir", root / "interim", "--out-dir", processed)
    voyages = pl.read_parquet(processed / "voyages_synthetic.parquet")
    assert voyages.filter("labelled").height == 16

    invoke("export-model", "--out", art, "--processed", processed, "--port", port)
    model, meta = artifact.load(art)
    assert meta["trained_on"]["voyages"] == 16 and "reference" in meta

    invoke("recalibrate", "--days", 20, "--path", art, "--processed", processed, "--port", port)
    assert len(json.loads((art / "meta.json").read_text())["recalibrations"]) == 1

    res = invoke("monitor", "--month", "2023-02", "--path", art, "--apply",
                 "--processed", processed, "--reports-dir", reports, "--port", port)  # fmt: skip
    assert "Monitoring report 2023-02" in (reports / "2023-02.md").read_text()
    assert len(json.loads((art / "meta.json").read_text())["recalibrations"]) == 2
    assert res.exit_code == (2 if "ALERT" in res.output else 0)

    res = invoke("report", "--out-dir", root / "interim")
    assert "clean rows" in res.output


def test_monitor_refuses_an_artifact_without_reference(workspace, tmp_path):
    root, port = workspace
    invoke("labels", "--port", port, "--in-dir", root / "interim", "--out-dir", tmp_path)
    invoke("export-model", "--out", tmp_path / "art", "--processed", tmp_path, "--port", port)
    meta = json.loads((tmp_path / "art" / "meta.json").read_text())
    meta.pop("reference")
    (tmp_path / "art" / "meta.json").write_text(json.dumps(meta))
    res = runner.invoke(app, ["monitor", "--month", "2023-02", "--path", str(tmp_path / "art"),
                              "--processed", str(tmp_path), "--port", str(port)])  # fmt: skip
    assert res.exit_code != 0 and "drift reference" in repr(res.exception)
