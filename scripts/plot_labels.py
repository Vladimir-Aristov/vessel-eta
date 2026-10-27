"""Figures for the README: labelled approaches on a map + approach-time histogram.

uv run python scripts/plot_labels.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from vessel_eta.config import PROCESSED_DIR, ROOT  # noqa: E402
from vessel_eta.geo import load_port  # noqa: E402

OUT = ROOT / "docs" / "img"


def main() -> None:
    port = load_port()
    labels = pl.read_parquet(PROCESSED_DIR / f"eta_points_{port.name}.parquet")
    voyages = pl.read_parquet(PROCESSED_DIR / f"voyages_{port.name}.parquet")
    OUT.mkdir(parents=True, exist_ok=True)

    # --- map: every labelled point coloured by hours to arrival (capped at 48 h)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.6, 1]})
    pts = labels.sample(n=min(labels.height, 250_000), seed=0).sort("eta_min", descending=True)
    hours = np.minimum(pts["eta_min"].to_numpy() / 60, 48)
    for ax, (x0, x1, y0, y1), title in [
        (axes[0], (-121.5, -117.0, 32.0, 35.0), "Region: approaches coloured by time to arrival"),
        (axes[1], (-118.40, -118.05, 33.55, 33.80), "San Pedro Bay: port polygon and gates"),
    ]:
        sc = ax.scatter(pts["lon"], pts["lat"], c=hours, s=0.3, cmap="viridis_r", rasterized=True)
        px, py = port.polygon.exterior.xy
        ax.plot(px, py, color="crimson", lw=1.2)
        for name, (glon, glat) in port.gates.items():
            ax.plot(glon, glat, "o", color="crimson", ms=5)
            if x0 < glon < x1 and ax is axes[1]:
                ax.annotate(name.replace("_", " "), (glon, glat), xytext=(4, -12),
                            textcoords="offset points", fontsize=8, color="crimson")  # fmt: skip
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect(1 / np.cos(np.radians((y0 + y1) / 2)))
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
    cb = fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label("hours to arrival (capped at 48)")
    fig.savefig(OUT / "approaches_map.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # --- approach time distribution
    lab = voyages.filter(pl.col("labelled"))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bins = np.logspace(np.log10(0.5), np.log10(max(lab["duration_h"].max(), 1)), 40)
    for from_port, label in [(False, "from sea"), (True, "returning from port")]:
        d = lab.filter(pl.col("from_port") == from_port)["duration_h"]
        ax.hist(d, bins=bins, alpha=0.7, label=f"{label} (n={d.len()})")
    ax.set_xscale("log")
    ax.set_xlabel("labelled approach length, hours (log scale)")
    ax.set_ylabel("arrivals")
    ax.legend(fontsize=8)
    ax.set_title("How long before arrival we first see the ship (January 2023)", fontsize=10)
    fig.savefig(OUT / "approach_hours.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {[p.name for p in Path(OUT).glob('*.png')]}")


if __name__ == "__main__":
    main()
