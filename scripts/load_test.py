"""Closed-loop load test of /v1/eta: N concurrent clients for a fixed duration.

Start the API first (one process), then:

    uv run python scripts/load_test.py --concurrency 8 --seconds 20

Each request carries the 120-point track in examples/request.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


async def client(http: httpx.AsyncClient, url: str, body: dict, stop: float, out: list) -> None:
    while time.perf_counter() < stop:
        t = time.perf_counter()
        r = await http.post(url, json=body)
        out.append((time.perf_counter() - t, r.status_code))


async def main(url: str, concurrency: int, seconds: float) -> None:
    body = json.loads((ROOT / "examples" / "request.json").read_text())
    results: list[tuple[float, int]] = []
    async with httpx.AsyncClient(timeout=30) as http:
        await http.post(url, json=body)  # warm-up
        start = time.perf_counter()
        stop = start + seconds
        await asyncio.gather(*(client(http, url, body, stop, results) for _ in range(concurrency)))
        elapsed = time.perf_counter() - start
    lat = np.array([r[0] for r in results]) * 1000
    ok = sum(1 for r in results if r[1] == 200)
    print(
        f"concurrency {concurrency}: {len(results) / elapsed:.0f} req/s, "
        f"latency p50 {np.percentile(lat, 50):.1f} ms, p95 {np.percentile(lat, 95):.1f} ms, "
        f"p99 {np.percentile(lat, 99):.1f} ms, errors {len(results) - ok}/{len(results)}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/eta")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=20)
    a = ap.parse_args()
    asyncio.run(main(a.url, a.concurrency, a.seconds))
