"""Parallel, resumable download of daily AIS files.

- Files are written to `<name>.part` and renamed only after the size matches
  the server's Content-Length, so a finished file is never half-written.
- An interrupted `.part` is resumed with an HTTP Range request.
- Transient errors are retried with exponential backoff.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import httpx
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from vessel_eta.data.sources import DailyFile

log = logging.getLogger(__name__)

CHUNK = 1 << 20  # 1 MiB
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class Status(StrEnum):
    DOWNLOADED = "downloaded"
    SKIPPED = "skipped"  # already on disk
    MISSING = "missing"  # 404 on the server
    FAILED = "failed"


@dataclass(frozen=True)
class Result:
    file: DailyFile
    status: Status
    path: Path | None = None
    error: str | None = None


async def _remote_size(client: httpx.AsyncClient, url: str) -> int | None:
    r = await client.head(url)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return int(r.headers["content-length"])


async def _fetch(
    client: httpx.AsyncClient,
    f: DailyFile,
    dest: Path,
    progress: Progress | None,
) -> Status:
    total = await _remote_size(client, f.url)
    if total is None:
        return Status.MISSING
    if dest.exists() and dest.stat().st_size == total:
        return Status.SKIPPED

    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > total:  # stale partial from a different file version
        part.unlink()
        have = 0

    task = progress.add_task(f.filename, total=total, completed=have) if progress else None
    headers = {"Range": f"bytes={have}-"} if have else {}
    async with client.stream("GET", f.url, headers=headers) as r:
        if have and r.status_code != 206:  # server ignored Range: start over
            have = 0
            if progress and task is not None:
                progress.reset(task, total=total)
        r.raise_for_status()
        with part.open("ab" if have else "wb") as fh:
            async for chunk in r.aiter_bytes(CHUNK):
                fh.write(chunk)
                if progress and task is not None:
                    progress.advance(task, len(chunk))
    if progress and task is not None:
        progress.remove_task(task)

    size = part.stat().st_size
    if size != total:
        raise OSError(f"size mismatch for {f.filename}: got {size}, expected {total}")
    part.rename(dest)
    return Status.DOWNLOADED


async def _fetch_with_retries(
    client: httpx.AsyncClient,
    f: DailyFile,
    out_dir: Path,
    sem: asyncio.Semaphore,
    progress: Progress | None,
    retries: int,
) -> Result:
    dest = out_dir / f.filename
    async with sem:
        for attempt in range(retries + 1):
            try:
                status = await _fetch(client, f, dest, progress)
                return Result(f, status, dest if status != Status.MISSING else None)
            except (httpx.TransportError, httpx.HTTPStatusError, OSError) as e:
                retryable = not isinstance(e, httpx.HTTPStatusError) or (
                    e.response.status_code in RETRYABLE_STATUS
                )
                if not retryable or attempt == retries:
                    return Result(f, Status.FAILED, error=repr(e))
                wait = 2**attempt
                log.warning("%s: %r, retry %d/%d in %ds", f.filename, e, attempt + 1, retries, wait)
                await asyncio.sleep(wait)
    raise AssertionError("unreachable")


async def download_async(
    files: list[DailyFile],
    out_dir: Path,
    concurrency: int = 4,
    retries: int = 4,
    show_progress: bool = True,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[Result]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(30.0, read=120.0)
    progress = (
        Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        )
        if show_progress
        else None
    )
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True, transport=transport
    ) as client:
        coros = [_fetch_with_retries(client, f, out_dir, sem, progress, retries) for f in files]
        if progress:
            with progress:
                return await asyncio.gather(*coros)
        return await asyncio.gather(*coros)


def download(files: list[DailyFile], out_dir: Path, **kwargs) -> list[Result]:
    return asyncio.run(download_async(files, out_dir, **kwargs))
