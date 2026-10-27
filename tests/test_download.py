import asyncio
import datetime as dt

import httpx
import pytest

from vessel_eta.data import download as dl
from vessel_eta.data.download import Status, download_async
from vessel_eta.data.sources import daily_file

PAYLOAD = bytes(range(256)) * 4000  # ~1 MB


class FakeServer:
    """Serves PAYLOAD for known URLs, honours Range, can fail the first N GETs."""

    def __init__(self, urls, fail_first=0, fail_status=503):
        self.urls = set(urls)
        self.fail_left = fail_first
        self.fail_status = fail_status
        self.range_headers = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url not in self.urls:
            return httpx.Response(404)
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(PAYLOAD))})
        if self.fail_left:
            self.fail_left -= 1
            return httpx.Response(self.fail_status)
        rng = request.headers.get("range")
        self.range_headers.append(rng)
        if rng:
            start = int(rng.removeprefix("bytes=").rstrip("-"))
            return httpx.Response(206, content=PAYLOAD[start:])
        return httpx.Response(200, content=PAYLOAD)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    async def instant(_):
        return None

    monkeypatch.setattr(dl.asyncio, "sleep", instant)


def run(files, out, server, **kw):
    return asyncio.run(
        download_async(files, out, show_progress=False, transport=httpx.MockTransport(server), **kw)
    )


F1 = daily_file(dt.date(2023, 1, 1))
F2 = daily_file(dt.date(2023, 1, 2))


def test_downloads_then_skips(tmp_path):
    server = FakeServer([F1.url, F2.url])
    res = run([F1, F2], tmp_path, server)
    assert [r.status for r in res] == [Status.DOWNLOADED, Status.DOWNLOADED]
    assert (tmp_path / F1.filename).read_bytes() == PAYLOAD
    assert not list(tmp_path.glob("*.part"))

    res = run([F1, F2], tmp_path, server)
    assert [r.status for r in res] == [Status.SKIPPED, Status.SKIPPED]


def test_resumes_partial_file(tmp_path):
    (tmp_path / (F1.filename + ".part")).write_bytes(PAYLOAD[:1000])
    server = FakeServer([F1.url])
    [res] = run([F1], tmp_path, server)
    assert res.status is Status.DOWNLOADED
    assert server.range_headers == ["bytes=1000-"]
    assert (tmp_path / F1.filename).read_bytes() == PAYLOAD


def test_missing_day_is_reported_not_failed(tmp_path):
    [res] = run([F1], tmp_path, FakeServer([]))
    assert res.status is Status.MISSING


def test_retries_transient_errors(tmp_path):
    [res] = run([F1], tmp_path, FakeServer([F1.url], fail_first=2), retries=3)
    assert res.status is Status.DOWNLOADED


def test_gives_up_after_retries(tmp_path):
    [res] = run([F1], tmp_path, FakeServer([F1.url], fail_first=10), retries=2)
    assert res.status is Status.FAILED


def test_does_not_retry_client_errors(tmp_path):
    server = FakeServer([F1.url], fail_first=1, fail_status=403)
    [res] = run([F1], tmp_path, server, retries=3)
    assert res.status is Status.FAILED
    assert server.fail_left == 0 and server.range_headers == []
