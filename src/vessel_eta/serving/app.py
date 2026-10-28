"""HTTP API for the ETA model.

    uv run uvicorn vessel_eta.serving.app:app --port 8000
    open http://localhost:8000          # map demo
    open http://localhost:8000/docs     # OpenAPI

Environment: VESSEL_ETA_MODEL (artifact directory, default artifacts/eta_v2).
"""

from __future__ import annotations

import datetime as dt
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import polars as pl
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AwareDatetime, BaseModel, Field, field_validator, model_validator

from vessel_eta import artifact
from vessel_eta.serving.predictor import AlreadyInPort, PortState, Predictor, Static

STATIC_DIR = Path(__file__).parent / "static"
MAX_POINTS = 20_000
MAX_SPAN = dt.timedelta(days=30)


class Point(BaseModel):
    ts: AwareDatetime
    lat: Annotated[float, Field(ge=-90, le=90)]
    lon: Annotated[float, Field(ge=-180, le=180)]
    sog: Annotated[float | None, Field(ge=0, le=102.2, description="knots")] = None
    cog: Annotated[float | None, Field(ge=0, lt=360, description="degrees")] = None
    heading: Annotated[int | None, Field(ge=0, lt=360)] = None
    status: Annotated[int | None, Field(ge=0, le=15, description="AIS navigational status")] = None


class Ship(BaseModel):
    mmsi: Annotated[int, Field(ge=200_000_000, le=799_999_999, description="ship station MMSI")]
    vessel_type: Annotated[
        int, Field(ge=70, le=89, description="AIS type: 70-79 cargo, 80-89 tanker")
    ]
    length: Annotated[float | None, Field(gt=0, le=500)] = None
    width: Annotated[float | None, Field(gt=0, le=80)] = None
    draft: Annotated[float | None, Field(gt=0, le=30)] = None


class PortCounts(BaseModel):
    waiting: Annotated[int, Field(ge=0)]
    inbound: Annotated[int, Field(ge=0)]
    in_port: Annotated[int, Field(ge=0)]


class EtaRequest(BaseModel):
    ship: Ship
    track: Annotated[
        list[Point],
        Field(min_length=1, max_length=MAX_POINTS,
              description="the current approach, oldest first; ideally since the ship entered "
                          "the region or since its last silence of more than 6 h"),
    ]  # fmt: skip
    left_port_recently: bool = Field(
        False, description="the approach started at a departure from this port (shift, anchoring)"
    )
    port_state: PortCounts | None = None

    @field_validator("track")
    @classmethod
    def ordered(cls, track: list[Point]) -> list[Point]:
        if any(b.ts < a.ts for a, b in zip(track, track[1:], strict=False)):
            raise ValueError("track points must be in time order")
        if track[-1].ts - track[0].ts > MAX_SPAN:
            raise ValueError(f"track spans more than {MAX_SPAN.days} days")
        return track

    @model_validator(mode="after")
    def has_speed(self) -> EtaRequest:
        if self.track[-1].sog is None:
            raise ValueError("the last point needs a speed over ground (sog)")
        return self


class Quantiles(BaseModel):
    p10: float
    p50: float
    p90: float


class QuantileTimes(BaseModel):
    p10: dt.datetime
    p50: dt.datetime
    p90: dt.datetime


class EtaResponse(BaseModel):
    as_of: dt.datetime = Field(description="time of the last track point used")
    eta: QuantileTimes = Field(description="arrival time: median and 80% interval")
    eta_minutes: Quantiles
    waiting_now: bool = Field(description="mean speed over the last 30 min below 2 kn")
    p_wait_before_arrival: float | None = Field(
        description="probability of a waiting episode before arrival (ships under way only)"
    )
    distance_to_gate_km: float
    points_received: int
    points_used: int
    model_version: str | None


def create_app(model_dir: Path | None = None) -> FastAPI:
    model_dir = model_dir or Path(os.environ.get("VESSEL_ETA_MODEL", artifact.DEFAULT_DIR))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = Predictor.load(model_dir)
        yield

    app = FastAPI(
        title="vessel-eta",
        version="2",
        summary="Probabilistic ETA to the ports of Los Angeles / Long Beach from AIS",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict:
        meta = app.state.predictor.meta
        return {
            "status": "ok",
            "model_version": meta.get("version"),
            "trained_on": meta.get("trained_on"),
            "recalibrations": len(meta.get("recalibrations", [])),
        }

    @app.post("/v1/eta", response_model=EtaResponse)
    def eta(req: EtaRequest) -> dict:
        points = pl.DataFrame([p.model_dump() for p in req.track]).with_columns(
            pl.col("sog", "cog").cast(pl.Float32),
            pl.col("heading", "status").cast(pl.Int16),
        )
        state = PortState(**req.port_state.model_dump()) if req.port_state else None
        try:
            return app.state.predictor.predict(
                points, Static(**req.ship.model_dump()), req.left_port_recently, state
            )
        except AlreadyInPort as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
