"""Sequence model (GRU) for the travel-time part of the decomposed ETA.

Drop-in alternative to the LightGBM `QuantileModel` for T (time to the next event)
on moving ships: same 7 quantiles of log1p(minutes), same `predict_log` contract.

Input per prediction row:
- the last `WINDOW` rows of the same voyage (5-minute grid, causal: only rows at or
  before the prediction time), each described by `STEP_COLS`;
- the tabular features of the prediction row (standardised, NaN -> 0 + a missing flag),
  plus embeddings of AIS ship type and navigational status.

With `use_sequence=False` the GRU branch is dropped: the same network on tabular
features only, to measure what the sequence itself adds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
import torch
from torch import nn

from vessel_eta.decompose import QUANTILES

WINDOW = 24  # 24 x 5 min = 2 h of track
STEP_COLS = ["_east", "_north", "_sog", "cog_sin", "cog_cos", "_stationary", "_gate", "_dt"]
ANGELS = (-118.2514, 33.7086)  # origin for the local east/north frame
TYPE_BUCKETS = 21  # vessel_type 70..89 -> 1..20, other/unknown -> 0
STATUS_BUCKETS = 17  # AIS nav status 0..15 -> 1..16, unknown -> 0


def _device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def step_table(df: pl.DataFrame) -> pl.DataFrame:
    """Per-row step features; df must be sorted by (voyage_id, ts)."""
    lat0 = np.radians(ANGELS[1])
    return df.with_columns(
        _east=((pl.col("lon") - ANGELS[0]) * 111.32 * np.cos(lat0) / 100).cast(pl.Float32),
        _north=((pl.col("lat") - ANGELS[1]) * 110.57 / 100).cast(pl.Float32),
        _sog=(pl.col("sog").fill_null(0).clip(0, 30) / 20).cast(pl.Float32),
        _stationary=pl.col("stationary").fill_null(False).cast(pl.Float32),
        _gate=(pl.col("gate_km") / 100).cast(pl.Float32),
        _dt=(
            (pl.col("ts").diff().over("voyage_id").dt.total_seconds().fill_null(0) / 3600)
            .clip(0, 2)
            .cast(pl.Float32)
        ),
    ).with_columns(pl.col("cog_sin", "cog_cos").fill_null(0).cast(pl.Float32))


def window_index(voyage_ids: np.ndarray, rows: np.ndarray, window: int = WINDOW):
    """(len(rows), window) indices of the previous rows of the same voyage, and a mask.

    voyage_ids: per-row voyage key of the full sorted table; rows: positions to predict.
    The last column is the row itself; earlier columns go back in time.
    """
    n = len(voyage_ids)
    is_start = np.ones(n, dtype=bool)
    is_start[1:] = voyage_ids[1:] != voyage_ids[:-1]
    start_of = np.maximum.accumulate(np.where(is_start, np.arange(n), 0))
    idx = rows[:, None] + np.arange(-window + 1, 1)[None, :]
    mask = idx >= start_of[rows][:, None]
    return np.where(mask, idx, 0), mask


@dataclass
class _Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> _Scaler:
        return cls(np.nanmean(x, axis=0), np.nanstd(x, axis=0) + 1e-6)

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.std
        missing = np.isnan(z)
        return np.concatenate([np.nan_to_num(z, nan=0.0), missing], axis=1).astype(np.float32)


class _Net(nn.Module):
    def __init__(self, n_tab: int, use_sequence: bool, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.use_sequence = use_sequence
        self.type_emb = nn.Embedding(TYPE_BUCKETS, 4)
        self.status_emb = nn.Embedding(STATUS_BUCKETS, 4)
        self.tab = nn.Sequential(nn.Linear(n_tab + 8, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        if use_sequence:
            self.gru = nn.GRU(len(STEP_COLS) + 1, hidden, batch_first=True)
        width = hidden * (2 if use_sequence else 1)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(QUANTILES)),
        )

    def forward(self, seq, mask, tab, vtype, status):
        h = self.tab(torch.cat([tab, self.type_emb(vtype), self.status_emb(status)], dim=1))
        if self.use_sequence:
            x = torch.cat([seq, mask.unsqueeze(-1)], dim=-1)
            out, _ = self.gru(x)
            h = torch.cat([h, out[:, -1]], dim=1)
        raw = self.head(h)
        # monotone quantiles: first knot + cumulative positive gaps
        return torch.cat(
            [raw[:, :1], raw[:, :1] + torch.cumsum(nn.functional.softplus(raw[:, 1:]), 1)], 1
        )


def _pinball(pred: torch.Tensor, y: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    d = y.unsqueeze(1) - pred
    return torch.maximum(q * d, (q - 1) * d).mean()


@dataclass
class SequenceTravelModel:
    """GRU (+ tabular) quantile model; `predict_log` matches decompose.QuantileModel."""

    features: list[str]
    use_sequence: bool = True
    epochs: int = 20
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 0.0
    dropout: float = 0.0
    patience: int = 3
    seed: int = 0
    history: list[dict] = field(default_factory=list)

    # ---------------------------------------------------------------- batching
    def _prepare(self, full: pl.DataFrame):
        full = step_table(full.sort("voyage_id", "ts"))
        return full, full["voyage_id"].to_numpy(), full.select(STEP_COLS).to_numpy()

    def _tab(self, rows_df: pl.DataFrame):
        num = [c for c in self.features if c not in ("vessel_type", "status")]
        x = rows_df.select(pl.col(num).cast(pl.Float64)).to_numpy()
        vtype = rows_df["vessel_type"].fill_null(0).cast(pl.Int64).to_numpy()
        vtype = np.where((vtype >= 70) & (vtype <= 89), vtype - 69, 0)
        status = rows_df["status"].fill_null(-1).cast(pl.Int64).to_numpy()
        status = np.where((status >= 0) & (status <= 15), status + 1, 0)
        return x, vtype, status

    def _batches(self, steps, vids, rows, tab, vtype, status, y=None, shuffle=False):
        order = (
            np.random.default_rng(self.seed).permutation(len(rows))
            if shuffle
            else np.arange(len(rows))
        )
        dev = _device()
        for s in range(0, len(rows), self.batch_size):
            b = order[s : s + self.batch_size]
            idx, mask = window_index(vids, rows[b])
            seq = steps[idx] * mask[..., None]
            out = [
                torch.from_numpy(seq).to(dev),
                torch.from_numpy(mask.astype(np.float32)).to(dev),
                torch.from_numpy(tab[b]).to(dev),
                torch.from_numpy(vtype[b]).to(dev),
                torch.from_numpy(status[b]).to(dev),
            ]
            if y is not None:
                out.append(torch.from_numpy(y[b]).to(dev))
            yield out

    def _rows(self, full: pl.DataFrame, subset: pl.DataFrame) -> np.ndarray:
        """Positions of `subset` rows inside the sorted full table."""
        key = full.select("voyage_id", "ts").with_row_index("_pos")
        pos = subset.select("voyage_id", "ts").join(key, on=["voyage_id", "ts"], how="left")["_pos"]
        if pos.null_count():
            raise ValueError("prediction rows must be part of the full table")
        return pos.to_numpy().astype(np.int64)

    # ---------------------------------------------------------------- fit / predict
    def fit(
        self,
        train_full: pl.DataFrame,
        train_rows: pl.DataFrame,
        valid_full: pl.DataFrame,
        valid_rows: pl.DataFrame,
        target: str,
    ) -> SequenceTravelModel:
        torch.manual_seed(self.seed)
        dev = _device()
        q = torch.tensor(QUANTILES, dtype=torch.float32, device=dev)
        tr_full, tr_vids, tr_steps = self._prepare(train_full)
        va_full, va_vids, va_steps = self._prepare(valid_full)
        tr_rows, va_rows = self._rows(tr_full, train_rows), self._rows(va_full, valid_rows)
        xtr, ttr, str_ = self._tab(train_rows)
        xva, tva, sva = self._tab(valid_rows)
        self.scaler = _Scaler.fit(xtr)
        xtr, xva = self.scaler.transform(xtr), self.scaler.transform(xva)
        ytr = np.log1p(train_rows[target].to_numpy()).astype(np.float32)
        yva = np.log1p(valid_rows[target].to_numpy()).astype(np.float32)

        self.net = _Net(xtr.shape[1], self.use_sequence, dropout=self.dropout).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, bad = np.inf, 0
        # start from the initial weights: if validation never improves (or is NaN) we still
        # restore a valid state instead of crashing on None
        best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        for epoch in range(self.epochs):
            self.net.train()
            for *inp, y in self._batches(tr_steps, tr_vids, tr_rows, xtr, ttr, str_, ytr, True):
                opt.zero_grad()
                loss = _pinball(self.net(*inp), y, q)
                loss.backward()
                opt.step()
            val = self._loss(va_steps, va_vids, va_rows, xva, tva, sva, yva, q)
            self.history.append({"epoch": epoch, "valid_pinball_log": val})
            if val < best - 1e-4:
                best, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
            self.seed += 1  # new shuffle each epoch
        self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _loss(self, steps, vids, rows, x, t, s, y, q) -> float:
        self.net.eval()
        total, n = 0.0, 0
        for *inp, yb in self._batches(steps, vids, rows, x, t, s, y):
            total += _pinball(self.net(*inp), yb, q).item() * len(yb)
            n += len(yb)
        return total / n

    @torch.no_grad()
    def predict_log(self, rows_df: pl.DataFrame, full: pl.DataFrame | None = None) -> np.ndarray:
        """(n_rows, 7) quantiles of log1p(minutes). `full`: the table the windows come from."""
        self.net.eval()
        full_sorted, vids, steps = self._prepare(full if full is not None else rows_df)
        rows = self._rows(full_sorted, rows_df)
        x, t, s = self._tab(rows_df)
        x = self.scaler.transform(x)
        out = [self.net(*inp).cpu().numpy() for inp in self._batches(steps, vids, rows, x, t, s)]
        return np.concatenate(out) if out else np.empty((0, len(QUANTILES)))
