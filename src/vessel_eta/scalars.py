"""Typed access to polars scalars.

`Series.max()`, `.mean()` and friends return "any Python literal or None", which is
true but useless to a type checker. These helpers check at runtime and narrow the type.
"""

from __future__ import annotations

import datetime as dt
from typing import Any


def as_float(value: Any) -> float:
    """Numeric scalar -> float; None (empty or all-null) -> NaN."""
    if value is None:
        return float("nan")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"expected a number, got {type(value).__name__}")
    return float(value)


def as_datetime(value: Any) -> dt.datetime:
    if not isinstance(value, dt.datetime):
        raise TypeError(f"expected a datetime, got {type(value).__name__}")
    return value


def as_date(value: Any) -> dt.date:
    if not isinstance(value, dt.date):
        raise TypeError(f"expected a date, got {type(value).__name__}")
    return value
