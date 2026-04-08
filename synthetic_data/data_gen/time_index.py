from __future__ import annotations
import pandas as pd

def build_time_index(days: int, step_minutes: int, tz: str) -> pd.DatetimeIndex:
    """Create a timezone-aware, DST-safe DatetimeIndex starting today at 00:00 local time."""
    start = pd.Timestamp.now(tz=tz).normalize()
    freq = f"{step_minutes}min"
    periods = int(days * (1440 // step_minutes))
    idx = pd.date_range(start=start, periods=periods, freq=freq, tz=tz)
    return idx
