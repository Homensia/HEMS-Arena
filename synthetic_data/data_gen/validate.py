from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict

def validate_basic(time_index: pd.DatetimeIndex,
                   loads: dict[int, pd.Series],
                   pvs: dict[int, pd.Series],
                   prices: pd.Series,
                   evs: dict[tuple[int,int], pd.DataFrame]) -> dict:
    out: Dict[str, str | float | int | bool] = {}
    out["n_steps"] = len(time_index)
    out["all_indexes_aligned"] = all(s.index.equals(time_index) for s in list(loads.values()) + list(pvs.values()) + [prices])
    # PV checks
    pv_ok = True
    night_hours = (time_index.hour < 6) | (time_index.hour > 21)
    for s in pvs.values():
        if (s < -1e-6).any(): pv_ok = False
        if len(s[night_hours]) > 0 and s[night_hours].median() > 0.05: pv_ok = False
    out["pv_ok"] = pv_ok
    # Load non-negative
    out["load_nonneg"] = all((s >= 0.0).all() for s in loads.values())
    # EV SoC bounds
    soc_ok = True
    for df in evs.values():
        if (df["soc"] < -1e-6).any() or (df["soc"] > 1.0+1e-6).any():
            soc_ok = False
    out["ev_soc_ok"] = soc_ok
    # Prices finite
    out["prices_finite"] = prices.replace([np.inf, -np.inf], np.nan).notna().all()
    return out
