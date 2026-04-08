from __future__ import annotations
import numpy as np
import pandas as pd
from .config import PVConfig

def pv_power_ac(weather: pd.DataFrame, cfg: PVConfig) -> pd.Series:
    """Simple PV power model (PVWatts-like). Returns AC power in kW."""
    ghi = weather["ghi_wm2"].to_numpy()
    Tamb = weather["temp_c"].to_numpy()

    G = ghi  # simplification: POA ~= GHI

    Tcell = Tamb + (cfg.noct_c - 20.0) / 800.0 * G

    Pdc = cfg.dc_capacity_kwp * (G / 1000.0) * (1.0 + cfg.gamma_temp_coeff_per_c * (Tcell - 25.0))
    Pdc = np.maximum(0.0, Pdc)

    Pac = Pdc * cfg.inverter_efficiency * cfg.performance_ratio * (1.0 - cfg.shading_factor)
    Pac = np.clip(Pac, 0.0, cfg.dc_capacity_kwp)
    return pd.Series(Pac, index=weather.index, name="pv_kw")
