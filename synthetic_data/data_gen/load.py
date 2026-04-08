from __future__ import annotations
import numpy as np
import pandas as pd
from .config import HouseConfig

def _daily_shape_vectorized(hour: np.ndarray, morning_peak_kw: float, evening_peak_kw: float) -> np.ndarray:
    m = morning_peak_kw * np.exp(-0.5*((hour-8.0)/2.0)**2)
    e = evening_peak_kw * np.exp(-0.5*((hour-20.0)/3.0)**2)
    base = 0.2
    return base + m + e

def generate_house_load(index: pd.DatetimeIndex, cfg: HouseConfig, rng: np.random.Generator) -> pd.Series:
    # Time-of-day
    hour = index.hour.to_numpy() + index.minute.to_numpy()/60.0
    shape = _daily_shape_vectorized(hour, cfg.morning_peak_kw, cfg.evening_peak_kw)

    # Step hours
    step_hours = (index[1] - index[0]).total_seconds()/3600.0
    steps_per_day = int(round(24.0/step_hours))

    # Normalize to daily kWh mean
    # First compute the shape for a single day at same sampling
    # Using 0..24 with steps_per_day samples
    per_day_hours = np.linspace(0, 24, steps_per_day, endpoint=False)
    per_day_shape = _daily_shape_vectorized(per_day_hours, cfg.morning_peak_kw, cfg.evening_peak_kw)
    per_day_shape_sum = per_day_shape.sum()

    # Energy per step (kWh) target for each timestamp follows shape proportion
    # But since the shape is the same for each day, we can scale globally by per_day_shape_sum
    energy_per_step_kwh = cfg.base_load_kwh_day_mean * (shape / per_day_shape_sum)

    # Convert to kW given step duration
    power_kw = energy_per_step_kwh / step_hours

    # Weekend multiplier
    is_weekend = (index.dayofweek >= 5)  # already a NumPy boolean array
    scale = np.where(is_weekend, cfg.weekend_multiplier, cfg.weekday_multiplier)
    power_kw = power_kw * scale

    # AR(1) noise
    phi = cfg.noise_ar1_phi
    sigma = cfg.noise_sigma_kw
    eps = rng.normal(0.0, sigma, size=len(index))
    noise = np.zeros(len(index))
    for t in range(1, len(index)):
        noise[t] = phi * noise[t-1] + eps[t]

    load_kw = power_kw + noise
    load_kw = np.clip(load_kw, 0.02, None)
    return pd.Series(load_kw, index=index, name="load_kw")
