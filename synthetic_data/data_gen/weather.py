from __future__ import annotations
import numpy as np
import pandas as pd
from .config import WeatherConfig

def seasonal_factor(doy: np.ndarray) -> np.ndarray:
    return 0.65 + 0.35 * np.sin(2*np.pi*(doy-80)/365.25)

def diurnal_solar_elevation_factor(hours: np.ndarray) -> np.ndarray:
    phi = (hours - 13.0) / 12.0 * np.pi
    f = np.sin(np.clip(phi, -np.pi/2, np.pi/2))
    return np.maximum(0.0, f)

def generate_weather(index: pd.DatetimeIndex, cfg: WeatherConfig, rng: np.random.Generator) -> pd.DataFrame:
    doy = index.day_of_year.to_numpy()
    hour = index.hour.to_numpy() + index.minute.to_numpy()/60.0

    season = cfg.ambient_temp_mean_c + cfg.ambient_temp_amp_c * np.sin(2*np.pi*(doy-172)/365.25)
    diurnal = cfg.daily_temp_amp_c * np.sin(2*np.pi*(hour-15)/24.0)
    temp = season + diurnal + rng.normal(0.0, cfg.noise_std_c, size=len(index))

    ghi_cs = 1000.0 * seasonal_factor(doy) * diurnal_solar_elevation_factor(hour)

    levels = np.array(cfg.cloud_markov_levels, dtype=float)
    T = np.array(cfg.cloud_transition, dtype=float)
    state = 0
    cloud_mult = np.empty(len(index))
    for t in range(len(index)):
        cloud_mult[t] = levels[state]
        state = rng.choice([0,1,2], p=T[state])

    ghi = ghi_cs * cloud_mult

    return pd.DataFrame({"temp_c": temp, "ghi_wm2": ghi}, index=index)
