from __future__ import annotations
import numpy as np
import pandas as pd
from .base_tariff import TariffBase

class DynamicSpot(TariffBase):
    def __init__(self, index: pd.DatetimeIndex, mean_price: float, daily_amp: float,
                 ar1_phi: float, noise_sigma: float, spike_prob: float, spike_amp: float,
                 allow_negative: bool, seed: int = 42):
        super().__init__()
        self.index = index
        self.meta = {"type":"DYNAMIC"}
        rng = np.random.default_rng(seed)
        n = len(index)
        hour = index.hour.to_numpy() + index.minute.to_numpy()/60.0
        base = mean_price + daily_amp * np.sin(2*np.pi*(hour-16)/24.0)
        noise = np.zeros(n)
        eps = rng.normal(0.0, noise_sigma, size=n)
        for t in range(1, n):
            noise[t] = ar1_phi * noise[t-1] + eps[t]
        price = base + noise
        spikes = rng.random(n) < spike_prob
        price = price + spikes * (rng.choice([-1,1], size=n) * spike_amp)
        if not allow_negative:
            price = np.maximum(price, 0.0)
        self.series = pd.Series(price, index=index, name="price_eur_per_kwh")

    def price_at(self, ts: pd.Timestamp) -> float:
        return float(self.series.loc[ts])
