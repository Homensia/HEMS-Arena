from __future__ import annotations
import numpy as np
import pandas as pd
from .base_tariff import TariffBase

class RandomPrice(TariffBase):
    def __init__(self, index: pd.DatetimeIndex, mean: float, std: float, min_price: float, max_price: float, seed: int = 42):
        super().__init__()
        self.index = index
        self.meta = {"type":"RANDOM"}
        rng = np.random.default_rng(seed)
        n = len(index)
        price = rng.normal(mean, std, size=n)
        price = np.clip(price, min_price, max_price)
        self.series = pd.Series(price, index=index, name="price_eur_per_kwh")
    def price_at(self, ts: pd.Timestamp) -> float:
        return float(self.series.loc[ts])
