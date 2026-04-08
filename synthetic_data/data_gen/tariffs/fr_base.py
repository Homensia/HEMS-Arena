from __future__ import annotations
import pandas as pd
from .base_tariff import TariffBase

class FRBase(TariffBase):
    def __init__(self, price_eur_per_kwh: float):
        super().__init__()
        self.p = float(price_eur_per_kwh)
        self.meta = {"type": "FR_BASE"}
    def price_at(self, ts: pd.Timestamp) -> float:
        return self.p
