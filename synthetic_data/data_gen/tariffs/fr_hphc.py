from __future__ import annotations
import pandas as pd
from .base_tariff import TariffBase

class FRHPHC(TariffBase):
    def __init__(self, hp_hours: list[int], hc_hours: list[int], hp_price: float, hc_price: float):
        super().__init__()
        self.hp_hours = set(hp_hours)
        self.hc_hours = set(hc_hours)
        self.hp_price = float(hp_price)
        self.hc_price = float(hc_price)
        self.meta = {"type":"FR_HP_HC", "hp_hours": sorted(self.hp_hours), "hc_hours": sorted(self.hc_hours)}
    def price_at(self, ts: pd.Timestamp) -> float:
        return self.hp_price if ts.hour in self.hp_hours else self.hc_price
