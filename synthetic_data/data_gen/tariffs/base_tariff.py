from __future__ import annotations
import pandas as pd

class TariffBase:
    def __init__(self):
        self.meta = {}
    def price_at(self, ts: pd.Timestamp) -> float:
        raise NotImplementedError
