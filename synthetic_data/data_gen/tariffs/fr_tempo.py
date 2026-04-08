from __future__ import annotations
import pandas as pd, numpy as np
from .base_tariff import TariffBase

class FRTEMPO(TariffBase):
    def __init__(self, index: pd.DatetimeIndex, blue_days: int, white_days: int, red_days: int,
                 price_hp_blue: float, price_hc_blue: float,
                 price_hp_white: float, price_hc_white: float,
                 price_hp_red: float, price_hc_red: float,
                 hp_hours: list[int] | None = None, hc_hours: list[int] | None = None, seed: int = 42):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.hp_hours = set(hp_hours if hp_hours is not None else list(range(6,22)))
        self.hc_hours = set(hc_hours if hc_hours is not None else list(range(22,24))+list(range(0,6)))
        days = pd.Index(index.normalize().unique())
        # Scale counts if needed
        total = blue_days + white_days + red_days
        if len(days) < total:
            scale = len(days)/total
            blue_days = max(0, int(round(blue_days*scale)))
            white_days = max(0, int(round(white_days*scale)))
            red_days = max(0, int(round(red_days*scale)))
        colors = ["blue"]*blue_days + ["white"]*white_days + ["red"]*red_days
        if len(colors) < len(days):
            colors += ["blue"] * (len(days)-len(colors))
        rng.shuffle(colors)
        self.day_color = {d: c for d, c in zip(days, colors)}
        self.prices = {
            "blue": (float(price_hp_blue), float(price_hc_blue)),
            "white": (float(price_hp_white), float(price_hc_white)),
            "red": (float(price_hp_red), float(price_hc_red)),
        }
        self.meta = {"type":"FR_TEMPO", "hp_hours": sorted(self.hp_hours), "hc_hours": sorted(self.hc_hours)}

    def price_at(self, ts: pd.Timestamp) -> float:
        color = self.day_color.get(ts.normalize(), "blue")
        hp, hc = self.prices[color]
        return hp if ts.hour in self.hp_hours else hc

    def color_of_day(self, ts: pd.Timestamp) -> str:
        return self.day_color.get(ts.normalize(), "blue")
