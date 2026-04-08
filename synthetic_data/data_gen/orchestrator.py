from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .config import ScenarioConfig, FRBaseTariff, FRHPHCTariff, FRTEMPOConfig, DynamicTariffConfig, RandomTariffConfig
from .time_index import build_time_index
from .weather import generate_weather
from .pv import pv_power_ac
from .load import generate_house_load
from .ev import generate_ev_timeseries
from .validate import validate_basic
from .persist import persist_scenario
from .tariffs.fr_base import FRBase
from .tariffs.fr_hphc import FRHPHC
from .tariffs.fr_tempo import FRTEMPO
from .tariffs.dynamic_spot import DynamicSpot
from .tariffs.random_price import RandomPrice

def _resolve_tariff(index: pd.DatetimeIndex, cfg) -> Tuple[object, pd.Series]:
    if isinstance(cfg, FRBaseTariff):
        t = FRBase(cfg.price_eur_per_kwh)
        prices = pd.Series([t.price_at(ts) for ts in index], index=index, name="price_eur_per_kwh")
        return t, prices
    elif isinstance(cfg, FRHPHCTariff):
        t = FRHPHC(cfg.hp_hours, cfg.hc_hours, cfg.hp_price, cfg.hc_price)
        prices = pd.Series([t.price_at(ts) for ts in index], index=index, name="price_eur_per_kwh")
        return t, prices
    elif isinstance(cfg, FRTEMPOConfig):
        t = FRTEMPO(index, cfg.calendar.blue_days, cfg.calendar.white_days, cfg.calendar.red_days,
                    cfg.price_hp_blue, cfg.price_hc_blue,
                    cfg.price_hp_white, cfg.price_hc_white,
                    cfg.price_hp_red, cfg.price_hc_red)
        prices = pd.Series([t.price_at(ts) for ts in index], index=index, name="price_eur_per_kwh")
        return t, prices
    elif isinstance(cfg, DynamicTariffConfig):
        t = DynamicSpot(index, cfg.mean_price, cfg.daily_amp, cfg.ar1_phi, cfg.noise_sigma, cfg.spike_prob, cfg.spike_amp, cfg.allow_negative)
        return t, t.series
    elif isinstance(cfg, RandomTariffConfig):
        t = RandomPrice(index, cfg.mean, cfg.std, cfg.min_price, cfg.max_price)
        return t, t.series
    else:
        raise ValueError("Unknown tariff type")

def run_scenario(config: ScenarioConfig):
    rng = np.random.default_rng(config.seed)
    index = build_time_index(config.days, config.step_minutes, config.timezone)

    weather = generate_weather(index, config.weather, rng)

    tariff_obj, price_series = _resolve_tariff(index, config.tariff)

    loads: Dict[int, pd.Series] = {}
    pvs: Dict[int, pd.Series] = {}
    evs: Dict[tuple[int,int], pd.DataFrame] = {}

    for house_id in range(1, config.n_houses+1):
        loads[house_id] = generate_house_load(index, config.house, rng)
        pvs[house_id] = pv_power_ac(weather, config.pv)
        ev_df = generate_ev_timeseries(index, config.ev, tariff_obj, rng)
        for ev_id, sub in ev_df.groupby("ev_id"):
            evs[(house_id, int(ev_id))] = sub.drop(columns=["ev_id"])

    val = validate_basic(index, loads, pvs, price_series, evs)

    meta = {
        "config": config.model_dump(mode="json"),
        "seed": config.seed,
        "generator_version": "0.1.0",
        "validation": val,
    }
    out_dir = persist_scenario(config.output_dir, config.scenario_name, index, loads, pvs, price_series, evs, meta)
    return out_dir, val
