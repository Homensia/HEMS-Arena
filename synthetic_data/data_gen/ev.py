from __future__ import annotations
import numpy as np
import pandas as pd
from .config import EVFleetConfig

def _normal_hour_samples(rng, mean, std, size):
    s = rng.normal(mean, std, size=size)
    return np.mod(s, 24.0)

def generate_ev_timeseries(index: pd.DatetimeIndex, cfg: EVFleetConfig, price_resolver, rng: np.random.Generator) -> pd.DataFrame:
    """Per-house EV timeseries for cfg.evs_per_house EVs. Charging-only dataset.
    Policies:
      - 'immediate': charge upon arrival until target SOC.
      - 'hp_hc_aware': if tariff exposes HC hours, prioritize charging in HC windows when time allows.
    """
    step_hours = (index[1] - index[0]).total_seconds()/3600.0
    n = len(index)
    rows = []
    days = index.normalize().unique()

    # Meta for HP/HC hours, if any
    hc_hours = None
    if hasattr(price_resolver, "meta") and isinstance(price_resolver.meta, dict):
        if price_resolver.meta.get("type") in ("FR_HP_HC", "FR_TEMPO"):
            hc_hours = set(price_resolver.meta.get("hc_hours", []))

    for ev_id in range(1, cfg.evs_per_house+1):
        soc = np.zeros(n, dtype=float)
        present = np.zeros(n, dtype=int)
        chg_kw = np.zeros(n, dtype=float)
        driving_kwh = np.zeros(n, dtype=float)

        soc0 = float(np.clip(rng.normal(0.6, 0.1), 0.2, 0.9))
        soc[0] = soc0

        for d in days:
            mask = (index.normalize() == d)  

            idxs = np.where(mask)[0]
            if idxs.size == 0:
                continue

            # Trips
            dist = max(0.0, rng.normal(cfg.daily_trip_km_mean, cfg.daily_trip_km_std))
            e_drive = dist * cfg.kwh_per_km  # kWh per day

            # Arrival / departure
            arr_h = float(_normal_hour_samples(rng, cfg.arrival_hour_mean, cfg.arrival_hour_std, 1)[0])
            dep_h = float(_normal_hour_samples(rng, cfg.departure_hour_mean, cfg.departure_hour_std, 1)[0])

            hours = index[idxs].hour + index[idxs].minute/60.0  

            if dep_h < arr_h:
                present[idxs] = ((hours >= arr_h) | (hours < dep_h)).astype(int)
            else:
                present[idxs] = ((hours >= arr_h) & (hours < dep_h)).astype(int)

            away_mask = 1 - present[idxs]
            if away_mask.sum() > 0:
                per_step = e_drive / max(1, away_mask.sum())
                driving_kwh[idxs] = away_mask * per_step

            target = cfg.min_target_soc_at_departure

            for k, t in enumerate(idxs):
                # carry SOC from previous step minus driving
                if t > 0:
                    soc[t] = max(0.0, soc[t-1] - driving_kwh[t] / cfg.battery_kwh)
                # charging policy
                want_charge = soc[t] < target and present[t] == 1
                if want_charge:
                    if cfg.policy == "immediate":
                        power = cfg.charger_power_kw
                    elif cfg.policy == "hp_hc_aware" and hc_hours is not None:
                        hr = index[t].hour
                        power = cfg.charger_power_kw if hr in hc_hours else 0.0
                    else:
                        power = cfg.charger_power_kw
                    if power > 0.0:
                        delta = power * step_hours * cfg.charging_efficiency / cfg.battery_kwh
                        soc[t] = min(1.0, soc[t] + delta)
                        chg_kw[t] = power
                # else keep as is

        rows.append(pd.DataFrame({
            "soc": soc,
            "available_at_home": present,
            "charging_power_kw": chg_kw,
            "driving_kwh_step": driving_kwh,
            "ev_id": ev_id,
        }, index=index))

    return pd.concat(rows, axis=0, ignore_index=False)
