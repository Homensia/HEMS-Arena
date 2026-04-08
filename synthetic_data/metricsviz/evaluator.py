from __future__ import annotations
"""
Comprehensive evaluator for MATRIX synthetic dataset scenarios.
Computes quality checks, descriptive statistics, KPIs, correlations, EV and price metrics.
"""
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ------------------------ Helpers ------------------------

def _step_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    return (index[1] - index[0]).total_seconds() / 3600.0


def _autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Unbiased normalized autocorrelation 0..max_lag.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    n = len(x)
    ac = np.zeros(max_lag + 1)
    denom = np.nansum(x * x)
    if denom == 0 or np.isnan(denom):
        return ac
    for k in range(max_lag + 1):
        a = x[: n - k]
        b = x[k:]
        num = np.nansum(a * b)
        ac[k] = num / denom
    return ac


def _series_quality(index: pd.DatetimeIndex, s: pd.Series) -> Dict[str, Any]:
    null_count = int(s.isna().sum())
    duplicated_ts = int(index.duplicated().sum())
    monotonic = bool(index.is_monotonic_increasing)
    return {
        "null_count": null_count,
        "null_fraction": float(null_count) / max(1, len(s)),
        "duplicates_in_index": duplicated_ts,
        "index_monotonic_increasing": monotonic,
    }


def series_stats(s: pd.Series) -> Dict[str, Any]:
    s_clean = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s_clean) == 0:
        return {
            "count": 0, "mean": float("nan"), "std": float("nan"),
            "min": float("nan"), "max": float("nan"),
            "q01": float("nan"), "q05": float("nan"), "q25": float("nan"),
            "q50": float("nan"), "q75": float("nan"), "q95": float("nan"), "q99": float("nan"),
            "zero_fraction": float("nan"), "negative_fraction": float("nan"), "positive_fraction": float("nan"),
        }
    q = s_clean.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()
    return {
        "count": int(s.count()),
        "mean": float(s_clean.mean()),
        "std": float(s_clean.std()),
        "min": float(s_clean.min()),
        "max": float(s_clean.max()),
        "q01": float(q.get(0.01, np.nan)),
        "q05": float(q.get(0.05, np.nan)),
        "q25": float(q.get(0.25, np.nan)),
        "q50": float(q.get(0.50, np.nan)),
        "q75": float(q.get(0.75, np.nan)),
        "q95": float(q.get(0.95, np.nan)),
        "q99": float(q.get(0.99, np.nan)),
        "zero_fraction": float((s_clean == 0).mean()),
        "negative_fraction": float((s_clean < 0).mean()),
        "positive_fraction": float((s_clean > 0).mean()),
    }


def hourly_profile(s: pd.Series) -> pd.Series:
    return s.groupby(s.index.hour).mean()


def weekday_weekend_profiles(s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Return average profiles by hour-of-day for weekday and weekend separately."""
    wd_s = s[s.index.dayofweek < 5]
    we_s = s[s.index.dayofweek >= 5]
    wd = wd_s.groupby(wd_s.index.hour).mean()
    we = we_s.groupby(we_s.index.hour).mean()
    return wd, we



def daily_energy(s_kw: pd.Series) -> pd.Series:
    step_h = _step_hours(s_kw.index)
    return (s_kw * step_h).resample("D").sum()


# ------------------------ Metrics ------------------------

def kpis_house(load_kw: pd.Series, pv_kw: pd.Series, price_eur_per_kwh: pd.Series) -> Dict[str, Any]:
    idx = load_kw.index
    step_h = _step_hours(idx)

    pv_kwh = float((pv_kw * step_h).sum())
    load_kwh = float((load_kw * step_h).sum())

    net_from_grid_kw = (load_kw - pv_kw).clip(lower=0.0)
    net_to_grid_kw = (pv_kw - load_kw).clip(lower=0.0)

    import_kwh = float((net_from_grid_kw * step_h).sum())
    export_kwh = float((net_to_grid_kw * step_h).sum())

    price_aligned = price_eur_per_kwh.reindex(idx).ffill()
    cost_eur = float((net_from_grid_kw * step_h * price_aligned).sum())

    self_consumption_ratio = float(((pv_kw - net_to_grid_kw) * step_h).sum() / (pv_kwh + 1e-9))
    self_sufficiency_ratio = float(((load_kw - net_from_grid_kw) * step_h).sum() / (load_kwh + 1e-9))

    pv_cap = pv_kw.max() if pv_kw.max() > 0 else 1.0
    pv_capacity_factor = float(pv_kwh / (pv_cap * 8760.0)) if len(idx) >= 8760 and pv_cap > 0 else float("nan")
    load_factor = float(load_kw.mean() / (load_kw.max() + 1e-9))

    ramps = load_kw.diff().abs().dropna()
    ramp95 = float(ramps.quantile(0.95)) if len(ramps) else float("nan")

    max_lag = min(len(load_kw) - 1, int(round(24.0 / step_h))) if len(load_kw) > 1 else 0
    acf = _autocorr(load_kw.to_numpy(), max_lag=max_lag) if max_lag > 0 else np.array([1.0])
    acf_24h = float(acf[max_lag]) if max_lag < len(acf) else float("nan")

    return {
        "energy": {
            "total_load_kwh": load_kwh,
            "total_pv_kwh": pv_kwh,
            "import_kwh": import_kwh,
            "export_kwh": export_kwh,
        },
        "peaks": {
            "peak_load_kw": float(load_kw.max()),
            "peak_pv_kw": float(pv_kw.max()),
            "peak_net_from_grid_kw": float(net_from_grid_kw.max()),
        },
        "ratios": {
            "self_consumption_ratio": self_consumption_ratio,
            "self_sufficiency_ratio": self_sufficiency_ratio,
            "load_factor": load_factor,
            "pv_capacity_factor": pv_capacity_factor,
        },
        "costs": {
            "total_cost_eur": cost_eur,
            "avg_price_eur_per_kwh": float(price_eur_per_kwh.mean()),
        },
        "ramping": {
            "ramp95_kw_per_step": ramp95
        },
        "autocorr": {
            "acf_24h": acf_24h
        }
    }


def cost_breakdown_hp_hc(price_series: pd.Series, load_kw: pd.Series, pv_kw: pd.Series,
                         hp_hours: Optional[List[int]], hc_hours: Optional[List[int]]) -> Dict[str, Any]:
    """Compute import energy and cost split by HP/HC if hours are provided."""
    if not hp_hours or not hc_hours:
        return {"hp": None, "hc": None}

    # Ensure datetime index
    idx = pd.to_datetime(price_series.index, errors="coerce")
    price_dt = pd.Series(price_series.values, index=idx).dropna()
    step_h = _step_hours(price_dt.index)

    # Align and compute net import
    net_from_grid_kw = (load_kw - pv_kw).clip(lower=0.0)
    aligned = pd.DataFrame({
        "import_kw": net_from_grid_kw.reindex(price_dt.index).fillna(0.0),
        "price": price_dt
    })
    # Now safe to use .hour
    aligned["hour"] = aligned.index.hour

    hp = aligned[aligned["hour"].isin(hp_hours)]
    hc = aligned[aligned["hour"].isin(hc_hours)]

    return {
        "hp": {
            "import_kwh": float((hp["import_kw"] * step_h).sum()),
            "cost_eur": float((hp["import_kw"] * step_h * hp["price"]).sum()),
        },
        "hc": {
            "import_kwh": float((hc["import_kw"] * step_h).sum()),
            "cost_eur": float((hc["import_kw"] * step_h * hc["price"]).sum()),
        },
    }


def ev_metrics(ev_df: pd.DataFrame, target_soc: Optional[float] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    idx = ev_df.index
    step_h = _step_hours(idx)

    avail_frac = float(ev_df["available_at_home"].mean())
    charge_events = (ev_df["charging_power_kw"] > 0).astype(int).diff().clip(lower=0).sum()
    avg_chg_power = float(ev_df.loc[ev_df["charging_power_kw"] > 0, "charging_power_kw"].mean()) if (ev_df["charging_power_kw"] > 0).any() else 0.0
    chg_energy_kwh = float((ev_df["charging_power_kw"] * step_h).sum())

    soc_stats = series_stats(ev_df["soc"])
    if target_soc is not None:
        soc_violations = float((ev_df["soc"] < (target_soc - 1e-6)).mean())
    else:
        soc_violations = float("nan")

    out.update({
        "availability_fraction": avail_frac,
        "charging_sessions_count": int(charge_events),
        "avg_charging_power_kw": avg_chg_power,
        "total_charging_energy_kwh": chg_energy_kwh,
        "soc": soc_stats,
        "soc_below_target_fraction": soc_violations
    })
    return out


def price_metrics(price: pd.Series) -> Dict[str, Any]:
    s = price.replace([np.inf, -np.inf], np.nan).dropna()
    z = (s - s.mean()) / (s.std() + 1e-9)
    spikes = int((np.abs(z) > 3).sum())
    neg_hours = int((s < 0).sum())
    out = series_stats(s)
    out.update({
        "spike_count_z3": spikes,
        "negative_price_hours": neg_hours
    })
    return out


def correlation_metrics(load_kw: pd.Series, pv_kw: pd.Series, price: pd.Series) -> Dict[str, Any]:
    aligned = pd.concat([load_kw, pv_kw, price], axis=1).dropna()
    aligned.columns = ["load_kw", "pv_kw", "price"]
    corr = aligned.corr()
    return {
        "corr_load_pv": float(corr.loc["load_kw", "pv_kw"]),
        "corr_load_price": float(corr.loc["load_kw", "price"]),
        "corr_pv_price": float(corr.loc["pv_kw", "price"]),
    }


# ------------------------ IO ------------------------

def load_scenario(scen_dir: Path) -> Dict[str, Any]:
    """
    Load all CSVs for a scenario into memory.
    Assumes file names like:
      <scenario>_house_elec_<N>_<house>.csv
      <scenario>_pv_generation_<N>_<house>.csv
      <scenario>_price_<N>.csv
      <scenario>_ev_<N>_<house>_<ev>.csv
    """
    csv_dir = Path(scen_dir) / "csv"
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV folder not found: {csv_dir}")

    price_files = list(csv_dir.glob("*_price_*.csv"))
    if not price_files:
        raise FileNotFoundError("No price file found.")
    price_path = price_files[0]
    parts = price_path.stem.split("_")
    scenario_name = "_".join(parts[:-2])
    scen_num = parts[-1]

    price = pd.read_csv(price_path, index_col=0, parse_dates=True).iloc[:, 0]
    price.index = pd.to_datetime(price.index, errors="coerce")   # force datetime
    price = price[price.index.notna()].sort_index()


    loads: Dict[int, pd.Series] = {}
    pvs: Dict[int, pd.Series] = {}
    evs: Dict[Tuple[int, int], pd.DataFrame] = {}

    for f in csv_dir.glob(f"{scenario_name}_house_elec_{scen_num}_*.csv"):
        h = int(f.stem.split("_")[-1])
        s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
        s.index = pd.to_datetime(s.index, errors="coerce")       # force datetime
        s = s[s.index.notna()].sort_index()
        loads[h] = s

    for f in csv_dir.glob(f"{scenario_name}_pv_generation_{scen_num}_*.csv"):
        h = int(f.stem.split("_")[-1])
        s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
        s.index = pd.to_datetime(s.index, errors="coerce")       # force datetime
        s = s[s.index.notna()].sort_index()
        pvs[h] = s

    for f in csv_dir.glob(f"{scenario_name}_ev_{scen_num}_*.csv"):
        toks = f.stem.split("_")
        house = int(toks[-2])
        ev = int(toks[-1])
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, errors="coerce")     # force datetime
        df = df[df.index.notna()].sort_index()
        evs[(house, ev)] = df

    metadata = None
    meta_path = Path(scen_dir) / "metadata.json"
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = None

    return {
        "scenario_name": scenario_name,
        "scenario_number": scen_num,
        "price": price,
        "loads": loads,
        "pvs": pvs,
        "evs": evs,
        "metadata": metadata,
    }


def evaluate_scenario(scen_dir: Path, hp_hours: Optional[List[int]] = None, hc_hours: Optional[List[int]] = None) -> Dict[str, Any]:
    data = load_scenario(scen_dir)
    price = data["price"]
    loads: Dict[int, pd.Series] = data["loads"]
    pvs: Dict[int, pd.Series] = data["pvs"]
    evs: Dict[Tuple[int, int], pd.DataFrame] = data["evs"]
    meta = data["metadata"] or {}

    if not hp_hours and not hc_hours:
        try:
            tmeta = meta.get("config", {}).get("tariff", {})
            if isinstance(tmeta, dict) and tmeta.get("type") in ("FR_HP_HC", "FR_TEMPO"):
                hp_hours = tmeta.get("hp_hours")
                hc_hours = tmeta.get("hc_hours")
        except Exception:
            pass

    overall: Dict[str, Any] = {}
    houses: Dict[str, Any] = {}

    overall["price_stats"] = price_metrics(price)
    overall["price_quality"] = _series_quality(price.index, price)

    for h in sorted(loads.keys()):
        load_kw = loads[h]
        pv_kw = pvs.get(h)
        if pv_kw is None:
            continue

        q_load = _series_quality(load_kw.index, load_kw)
        q_pv = _series_quality(pv_kw.index, pv_kw)
        st_load = series_stats(load_kw)
        st_pv = series_stats(pv_kw)
        k = kpis_house(load_kw, pv_kw, price)
        cor = correlation_metrics(load_kw, pv_kw, price)
        hp_hc = cost_breakdown_hp_hc(price, load_kw, pv_kw, hp_hours, hc_hours)
        prof_load = hourly_profile(load_kw).to_dict()
        prof_pv = hourly_profile(pv_kw).to_dict()
        wd_load, we_load = weekday_weekend_profiles(load_kw)
        wd_pv, we_pv = weekday_weekend_profiles(pv_kw)
        ev_info = {}
        for (hh, ev) in evs.keys():
            if hh != h:
                continue
            ev_df = evs[(hh, ev)]
            tgt = None
            try:
                cfg = meta.get("config", {}).get("ev", {})
                tgt = cfg.get("min_target_soc_at_departure", None)
            except Exception:
                pass
            ev_info[str(ev)] = ev_metrics(ev_df, target_soc=tgt)

        houses[str(h)] = {
            "quality": {"load": q_load, "pv": q_pv},
            "stats": {"load": st_load, "pv": st_pv},
            "kpis": k,
            "correlations": cor,
            "hp_hc_cost_breakdown": hp_hc,
            "profiles": {
                "hourly": {"load": prof_load, "pv": prof_pv},
                "weekday_weekend": {
                    "weekday": {"load": wd_load.to_dict(), "pv": wd_pv.to_dict()},
                    "weekend": {"load": we_load.to_dict(), "pv": we_pv.to_dict()},
                },
            },
            "evs": ev_info,
        }

    overall["houses"] = houses
    return overall


# ====================== DEEP DIVE ANALYTICS ======================

def _combined_dataframe(price: pd.Series,
                        loads: Dict[int, pd.Series],
                        pvs: Dict[int, pd.Series],
                        evs: Dict[Tuple[int, int], pd.DataFrame]) -> pd.DataFrame:
    """
    Build a wide DataFrame with:
      - price (€/kWh)
      - per-house load columns:  load_h{h}
      - per-house pv columns:    pv_h{h}
      - per-house EV total charging power: evchg_h{h}
    All aligned on the union time index and forward-filled where needed.
    """
    cols = {}
    cols["price"] = price
    for h, s in loads.items():
        cols[f"load_h{h}"] = s
    for h, s in pvs.items():
        cols[f"pv_h{h}"] = s
    # sum EV charging power per house
    ev_house = {}
    for (h, ev), df in evs.items():
        ev_house.setdefault(h, 0)
    for h in loads.keys():
        ev_sum = None
        for (hh, ev), df in evs.items():
            if hh != h:
                continue
            s = df["charging_power_kw"]
            ev_sum = s if ev_sum is None else ev_sum.add(s, fill_value=0)
        if ev_sum is None:
            # no EVs for this house => zeros aligned with load index
            ev_sum = loads[h].copy() * 0.0
        cols[f"evchg_h{h}"] = ev_sum

    df = pd.DataFrame(cols)
    # ensure datetime index, sorted, and ffill minimal gaps
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    df = df.ffill()
    return df


def full_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation matrix across all available columns.
    """
    return df.corr()


def house_to_house_correlation(loads: Dict[int, pd.Series]) -> pd.DataFrame:
    """
    Correlation matrix between house loads only (columns: h1,h2,...).
    """
    if not loads:
        return pd.DataFrame()
    df = pd.DataFrame({f"h{h}": s for h, s in loads.items()})
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index().ffill()
    return df.corr()


def house_to_house_pv_correlation(pvs: Dict[int, pd.Series]) -> pd.DataFrame:
    if not pvs:
        return pd.DataFrame()
    df = pd.DataFrame({f"h{h}": s for h, s in pvs.items()})
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index().ffill()
    return df.corr()


def cross_correlation(a: pd.Series, b: pd.Series, max_lag_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symmetric cross-correlation at lags [-max_lag_steps .. +max_lag_steps].
    Returns (lags, ccf) where lags are integers in steps.
    """
    # align
    ab = pd.concat([a, b], axis=1).dropna()
    if ab.empty:
        return np.arange(-max_lag_steps, max_lag_steps+1), np.zeros(2*max_lag_steps+1)
    x = ab.iloc[:,0].to_numpy(dtype=float)
    y = ab.iloc[:,1].to_numpy(dtype=float)
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    denom = np.sqrt(np.nansum(x*x) * np.nansum(y*y)) + 1e-12

    def corr_at_lag(k):
        if k >= 0:
            return np.nansum(x[k:] * y[:len(y)-k]) / denom
        else:
            k = -k
            return np.nansum(x[:len(x)-k] * y[k:]) / denom

    lags = np.arange(-max_lag_steps, max_lag_steps+1)
    ccf = np.array([corr_at_lag(k) for k in lags])
    return lags, ccf


def monthly_hourly_profile(s: pd.Series) -> pd.DataFrame:
    """
    Matrix (index=hour 0..23, columns=1..12) with mean value for each (month, hour).
    """
    df = s.copy().to_frame("v")
    df["month"] = df.index.month
    df["hour"] = df.index.hour
    prof = df.pivot_table(index="hour", columns="month", values="v", aggfunc="mean")
    # ensure full axes
    for m in range(1, 13):
        if m not in prof.columns:
            prof[m] = np.nan
    prof = prof.sort_index(axis=1)
    return prof


def spectral_fft(s: pd.Series, step_hours: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple periodogram-style FFT. Returns (period_hours, power).
    Only positive frequencies are returned (excluding zero).
    """
    x = s.to_numpy(dtype=float)
    x = x - np.nanmean(x)
    n = len(x)
    if n < 8:
        return np.array([]), np.array([])
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=step_hours)  # cycles per hour
    power = (np.abs(X)**2) / n
    # avoid the zero frequency (mean)
    if len(freqs) > 1:
        freqs = freqs[1:]
        power = power[1:]
    # convert to period (hours) = 1/f
    with np.errstate(divide='ignore', invalid='ignore'):
        period_h = np.where(freqs > 0, 1.0 / freqs, np.inf)
    return period_h, power


def price_elasticity_signals(price: pd.Series, load_kw: pd.Series, pv_kw: pd.Series) -> pd.Series:
    """
    Compute net import from grid (kW) aligned to price.
    """
    net_from_grid_kw = (load_kw - pv_kw).clip(lower=0.0)
    return net_from_grid_kw.reindex(price.index).ffill()


def ev_arrival_departure_hours(ev_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Infer arrival (0->1) and departure (1->0) transitions from 'available_at_home'.
    Return arrays of hours (0..23) for arrivals and departures.
    """
    a = ev_df["available_at_home"].astype(int)
    d = a.diff()
    arrivals = ev_df.index[d == 1].hour.to_numpy()
    departures = ev_df.index[d == -1].hour.to_numpy()
    return arrivals, departures
