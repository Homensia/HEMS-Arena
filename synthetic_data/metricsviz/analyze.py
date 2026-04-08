from __future__ import annotations
"""
Enhanced Analysis CLI: runs comprehensive analysis over a scenario and saves metrics + plots.
Adds:
- Full correlation matrices (all variables, house-to-house load & PV)
- Cross-correlation with lags (load↔price, load↔PV, PV↔price)
- Monthly×hour profiles (heatmaps)
- Spectral (FFT) analysis (dominant periods)
- EV arrival/departure hour histograms
- Tariff elasticity proxy via price↔net-import cross-correlation
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluator import (
    evaluate_scenario, load_scenario, _step_hours, _autocorr,
    _combined_dataframe, full_correlation_matrix,
    house_to_house_correlation, house_to_house_pv_correlation,
    cross_correlation, monthly_hourly_profile, spectral_fft,
    price_elasticity_signals, ev_arrival_departure_hours
)
from .plots import (
    plot_timeseries, plot_histogram, plot_boxplot, plot_duration_curve,
    plot_hourly_profile, plot_heatmap_day_hour, plot_line, plot_autocorr,
    plot_corr_heatmap, plot_month_hour_matrix, plot_fft, plot_crosscorr
)


def _to_native(o):
    import numpy as np
    if isinstance(o, dict):
        return {k: _to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_to_native(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, pd.DataFrame):
        return o.to_dict()
    return o


def analyze_scenario(scen_dir: str) -> Path:
    scen = Path(scen_dir)
    data = load_scenario(scen)
    price = data["price"]
    loads = data["loads"]
    pvs = data["pvs"]
    evs = data["evs"]
    meta = data["metadata"] or {}

    reports = scen / "reports"
    plots = reports / "plots"
    reports.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    # ---------------- Basic metrics (existing) ----------------
    metrics = evaluate_scenario(scen)
    (reports / "summary_metrics.json").write_text(json.dumps(_to_native(metrics), indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- Standard plots per house ----------------
    for h in sorted(loads.keys()):
        load_kw = loads[h]
        pv_kw = pvs.get(h)
        if pv_kw is None:
            continue

        first_week_len = int(round(24/_step_hours(load_kw.index))) * 7
        df_first = pd.DataFrame({"load_kw": load_kw.iloc[:first_week_len],
                                 "pv_kw": pv_kw.iloc[:first_week_len]})
        plot_timeseries(df_first, plots / f"house{h}_first_week_timeseries.png", f"House {h} - First week Load/PV", "kW")

        df_all = pd.DataFrame({"load_kw": load_kw, "pv_kw": pv_kw})
        plot_timeseries(df_all, plots / f"house{h}_full_timeseries.png", f"House {h} - Full Load/PV", "kW")

        plot_duration_curve(load_kw, plots / f"house{h}_load_duration.png", f"House {h} - Load duration curve", "kW")
        plot_duration_curve(pv_kw, plots / f"house{h}_pv_duration.png", f"House {h} - PV duration curve", "kW")

        plot_histogram(load_kw, plots / f"house{h}_load_hist.png", 60, f"House {h} - Load histogram", "kW")
        plot_histogram(pv_kw, plots / f"house{h}_pv_hist.png", 60, f"House {h} - PV histogram", "kW")

        prof_load = load_kw.groupby(load_kw.index.hour).mean()
        prof_pv = pv_kw.groupby(pv_kw.index.hour).mean()
        plot_hourly_profile(prof_load, plots / f"house{h}_load_hourly_profile.png", f"House {h} - Load hourly profile", "kW")
        plot_hourly_profile(prof_pv, plots / f"house{h}_pv_hourly_profile.png", f"House {h} - PV hourly profile", "kW")

        wd = load_kw[load_kw.index.dayofweek < 5].groupby(load_kw[load_kw.index.dayofweek < 5].index.hour).mean()
        we = load_kw[load_kw.index.dayofweek >= 5].groupby(load_kw[load_kw.index.dayofweek >= 5].index.hour).mean()
        plot_boxplot([wd, we], ["Weekday", "Weekend"], plots / f"house{h}_load_weekday_weekend_box.png",
                     f"House {h} - Load weekday/weekend hour means", "kW")

        plot_heatmap_day_hour(load_kw, plots / f"house{h}_load_heatmap.png", f"House {h} - Load day x hour")
        plot_heatmap_day_hour(pv_kw, plots / f"house{h}_pv_heatmap.png", f"House {h} - PV day x hour")

        # Autocorr
        step_h = _step_hours(load_kw.index)
        max_lag = min(len(load_kw)-1, int(round(24.0/step_h)))
        acf = _autocorr(load_kw.to_numpy(), max_lag=max_lag)
        plot_autocorr(acf, step_h, plots / f"house{h}_load_acf.png", f"House {h} - Load autocorrelation (<=24h)")

        # Net load
        net = (load_kw - pv_kw)
        plot_histogram(net, plots / f"house{h}_net_load_hist.png", 60, f"House {h} - Net load histogram", "kW")
        plot_timeseries(pd.DataFrame({"net_kw": net.iloc[:first_week_len]}), plots / f"house{h}_net_first_week.png",
                        f"House {h} - Net load (first week)", "kW")

        # ---------------- Deep additions per house ----------------
        # Cross-correlation (±24h) for load↔PV, load↔price, PV↔price
        lag_steps = int(round(24.0/step_h))
        lags, ccf_lp = cross_correlation(load_kw, price.reindex(load_kw.index).ffill(), lag_steps)
        plot_crosscorr(lags*step_h, ccf_lp, plots / f"house{h}_ccf_load_price.png", f"House {h} - Cross-corr load↔price (±24h)")
        lags, ccf_lpv = cross_correlation(load_kw, pv_kw, lag_steps)
        plot_crosscorr(lags*step_h, ccf_lpv, plots / f"house{h}_ccf_load_pv.png", f"House {h} - Cross-corr load↔pv (±24h)")
        lags, ccf_pp = cross_correlation(pv_kw, price.reindex(pv_kw.index).ffill(), lag_steps)
        plot_crosscorr(lags*step_h, ccf_pp, plots / f"house{h}_ccf_pv_price.png", f"House {h} - Cross-corr pv↔price (±24h)")

        # Monthly×hour profiles
        mh_load = monthly_hourly_profile(load_kw)
        mh_pv   = monthly_hourly_profile(pv_kw)
        plot_month_hour_matrix(mh_load, plots / f"house{h}_load_month_hour.png", f"House {h} - Load month×hour")
        plot_month_hour_matrix(mh_pv,   plots / f"house{h}_pv_month_hour.png",   f"House {h} - PV month×hour")

        # FFT spectra (dominant periods)
        ph, pw = spectral_fft(load_kw, step_h)
        plot_fft(ph, pw, plots / f"house{h}_fft_load.png", f"House {h} - FFT Load")
        ph, pw = spectral_fft(pv_kw, step_h)
        plot_fft(ph, pw, plots / f"house{h}_fft_pv.png", f"House {h} - FFT PV")

    # Price plots
    first_week_len = int(round(24/_step_hours(price.index))) * 7
    plot_timeseries(price.iloc[:first_week_len].to_frame("price_eur_per_kwh"), plots / "price_first_week.png", "Price - First week", "€/kWh")
    plot_histogram(price, plots / "price_hist.png", 60, "Price histogram", "€/kWh")

    # EV plots per EV + arrivals/departures
    ev_arr_dep_summary = {}
    for (h, ev), ev_df in evs.items():
        plot_histogram(ev_df["charging_power_kw"], plots / f"house{h}_ev{ev}_charging_hist.png", 40, f"House {h} EV{ev} - Charging power", "kW")
        plot_histogram(ev_df["soc"], plots / f"house{h}_ev{ev}_soc_hist.png", 40, f"House {h} EV{ev} - SoC", "SoC")
        avail_prof = ev_df["available_at_home"].groupby(ev_df.index.hour).mean()
        plot_hourly_profile(avail_prof, plots / f"house{h}_ev{ev}_availability_hourly.png", f"House {h} EV{ev} - Availability by hour", "Fraction")
        step_h = _step_hours(ev_df.index)
        first_week_len = int(round(24/step_h))*7
        plot_timeseries(ev_df.iloc[:first_week_len][["charging_power_kw"]], plots / f"house{h}_ev{ev}_charging_first_week.png", f"House {h} EV{ev} - Charging (first week)", "kW")
        # arrivals/departures
        arr_h, dep_h = ev_arrival_departure_hours(ev_df)
        ev_arr_dep_summary[f"house{h}_ev{ev}"] = {
            "arrivals_hour_hist": np.bincount(arr_h, minlength=24).tolist(),
            "departures_hour_hist": np.bincount(dep_h, minlength=24).tolist(),
        }

    # ---------------- Global deep analyses ----------------
    # Combined correlation matrix
    combined = _combined_dataframe(price, loads, pvs, evs)
    corr_all = full_correlation_matrix(combined)
    plot_corr_heatmap(corr_all, plots / "corr_all_heatmap.png", "Correlation matrix (all variables)")
    (reports / "corr_all_matrix.json").write_text(json.dumps(_to_native(corr_all), indent=2, ensure_ascii=False), encoding="utf-8")

    # House-to-house correlations
    hh_load_corr = house_to_house_correlation(loads)
    if not hh_load_corr.empty:
        plot_corr_heatmap(hh_load_corr, plots / "corr_house_to_house_load.png", "House-to-house load correlation")
        (reports / "corr_house_to_house_load.json").write_text(json.dumps(_to_native(hh_load_corr), indent=2, ensure_ascii=False), encoding="utf-8")

    hh_pv_corr = house_to_house_pv_correlation(pvs)
    if not hh_pv_corr.empty:
        plot_corr_heatmap(hh_pv_corr, plots / "corr_house_to_house_pv.png", "House-to-house PV correlation")
        (reports / "corr_house_to_house_pv.json").write_text(json.dumps(_to_native(hh_pv_corr), indent=2, ensure_ascii=False), encoding="utf-8")

    # Tariff elasticity proxy: price vs net import (sum over houses)
    if loads and pvs:
        # total net import
        idx_union = combined.index
        total_load = pd.concat(loads.values(), axis=1).sum(axis=1).reindex(idx_union).ffill()
        total_pv = pd.concat(pvs.values(), axis=1).sum(axis=1).reindex(idx_union).ffill()
        net = (total_load - total_pv).clip(lower=0.0)
        step_h = _step_hours(net.index)
        lag_steps = int(round(24.0/step_h))
        lags, ccf_price_net = cross_correlation(price.reindex(idx_union).ffill(), net, lag_steps)
        plot_crosscorr(lags*step_h, ccf_price_net, plots / "ccf_price_net_import.png",
                       "Cross-corr price↔net import (±24h)")

    # Save EV arrival/departure histograms summary
    (reports / "ev_arr_dep_summary.json").write_text(json.dumps(_to_native(ev_arr_dep_summary), indent=2, ensure_ascii=False), encoding="utf-8")

    return reports


def main():
    import argparse
    p = argparse.ArgumentParser(description="Analyze a MATRIX dataset scenario and produce deep metrics + plots")
    p.add_argument("scenario_dir", type=str, help="Path to datasets/<scenario_name>_<N>")
    args = p.parse_args()
    out = analyze_scenario(args.scenario_dir)
    print(f"Analysis complete. See: {out}")


if __name__ == "__main__":
    main()
