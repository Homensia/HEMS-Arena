from __future__ import annotations
"""
Plot utilities for MATRIX datasets. Uses matplotlib only, one plot per figure, no specific colors.
"""
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def plot_timeseries(df: pd.DataFrame, path: Path, title: Optional[str] = None, ylabel: Optional[str] = None):
    _ensure_dir(path)
    ax = df.plot(figsize=(12, 4))
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_histogram(s: pd.Series, path: Path, bins: int = 50, title: Optional[str] = None, xlabel: Optional[str] = None):
    _ensure_dir(path)
    plt.figure(figsize=(8, 5))
    plt.hist(s.dropna().to_numpy(), bins=bins)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_boxplot(series_list: List[pd.Series], labels: List[str], path: Path, title: Optional[str] = None, ylabel: Optional[str] = None):
    _ensure_dir(path)
    plt.figure(figsize=(8, 5))
    plt.boxplot([s.dropna().to_numpy() for s in series_list], labels=labels, showfliers=False)
    if title:
        plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_duration_curve(s: pd.Series, path: Path, title: Optional[str] = None, ylabel: Optional[str] = None):
    """Sort descending and plot vs percentile."""
    _ensure_dir(path)
    vals = np.sort(s.dropna().to_numpy())[::-1]
    x = np.linspace(0, 100, len(vals))
    plt.figure(figsize=(10, 5))
    plt.plot(x, vals)
    if title:
        plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    plt.xlabel("Percentile (%)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_hourly_profile(profile: pd.Series, path: Path, title: Optional[str] = None, ylabel: Optional[str] = None):
    _ensure_dir(path)
    plt.figure(figsize=(10, 4))
    hours = profile.index.to_numpy()
    plt.plot(hours, profile.to_numpy(), marker="o")
    if title:
        plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    plt.xlabel("Hour of day")
    plt.xticks(range(0,24,2))
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_heatmap_day_hour(s: pd.Series, path: Path, title: Optional[str] = None):
    """Create a day x hour heatmap (mean per hour for each day)."""
    _ensure_dir(path)
    df = s.copy()
    df = df.to_frame("v")
    df["day"] = df.index.normalize()
    df["hour"] = df.index.hour
    pivot = df.pivot_table(index="day", columns="hour", values="v", aggfunc="mean")
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
    if title:
        plt.title(title)
    plt.xlabel("Hour")
    plt.ylabel("Day index")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_line(y: pd.Series, path: Path, title: Optional[str] = None, ylabel: Optional[str] = None, xlabel: Optional[str] = None):
    _ensure_dir(path)
    plt.figure(figsize=(10,4))
    plt.plot(y.index.to_numpy(), y.to_numpy())
    if title:
        plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_autocorr(acf_vals: np.ndarray, step_hours: float, path: Path, title: Optional[str] = None):
    _ensure_dir(path)
    lags = np.arange(len(acf_vals))
    hours = lags * step_hours
    plt.figure(figsize=(10,4))
    plt.bar(hours, acf_vals, width=step_hours*0.8)
    if title:
        plt.title(title)
    plt.xlabel("Lag (hours)")
    plt.ylabel("Autocorrelation")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()



    # ====================== DEEP DIVE PLOTS ======================

def plot_corr_heatmap(corr_df: pd.DataFrame, path: Path, title: Optional[str] = None):
    _ensure_dir(path)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8 + 0.4 * corr_df.shape[1], 6 + 0.4 * corr_df.shape[0]))
    im = plt.imshow(corr_df.to_numpy(), aspect='auto', origin='lower', vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(corr_df.shape[1]), corr_df.columns, rotation=90)
    plt.yticks(range(corr_df.shape[0]), corr_df.index)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_month_hour_matrix(mat: pd.DataFrame, path: Path, title: Optional[str] = None, ylabel: str = "Hour"):
    _ensure_dir(path)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    im = plt.imshow(mat.to_numpy(), aspect='auto', origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Month")
    plt.ylabel(ylabel)
    plt.xticks(range(mat.shape[1]), mat.columns)
    plt.yticks(range(mat.shape[0]), mat.index)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_fft(period_hours: np.ndarray, power: np.ndarray, path: Path, title: Optional[str] = None):
    _ensure_dir(path)
    import matplotlib.pyplot as plt
    if len(period_hours) == 0:
        # create a blank figure with message
        plt.figure(figsize=(8,4))
        plt.title(title or "FFT (insufficient length)")
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()
        return
    # limit to reasonable periods (e.g., up to 1000h to focus)
    mask = np.isfinite(period_hours) & (period_hours > 0) & (period_hours <= 1000)
    ph = period_hours[mask]
    pw = power[mask]
    order = np.argsort(ph)
    ph, pw = ph[order], pw[order]
    plt.figure(figsize=(10,4))
    plt.plot(ph, pw)
    plt.xlabel("Period (hours)")
    plt.ylabel("Power")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_crosscorr(lags_hours: np.ndarray, ccf: np.ndarray, path: Path, title: Optional[str] = None):
    _ensure_dir(path)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.bar(lags_hours, ccf, width=(lags_hours[1]-lags_hours[0]) if len(lags_hours) > 1 else 1.0)
    plt.xlabel("Lag (hours)")
    plt.ylabel("Cross-correlation")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
