from __future__ import annotations
import json
import hashlib
from pathlib import Path
from typing import Tuple, Any

import numpy as np
import pandas as pd


def _next_scenario_dir(output_root: Path, scenario_name: str) -> Tuple[Path, int]:
    output_root.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        d = output_root / f"{scenario_name}_{n}"
        if not d.exists():
            d.mkdir(parents=True, exist_ok=False)
            return d, n
        n += 1


def _write_csv(df: pd.DataFrame | pd.Series, path: Path) -> None:
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df.to_csv(path, index=True)


def _to_native(o: Any) -> Any:
    """Recursively convert numpy/pandas/Path types to JSON-serializable Python types."""
    # dict-like
    if isinstance(o, dict):
        return {str(k): _to_native(v) for k, v in o.items()}
    # list/tuple/set
    if isinstance(o, (list, tuple, set)):
        return [_to_native(v) for v in o]
    # numpy scalars
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    # pandas types
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, pd.Timedelta):
        return o.isoformat()
    if isinstance(o, (pd.Series, pd.DataFrame)):
        # not expected inside metadata, but make safe
        return _to_native(o.to_dict())
    # pathlib
    if isinstance(o, Path):
        return str(o)
    # leave plain Python types as is
    return o


def persist_scenario(
    output_dir: str,
    scenario_name: str,
    time_index: pd.DatetimeIndex,
    loads: dict[int, pd.Series],
    pvs: dict[int, pd.Series],
    prices: pd.Series,
    evs: dict[tuple[int, int], pd.DataFrame],
    metadata: dict,
) -> Path:
    root = Path(output_dir)
    scen_dir, scen_num = _next_scenario_dir(root, scenario_name)
    csv_dir = scen_dir / "csv"
    rpt_dir = scen_dir / "reports" / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    rpt_dir.mkdir(parents=True, exist_ok=True)

    # Write CSVs
    for h, s in loads.items():
        _write_csv(s.rename("load_kw"), csv_dir / f"{scenario_name}_house_elec_{scen_num}_{h}.csv")
    for h, s in pvs.items():
        _write_csv(s.rename("pv_kw"), csv_dir / f"{scenario_name}_pv_generation_{scen_num}_{h}.csv")
    _write_csv(prices.rename("price_eur_per_kwh"), csv_dir / f"{scenario_name}_price_{scen_num}.csv")
    for (h, ev), df in evs.items():
        _write_csv(df, csv_dir / f"{scenario_name}_ev_{scen_num}_{h}_{ev}.csv")

    # Metadata + file hashes (converted to native types for JSON)
    meta = dict(metadata)
    hashes: dict[str, str] = {}
    for p in csv_dir.glob("*.csv"):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        hashes[p.name] = h
    meta["file_hashes"] = hashes

    meta_native = _to_native(meta)
    (scen_dir / "metadata.json").write_text(json.dumps(meta_native, indent=2, ensure_ascii=False), encoding="utf-8")
    (scen_dir / "README.md").write_text("# Synthetic dataset\n\nSee metadata.json and csv/ folder.\n", encoding="utf-8")
    return scen_dir
