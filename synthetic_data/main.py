from __future__ import annotations
import argparse, json
import numpy as np  # for numpy -> python casting
from synthetic_data.data_gen.config import (
    ScenarioConfig,
    FRHPHCTariff, FRBaseTariff, FRTEMPOConfig, DynamicTariffConfig, RandomTariffConfig
)
from synthetic_data.data_gen.orchestrator import run_scenario


def _to_native(o):
    """Recursively convert numpy scalars/containers to built-in Python types for JSON."""
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
    return o


def make_default_config() -> ScenarioConfig:
    # Default: 1 house, 1 EV, 365 days, 60-min step, FR HP/HC
    return ScenarioConfig()


def parse_args():
    p = argparse.ArgumentParser(description="Synthetic HEMS dataset generator")
    p.add_argument("--use-default", action="store_true",
                   help="Use built-in default scenario (ignores other flags).")
    p.add_argument("--output-dir", type=str, default="./datasets")
    p.add_argument("--scenario-name", type=str, default="basic_data")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--step-minutes", type=int, default=60)
    p.add_argument("--n-houses", type=int, default=1)
    p.add_argument("--tariff", type=str, default="FR_HP_HC",
                   choices=["FR_BASE", "FR_HP_HC", "FR_TEMPO", "DYNAMIC", "RANDOM"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_config_from_flags(args) -> ScenarioConfig:
    cfg = ScenarioConfig(
        scenario_name=args.scenario_name,
        output_dir=args.output_dir,
        days=args.days,
        step_minutes=args.step_minutes,
        n_houses=args.n_houses,
        seed=args.seed,
    )
    # Minimal tariff switch
    if args.tariff == "FR_BASE":
        cfg.tariff = FRBaseTariff()
    elif args.tariff == "FR_HP_HC":
        cfg.tariff = FRHPHCTariff()
    elif args.tariff == "FR_TEMPO":
        cfg.tariff = FRTEMPOConfig()
    elif args.tariff == "DYNAMIC":
        cfg.tariff = DynamicTariffConfig()
    elif args.tariff == "RANDOM":
        cfg.tariff = RandomTariffConfig()
    return cfg


def main():
    args = parse_args()
    cfg = make_default_config() if args.use_default else build_config_from_flags(args)
    out_dir, val = run_scenario(cfg)
    print(f"✅ Scenario saved to: {out_dir}")
    # Convert validation dict to native Python types before JSON
    val_native = _to_native(val)
    print("Validation summary:", json.dumps(val_native, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
