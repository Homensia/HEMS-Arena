from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Union

class WeatherConfig(BaseModel):
    model: Literal["simple"] = "simple"
    ambient_temp_mean_c: float = 15.0
    ambient_temp_amp_c: float = 10.0
    daily_temp_amp_c: float = 5.0
    noise_std_c: float = 1.5
    cloud_markov_levels: tuple[float, float, float] = (0.9, 0.6, 0.2)
    cloud_transition: list[list[float]] = Field(
        default_factory=lambda: [
            [0.85, 0.12, 0.03],
            [0.20, 0.60, 0.20],
            [0.10, 0.20, 0.70],
        ]
    )

class HouseConfig(BaseModel):
    base_load_kwh_day_mean: float = 10.0
    weekday_multiplier: float = 1.0
    weekend_multiplier: float = 1.1
    morning_peak_kw: float = 0.6
    evening_peak_kw: float = 0.9
    noise_ar1_phi: float = 0.7
    noise_sigma_kw: float = 0.05
    # Optional degree-day coefficients (set to 0 by default)
    heat_sensitivity_kwh_per_dd: float = 0.0
    cool_sensitivity_kwh_per_dd: float = 0.0

class PVConfig(BaseModel):
    dc_capacity_kwp: float = 3.0
    tilt_deg: float = 30.0
    azimuth_deg: float = 0.0  # 0 ~ south
    performance_ratio: float = 0.85
    gamma_temp_coeff_per_c: float = -0.0045
    inverter_efficiency: float = 0.96
    shading_factor: float = 0.0
    noct_c: float = 45.0

class EVFleetConfig(BaseModel):
    evs_per_house: int = 1
    battery_kwh: float = 50.0
    charger_power_kw: float = 7.4
    charging_efficiency: float = 0.92
    kwh_per_km: float = 0.16
    daily_trip_km_mean: float = 30.0
    daily_trip_km_std: float = 15.0
    arrival_hour_mean: float = 18.5
    arrival_hour_std: float = 1.5
    departure_hour_mean: float = 7.5
    departure_hour_std: float = 1.0
    min_target_soc_at_departure: float = 0.8
    policy: Literal["immediate", "hp_hc_aware"] = "immediate"
    enable_v2h: bool = False  # dataset does not inject V2H by default

class FRBaseTariff(BaseModel):
    type: Literal["FR_BASE"] = "FR_BASE"
    price_eur_per_kwh: float = 0.20

class FRHPHCTariff(BaseModel):
    type: Literal["FR_HP_HC"] = "FR_HP_HC"
    hp_hours: list[int] = Field(default_factory=lambda: list(range(6, 22)))  # 6-21
    hc_hours: list[int] = Field(default_factory=lambda: list(range(22, 24)) + list(range(0, 6)))  # 22-5
    hp_price: float = 0.24
    hc_price: float = 0.16

class FRTEMPOCalendar(BaseModel):
    blue_days: int = 300
    white_days: int = 43
    red_days: int = 22

class FRTEMPOConfig(BaseModel):
    type: Literal["FR_TEMPO"] = "FR_TEMPO"
    calendar: FRTEMPOCalendar = Field(default_factory=FRTEMPOCalendar)
    price_hp_blue: float = 0.18
    price_hc_blue: float = 0.14
    price_hp_white: float = 0.22
    price_hc_white: float = 0.16
    price_hp_red: float = 0.54
    price_hc_red: float = 0.30

class DynamicTariffConfig(BaseModel):
    type: Literal["DYNAMIC"] = "DYNAMIC"
    mean_price: float = 0.20
    daily_amp: float = 0.08
    ar1_phi: float = 0.8
    noise_sigma: float = 0.02
    spike_prob: float = 0.02
    spike_amp: float = 0.25
    allow_negative: bool = True

class RandomTariffConfig(BaseModel):
    type: Literal["RANDOM"] = "RANDOM"
    mean: float = 0.20
    std: float = 0.05
    min_price: float = 0.05
    max_price: float = 0.60

TariffConfig = Union[FRBaseTariff, FRHPHCTariff, FRTEMPOConfig, DynamicTariffConfig, RandomTariffConfig]

class ScenarioConfig(BaseModel):
    scenario_name: str = "basic_data"
    output_dir: str = "./datasets"
    days: int = 365
    step_minutes: int = 60
    seed: int = 42
    n_houses: int = 1
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    house: HouseConfig = Field(default_factory=HouseConfig)
    pv: PVConfig = Field(default_factory=PVConfig)
    ev: EVFleetConfig = Field(default_factory=EVFleetConfig)
    tariff: TariffConfig = Field(default_factory=FRHPHCTariff)
    timezone: str = "Europe/Paris"

    @field_validator("step_minutes")
    @classmethod
    def _step_valid(cls, v: int) -> int:
        if 1440 % v != 0:
            raise ValueError("step_minutes must divide 1440 exactly.")
        return v
