"""Forecasting-based Model Predictive Control algorithm for HEMS.

Implements the CUFE (Cairo University Faculty of Engineering) MPC approach
from the CityLearn Challenge 2022. The algorithm uses pretrained forecasters
for consumption, solar generation, and carbon intensity to produce 24-hour
ahead predictions, then solves an optimization problem via ``MPCFluid`` to
determine optimal battery charge/discharge actions. When pretrained models
are unavailable, simple heuristic fallback forecasters are used instead.

During the first 24 simulation steps a rule-based control (RBC) policy is
applied while the forecasters accumulate enough data. After that warm-up
window the full MPC pipeline takes over.
"""

#============================================
#hems/algorithms/mpc_forecast/mpc_forecast.py
#============================================

import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta

from ..base import BaseAlgorithm
from .mpcfluid import MPCFluid


class MPCForecastAlgorithm(BaseAlgorithm):
    """Model Predictive Control algorithm with pretrained CUFE forecasters.

    This non-trainable algorithm combines 24-hour ahead forecasts of
    consumption, solar generation, electricity price, and carbon intensity
    with an ``MPCFluid`` optimizer to compute optimal battery actions at
    each time step. A rolling 730-step historical baseline of net
    consumption (with and without battery) is maintained to calculate
    load-factor weights used by the optimizer.

    The algorithm operates in two phases:
        1. **Warm-up (steps 1-24):** A fixed RBC policy is applied while
           the forecasters gather initial observations.
        2. **MPC (steps 25+):** The full forecast-then-optimize pipeline
           produces battery charge/discharge decisions.

    Attributes:
        battery_capacity_kwh: Nominal battery energy capacity in kWh.
        battery_power_kw: Nominal battery power rating in kW.
        battery_efficiency: Round-trip battery efficiency (0-1).
        num_buildings: Number of buildings managed by this algorithm.
        mpc_optimizer: The ``MPCFluid`` optimization solver instance.
        use_pretrained: Whether pretrained CUFE forecaster models are loaded.
        step: Current simulation step within the episode.
        episode: Current episode counter.
        rbc_cycle: 24-element list of hourly RBC actions used during warm-up.
    """

    def __init__(self, env, config: Dict[str, Any]):
        """Initializes the MPC forecast algorithm.

        Extracts battery parameters from the environment, loads pretrained
        CUFE forecasters (falling back to heuristic models if data files
        are missing), and sets up the rolling historical baseline arrays.

        Args:
            env: The simulation environment instance. Expected to expose
                ``buildings`` or ``n_buildings`` attributes for multi-building
                support, and battery parameters on each building.
            config: Configuration dictionary. Recognized keys:
                - ``data_dir`` (str): Path to the directory containing
                  pretrained forecaster data files. Defaults to
                  ``'hems/algorithms/mpc_forecast/data'``.
        """
        super().__init__(env, config)
        
        self.env = env
        
        self.battery_capacity_kwh = self._get_battery_capacity(env)  
        self.battery_power_kw = self._get_battery_power(env)  
        self.battery_efficiency = self._get_battery_efficiency(env) 
        
        self.num_buildings = self._get_num_buildings(env)
        self.data_dir = Path(config.get('data_dir', 'hems/algorithms/mpc_forecast/data'))
        
        self.mpc_optimizer = MPCFluid()
        
        self.power_forecasters = []
        self.solar_forecasters = []
        self.carbon_forecaster = None
        self.weather = None
        
        self._init_cufe_forecasters()
        
        self.step = 0
        self.episode = 0
        self.observations_history = []
        
        self.rbc_cycle = [
            0.105914062, 0.160638021, 0.177486458, 0.158601042, 0.042078037, -0.097905316,
            -0.055921982, -0.086426772, -0.010381461, 0.031406874, 0.076785417, 0.075988542,
            0.05991125, 0.109503542, 0.081906146, 0.112274479, 0.058141662, -0.067366152,
            -0.094171464, -0.219347922, -0.348320602, -0.097350123, -0.048643353, -0.124800384
        ]
        
        self._historical_net_no_battery = [
            np.full(730, 0.5) for _ in range(self.num_buildings)
        ]
        self._historical_net_battery = [
            np.full(730, 0.5) for _ in range(self.num_buildings)
        ]
        self._warmup_complete = False
        
        self.start_date = datetime(2019, 1, 1)
        
        print(f"╔═══════════════════════════════════════════════════════════╗")
        print(f"║ CUFE MPC Initialized                                      ║")
        print(f"╠═══════════════════════════════════════════════════════════╣")
        print(f"║ Battery: {self.battery_capacity_kwh:.1f} kWh / {self.battery_power_kw:.1f} kW                        ║")
        print(f"║ Buildings: {self.num_buildings}                                                   ║")
        print(f"║ Forecasters: {'Pretrained' if self.use_pretrained else 'Fallback'}                                 ║")
        print(f"╚═══════════════════════════════════════════════════════════╝")
    
    def _initialize_historical_baseline(self):
        """Populates the historical baseline arrays from accumulated observations.

        Processes the first 730 stored observations to build per-building
        arrays of net consumption with and without battery contribution.
        These baselines are used by the MPC optimizer to compute
        load-change weights.

        Returns:
            bool: True if initialization succeeded (at least 730
                observations were available), False otherwise.
        """
        if len(self.observations_history) < 730:
            return False
        
        print(f"[Episode {self.episode}] Initializing historical baseline from {len(self.observations_history)} observations...")
        
        for building_idx in range(self.num_buildings):
            net_no_battery_values = []
            net_battery_values = []
            
            # Extract first 730 observations
            for step_idx, obs_list in enumerate(self.observations_history[:730]):
                if building_idx >= len(obs_list):
                    continue
                    
                obs_dict = self._extract_observations(obs_list[building_idx], building_idx)
                
                # Calculate net consumption without battery
                consumption = obs_dict['consumption']
                generation = obs_dict['solar_generation']
                net_no_bat = max(0.0, consumption - generation)
                net_no_battery_values.append(net_no_bat)
                
                # Net consumption with battery from observations
                net_with_bat = obs_dict['net_consumption']
                net_battery_values.append(net_with_bat)
            
            # Update historical arrays
            if len(net_no_battery_values) == 730:
                self._historical_net_no_battery[building_idx] = np.array(net_no_battery_values)
                self._historical_net_battery[building_idx] = np.array(net_battery_values)
                
                # Log statistics
                mean_no_bat = np.mean(net_no_battery_values)
                max_no_bat = np.max(net_no_battery_values)
                mean_bat = np.mean(net_battery_values)
                max_bat = np.max(net_battery_values)
                
                print(f"  Building {building_idx}: Net no-battery mean={mean_no_bat:.2f}, max={max_no_bat:.2f}")
                print(f"  Building {building_idx}: Net w/ battery mean={mean_bat:.2f}, max={max_bat:.2f}")
        
        self._warmup_complete = True
        print("[Warmup Complete] Historical baseline initialized successfully\n")
        return True
    
    def _get_num_buildings(self, env) -> int:
        """Returns the number of buildings in the environment.

        Args:
            env: The simulation environment.

        Returns:
            Number of buildings, defaulting to 1 if not determinable.
        """
        if hasattr(env, 'buildings'):
            return len(env.buildings)
        elif hasattr(env, 'n_buildings'):
            return env.n_buildings
        return 1
    
    def _get_battery_capacity(self, env) -> float:
        """Extracts the battery energy capacity from the environment.

        Args:
            env: The simulation environment.

        Returns:
            Battery capacity in kWh, defaulting to 6.4 if unavailable.
        """
        try:
            if hasattr(env, 'buildings') and len(env.buildings) > 0:
                return float(getattr(env.buildings[0].electrical_storage, 'capacity', 6.4))
        except:
            pass
        return 6.4
    
    def _get_battery_power(self, env) -> float:
        """Extracts the battery nominal power from the environment.

        Args:
            env: The simulation environment.

        Returns:
            Battery power in kW, defaulting to 5.0 if unavailable.
        """
        try:
            if hasattr(env, 'buildings') and len(env.buildings) > 0:
                return float(getattr(env.buildings[0].electrical_storage, 'nominal_power', 5.0))
        except:
            pass
        return 5.0
    
    def _get_battery_efficiency(self, env) -> float:
        """Extracts the battery round-trip efficiency from the environment.

        Args:
            env: The simulation environment.

        Returns:
            Battery efficiency as a float in [0, 1], defaulting to 0.9
            if unavailable.
        """
        try:
            if hasattr(env, 'buildings') and len(env.buildings) > 0:
                return float(getattr(env.buildings[0].electrical_storage, 'efficiency', 0.9))
        except:
            pass
        return 0.9
    
    def _init_cufe_forecasters(self):
        """Loads pretrained CUFE forecaster models from disk.

        Attempts to load power consumption, solar generation, and carbon
        intensity forecasters from NumPy data files in ``self.data_dir``.
        If any required files are missing or an error occurs, falls back
        to heuristic forecasting by setting ``self.use_pretrained`` to
        False.
        """
        self.use_pretrained = False
        
        try:
            consumed_file = self.data_dir / 'consumed.npy'
            consumed_beta_file = self.data_dir / 'consumed_beta.npy'
            solar_file = self.data_dir / 'solar.npy'
            solar_beta_file = self.data_dir / 'solar_beta.npy'
            solar_rank_file = self.data_dir / 'solar_rank.npy'
            carbon_file = self.data_dir / 'carbon_nn.sav'
            
            files_exist = all([
                consumed_file.exists(), 
                consumed_beta_file.exists(),
                solar_file.exists(),
                solar_beta_file.exists()
            ])
            
            if files_exist:
                from .cufe_forecasters import (
                    CUFEPowerForecaster, 
                    CUFESolarForecaster,
                    CUFECarbonForecaster,
                    Weather
                )
                
                for _ in range(self.num_buildings):
                    power_f = CUFEPowerForecaster(
                        lr_data_file=str(consumed_file),
                        ar_beta_file=str(consumed_beta_file)
                    )
                    self.power_forecasters.append(power_f)
                    
                    solar_f = CUFESolarForecaster(
                        lr_data_file=str(solar_file),
                        ar_beta_file=str(solar_beta_file),
                        rank_file=str(solar_rank_file) if solar_rank_file.exists() else None
                    )
                    self.solar_forecasters.append(solar_f)
                
                if carbon_file.exists():
                    self.carbon_forecaster = CUFECarbonForecaster(str(carbon_file))
                
                self.weather = Weather()
                self.use_pretrained = True
                print(f"✓ Loaded CUFE pretrained models")
            else:
                print(f"⚠ Missing data files, using fallback forecasts")
                
        except Exception as e:
            print(f"⚠ Error loading forecasters: {e}")
            self.use_pretrained = False
    
    def reset(self):
        """Resets the algorithm state for a new episode.

        Clears the step counter, observation history, and historical
        baseline arrays. Also resets all loaded forecasters so they
        start fresh.
        """
        self.step = 0
        self.episode += 1
        self.observations_history = []
        
        for forecaster in self.power_forecasters:
            forecaster.reset()
        
        for forecaster in self.solar_forecasters:
            forecaster.reset()
        
        if self.carbon_forecaster:
            self.carbon_forecaster.reset()
        
        self._historical_net_no_battery = [np.zeros(730) for _ in range(self.num_buildings)]
        self._historical_net_battery = [np.zeros(730) for _ in range(self.num_buildings)]
        
        print(f"[Episode {self.episode}] Reset complete")
    
    def _extract_observations(self, obs: List[float], building_idx: int) -> Dict[str, float]:
        """Parses a raw observation vector into a named dictionary.

        Extracts time, energy, and pricing fields from the flat
        observation list and derives additional quantities such as
        calendar date, normalized state-of-charge, and total
        consumption.

        Args:
            obs: Raw observation vector with at least 5 elements:
                [hour, net_consumption, solar_generation, soc, pricing].
            building_idx: Index of the building this observation
                belongs to (used for potential per-building handling).

        Returns:
            Dictionary with keys: ``month``, ``day``, ``hour``,
            ``pricing``, ``pricing_6h``, ``pricing_12h``,
            ``pricing_24h``, ``net_consumption``, ``soc``,
            ``solar_generation``, ``consumption``, ``temperature``,
            ``humidity``, ``carbon_intensity``.

        Raises:
            AssertionError: If ``obs`` has fewer than 5 elements.
        """
        assert len(obs) >= 5, f"Observation too short: {len(obs)}"
        
        hour_raw = int(obs[0])
        net_consumption = float(obs[1])
        solar_generation = float(obs[2])
        soc_raw = float(obs[3])
        pricing = float(obs[4])
        
        current_date = self.start_date + timedelta(hours=self.step)
        month = current_date.month
        day = current_date.day
        hour = current_date.hour + 1
        
        soc = np.clip(soc_raw / self.battery_capacity_kwh, 0.0, 1.0)
        consumption = net_consumption + solar_generation
        
        return {
            'month': month,
            'day': day,
            'hour': hour,
            'pricing': pricing,
            'pricing_6h': pricing,
            'pricing_12h': pricing,
            'pricing_24h': pricing,
            'net_consumption': net_consumption,
            'soc': soc,
            'solar_generation': solar_generation,
            'consumption': consumption,
            'temperature': 20.0,
            'humidity': 50.0,
            'carbon_intensity': 0.5
        }
    
    def _rbc_policy(self, obs_dict: Dict[str, float]) -> float:
        """Computes a battery action using the fixed rule-based control cycle.

        Used during the warm-up phase (first 24 steps) before the MPC
        optimizer has enough forecast data to operate.

        Args:
            obs_dict: Parsed observation dictionary (only ``hour`` is used).

        Returns:
            Battery action in [-1, 1] where negative means discharge
            and positive means charge.
        """
        hour = (obs_dict['hour'] - 1) % 24
        action = self.rbc_cycle[hour]
        return float(np.clip(action, -1.0, 1.0))
    
    def _mpc_policy(self, obs_dict: Dict[str, float], building_idx: int) -> float:
        """Computes a battery action using the full MPC optimization pipeline.

        Updates the forecasters with the latest observations, generates
        24-hour ahead forecasts for consumption, solar generation, price,
        and carbon intensity, computes load-factor weights from
        historical baselines, and invokes the ``MPCFluid`` optimizer.

        Args:
            obs_dict: Parsed observation dictionary containing current
                time, energy, pricing, and state-of-charge values.
            building_idx: Index of the building to compute the action for.

        Returns:
            Battery action in [-1, 1]. Returns 0.0 if an error occurs.
        """
        try:
            hour = obs_dict['hour']
            month = obs_dict['month']
            day = obs_dict['day']
            pricing = obs_dict['pricing']
            net_consumption = obs_dict['net_consumption']
            solar_generation = obs_dict['solar_generation']
            consumption = obs_dict['consumption']
            soc = obs_dict['soc']
            
            if self.weather is not None:
                self.weather.update_temperature(20.0, 20.0, 20.0, 20.0)
                self.weather.update_humidity(50.0, 50.0, 50.0, 50.0)
                self.weather.update_diffuse_solar_irradiance(
                    solar_generation, solar_generation, solar_generation, solar_generation
                )
                self.weather.update_direct_solar_irradiance(
                    solar_generation, solar_generation, solar_generation, solar_generation
                )
            
            if building_idx < len(self.power_forecasters):
                self.power_forecasters[building_idx].update_power(consumption, month, day, hour)
                self.solar_forecasters[building_idx].update_solar(solar_generation, month, day, hour)            
            
            if self.carbon_forecaster is not None and building_idx == 0:
                # Estimate carbon from pricing correlation
                carbon_estimate = self._estimate_carbon_from_pricing(pricing, hour)
                self.carbon_forecaster.update_carbon_intensity(carbon_estimate)
            
            if building_idx < len(self.power_forecasters) and self.use_pretrained:
                consumption_24h, c_error = self.power_forecasters[building_idx].forecast(self.weather, hour)
                generation_24h, g_error = self.solar_forecasters[building_idx].forecast(self.weather)
                forecast_error = c_error + g_error
            else:
                consumption_24h = self._fallback_consumption_forecast(consumption, solar_generation, hour)
                generation_24h = self._fallback_solar_forecast(solar_generation, hour)
                forecast_error = 0.1
            
            price_24h = self._generate_price_forecast(pricing, hour)
            
            if self.carbon_forecaster is not None:
                carbon_24h = self.carbon_forecaster.forecast()
            else:
                carbon_24h = self._generate_carbon_forecast(hour, pricing)
            
            if isinstance(carbon_24h, np.ndarray) and carbon_24h.ndim > 1:
                carbon_24h = carbon_24h.flatten()
            if isinstance(consumption_24h, np.ndarray) and consumption_24h.ndim > 1:
                consumption_24h = consumption_24h.flatten()
            if isinstance(generation_24h, np.ndarray) and generation_24h.ndim > 1:
                generation_24h = generation_24h.flatten()
            if isinstance(price_24h, np.ndarray) and price_24h.ndim > 1:
                price_24h = price_24h.flatten()
            
            max_net_consumption = float(np.max(self._historical_net_battery[building_idx]))
            max_net_no_battery = float(np.max(self._historical_net_no_battery[building_idx]))
            mean_net_no_battery = float(np.mean(self._historical_net_no_battery[building_idx]))

            # Safety: avoid division by zero
            if max_net_consumption < 0.01:
                max_net_consumption = 1.0
            if max_net_no_battery < 0.01:
                max_net_no_battery = 1.0

            # Calculate load factor weight with safety checks
            if max_net_no_battery > 0 and (max_net_no_battery - mean_net_no_battery) > 0:
                load_change_weight = 1.0 / max_net_consumption / (1.0 - mean_net_no_battery / max_net_no_battery)
            else:
                load_change_weight = 0.1  # Safe default

            # Clip to reasonable range
            load_change_weight = float(np.clip(load_change_weight, 0.0, 10.0))
            
            action = self.mpc_optimizer.forecast(
                price=price_24h,
                carbon=carbon_24h,
                consumption=consumption_24h,
                generation=generation_24h,
                battery=soc,
                net_consumption=net_consumption,
                error=forecast_error / self.step,
                max_net_consumption=max_net_consumption,
                load_change_weight=load_change_weight
            )
            
            return float(np.clip(action, -1.0, 1.0))
            
        except Exception as e:
            print(f"[MPC Error] {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    def _estimate_carbon_from_pricing(self, pricing: float, hour: int) -> float:
        """Estimates grid carbon intensity from electricity price and hour.

        Uses a linear mapping from normalized price to a carbon intensity
        range, with diurnal adjustments: reduced during solar hours
        (10-16) and increased during evening peak (18-21).

        Args:
            pricing: Current electricity price (e.g., EUR/kWh).
            hour: Hour of the day (1-indexed, i.e., 1 = midnight to 1 AM).

        Returns:
            Estimated carbon intensity in [0.1, 1.0].
        """
        # Price normalization (adjust these based on your data)
        base_price = 0.08   # Typical off-peak price
        peak_price = 0.25   # Typical peak price
        
        # Clip and normalize to [0, 1]
        price_factor = (pricing - base_price) / (peak_price - base_price)
        price_factor = float(np.clip(price_factor, 0.0, 1.0))
        
        # Carbon intensity mapping
        min_carbon = 0.25   # Low carbon (high renewable penetration)
        max_carbon = 0.85   # High carbon (coal/gas peaking plants)
        
        # Base estimate from price
        carbon_est = min_carbon + price_factor * (max_carbon - min_carbon)
        
        # Add diurnal pattern (solar reduces carbon during day)
        hour_of_day = (hour - 1) % 24
        if 10 <= hour_of_day <= 16:
            # Solar hours - reduce carbon estimate
            solar_factor = 0.85  # 15% reduction
            carbon_est *= solar_factor
        elif 18 <= hour_of_day <= 21:
            # Evening peak - increase carbon estimate
            peak_factor = 1.15  # 15% increase
            carbon_est *= peak_factor
        
        # Clip to valid range
        carbon_est = float(np.clip(carbon_est, 0.1, 1.0))
        
        return carbon_est
    
    def _generate_price_forecast(self, current_price: float, hour: int) -> np.ndarray:
        """Generates a 24-hour ahead electricity price forecast.

        Applies a simple heuristic: off-peak hours (23-06) are scaled
        to 70 % of the current price, peak hours (17-21) to 140 %,
        and all other hours remain at the current price.

        Args:
            current_price: The electricity price at the current step.
            hour: Current hour of the day (1-indexed).

        Returns:
            Array of shape ``(24,)`` with forecasted prices.
        """
        current_price = max(0.01, float(current_price))
        price_24h = np.full(24, current_price)
        for h in range(24):
            future_hour = int((hour + h - 1) % 24)
            if future_hour <= 6 or future_hour >= 23:
                price_24h[h] = current_price * 0.7
            elif 17 <= future_hour <= 21:
                price_24h[h] = current_price * 1.4
        return price_24h
    
    def _fallback_consumption_forecast(self, consumption: float, solar: float, hour: int) -> np.ndarray:
        """Generates a 24-hour consumption forecast without pretrained models.

        First attempts a persistence model using the last 24 hours of
        observed consumption. If insufficient history is available, falls
        back to a scaled diurnal pattern based on the current consumption.

        Args:
            consumption: Current total consumption (load + solar) in kW.
            solar: Current solar generation in kW.
            hour: Current hour of the day (1-indexed).

        Returns:
            Array of shape ``(24,)`` with forecasted consumption values.
        """
        consumption_24h = np.zeros(24)
        
        # Try to use recent history if available
        if hasattr(self, 'observations_history') and len(self.observations_history) >= 24:
            recent_consumption = []
            
            for obs_list in self.observations_history[-24:]:
                if len(obs_list) > 0:
                    obs_dict = self._extract_observations(obs_list[0], 0)
                    recent_consumption.append(obs_dict['consumption'] + obs_dict['solar_generation'])
            
            if len(recent_consumption) == 24:
                # Use last 24 hours as forecast (persistence model)
                for h in range(24):
                    consumption_24h[h] = recent_consumption[h]
                return consumption_24h
        
        # Fallback to simple diurnal pattern
        base = max(0.1, float(consumption + solar))
        for h in range(24):
            fh = int((hour + h - 1) % 24)
            if 0 <= fh <= 6:
                consumption_24h[h] = base * 0.5
            elif 7 <= fh <= 9:
                consumption_24h[h] = base * 0.8
            elif 10 <= fh <= 16:
                consumption_24h[h] = base * 1.0
            elif 17 <= fh <= 22:
                consumption_24h[h] = base * 1.3
            else:
                consumption_24h[h] = base * 0.7
        
        return consumption_24h
    
    def _fallback_solar_forecast(self, current_solar: float, hour: int) -> np.ndarray:
        """Generates a 24-hour solar generation forecast without pretrained models.

        First attempts a persistence model using the last 24 hours of
        observed solar generation. If insufficient history is available,
        falls back to a symmetric bell-shaped curve peaking at noon
        (hours 8-16).

        Args:
            current_solar: Current solar generation in kW.
            hour: Current hour of the day (1-indexed).

        Returns:
            Array of shape ``(24,)`` with forecasted solar generation.
        """
        solar_24h = np.zeros(24)
        
        # Try to use recent history
        if hasattr(self, 'observations_history') and len(self.observations_history) >= 24:
            recent_solar = []
            
            for obs_list in self.observations_history[-24:]:
                if len(obs_list) > 0:
                    obs_dict = self._extract_observations(obs_list[0], 0)
                    recent_solar.append(obs_dict['solar_generation'])
            
            if len(recent_solar) == 24:
                # Use last 24 hours as forecast
                for h in range(24):
                    solar_24h[h] = recent_solar[h]
                return solar_24h
        
        # Fallback to simple solar curve
        for h in range(24):
            fh = int((hour + h - 1) % 24)
            if 8 <= fh <= 16:
                # Solar production hours
                peak = 12
                dist = abs(fh - peak)
                factor = max(0.0, 1.0 - dist / 5.0)
                solar_24h[h] = current_solar * factor if current_solar > 0 else 0.5 * factor
        
        return solar_24h
    
    def _generate_carbon_forecast(self, hour: int, pricing: float) -> np.ndarray:
        """Generates a 24-hour carbon intensity forecast from price heuristics.

        For each future hour, estimates the electricity price using the
        same diurnal scaling as ``_generate_price_forecast``, then
        converts it to carbon intensity via ``_estimate_carbon_from_pricing``.

        Args:
            hour: Current hour of the day (1-indexed).
            pricing: Current electricity price.

        Returns:
            Array of shape ``(24,)`` with forecasted carbon intensities.
        """
        carbon_24h = np.zeros(24)
        
        for h in range(24):
            future_hour = (hour + h - 1) % 24
            
            # Estimate future price (reuse logic from _generate_price_forecast)
            if future_hour <= 6 or future_hour >= 23:
                future_price = pricing * 0.7
            elif 17 <= future_hour <= 21:
                future_price = pricing * 1.4
            else:
                future_price = pricing * 1.0
            
            # Estimate carbon from future price
            carbon_24h[h] = self._estimate_carbon_from_pricing(future_price, future_hour + 1)
        
        return carbon_24h
    
    def act(self, observations: List[List[float]], deterministic: bool = False) -> List[List[float]]:
        """Selects battery actions for all buildings at the current time step.

        Delegates to ``_rbc_policy`` during the first 24 steps and to
        ``_mpc_policy`` afterwards. Also updates the rolling historical
        baseline arrays with the latest net consumption values and
        triggers baseline initialization at step 730 if not yet done.

        Args:
            observations: List of per-building observation vectors.
                Each inner list is a raw observation with at least 5
                elements (see ``_extract_observations``).
            deterministic: Unused. Present for API compatibility with
                ``BaseAlgorithm``.

        Returns:
            List of per-building actions, each wrapped in a single-element
            list (e.g., ``[[0.3], [-0.1]]`` for two buildings). Returns
            all-zero actions if an unrecoverable error occurs.
        """
        self.step += 1
        self.observations_history.append(observations)

        if self.step == 730 and not self._warmup_complete:
            self._initialize_historical_baseline()
        
        if self.step % 200 == 0:
            policy = "RBC" if self.step <= 24 else "MPC"
            print(f"[Episode {self.episode}] Step {self.step} ({policy})")
        
        try:
            actions = []
            
            for building_idx, obs in enumerate(observations):
                obs_dict = self._extract_observations(obs, building_idx)
                
                if self.step <= 24:
                    action = self._rbc_policy(obs_dict)
                else:
                    action = self._mpc_policy(obs_dict, building_idx)
                
                net_no_bat = obs_dict['consumption'] - obs_dict['solar_generation']
                self._historical_net_no_battery[building_idx][0:729] = self._historical_net_no_battery[building_idx][1:730]
                self._historical_net_no_battery[building_idx][729] = net_no_bat
                
                self._historical_net_battery[building_idx][0:729] = self._historical_net_battery[building_idx][1:730]
                self._historical_net_battery[building_idx][729] = obs_dict['net_consumption']
                
                actions.append([action])
            
            return actions
            
        except Exception as e:
            print(f"[ERROR] act() failed: {e}")
            import traceback
            traceback.print_exc()
            return [[0.0] for _ in range(len(observations))]
    
    def learn(self, *args, **kwargs):
        """No-op. MPC is a non-trainable optimization-based algorithm."""
        pass
    
    def save(self, path: str):
        """Persists the algorithm state to a pickle file.

        Saves the step counter, episode counter, and historical baseline
        arrays so the algorithm can be resumed later.

        Args:
            path: File path to write the pickled state to.
        """
        import pickle
        state = {
            'step': self.step,
            'episode': self.episode,
            'historical_net_no_battery': self._historical_net_no_battery,
            'historical_net_battery': self._historical_net_battery
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: str):
        """Restores the algorithm state from a pickle file.

        Args:
            path: File path to read the pickled state from.
        """
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.step = state['step']
        self.episode = state['episode']
        self._historical_net_no_battery = state['historical_net_no_battery']
        self._historical_net_battery = state['historical_net_battery']
    
    def get_state(self) -> Dict[str, Any]:
        """Returns a summary of the current algorithm state.

        Returns:
            Dictionary with keys: ``algorithm_type``,
            ``steps_completed``, ``episode``, ``num_buildings``,
            ``trainable``, and ``using_pretrained``.
        """
        return {
            'algorithm_type': 'CUFE_MPC',
            'steps_completed': self.step,
            'episode': self.episode,
            'num_buildings': self.num_buildings,
            'trainable': False,
            'using_pretrained': self.use_pretrained
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Returns metadata describing this algorithm.

        Extends the base class info with description, trainability
        flag, algorithm type, and forecaster status.

        Returns:
            Dictionary with algorithm metadata including a
            ``forecasters`` key indicating ``'pretrained'`` or
            ``'fallback'``.
        """
        info = super().get_info()
        info.update({
            'description': 'CUFE MPC with Pretrained Forecasters',
            'trainable': False,
            'type': 'optimization',
            'forecasters': 'pretrained' if self.use_pretrained else 'fallback'
        })
        return info