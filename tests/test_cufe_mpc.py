#========================
# tests/test_cufe_mpc.py 
#========================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import numpy as np
import pytest

from hems.algorithms.mpc_forecast import (
    MPCForecastAlgorithm,
    MPCFluid
)

# CRITICAL FIX: Import from cufe_forecasters, not forecasting
from hems.algorithms.mpc_forecast.cufe_forecasters import (
    CUFEPowerForecaster,
    CUFESolarForecaster,
    CUFECarbonForecaster
)

# ---------------------------------------------------------
# Fixtures and Paths
# ---------------------------------------------------------

@pytest.fixture(scope="session")
def data_dir():
    """Path to CUFE pretrained assets."""
    base = "hems/algorithms/mpc_forecast/data"
    files = [
        "consumed.npy", "consumed_beta.npy",
        "solar.npy", "solar_beta.npy",
        "carbon_nn.sav"
    ]
    missing = [f for f in files if not os.path.exists(os.path.join(base, f))]
    assert not missing, f"Missing pretrained files: {missing}"
    return base

@pytest.fixture
def dummy_env():
    """Minimal dummy env that mimics CityLearn structure."""
    class DummyStorage:
        capacity = 6.4
        nominal_power = 5.0
        efficiency = 0.9

    class DummyBuilding:
        electrical_storage = DummyStorage()

    class DummyEnv:
        buildings = [DummyBuilding()]
    return DummyEnv()

# ---------------------------------------------------------
# Forecasting Tests
# ---------------------------------------------------------

def test_power_forecaster_basic(data_dir):
    """Test CUFE power forecaster with AR(168) + LR blending"""
    lr_file = os.path.join(data_dir, "consumed.npy")
    beta_file = os.path.join(data_dir, "consumed_beta.npy")

    pf = CUFEPowerForecaster(lr_file, beta_file)
    
    # Simulate a week of data to initialize AR(168)
    for i in range(24 * 7):
        pf.update_power(
            power=float(1.0 + 0.5 * np.sin(i * 2 * np.pi / 24)), 
            month=1, 
            day=(i // 24) + 1, 
            hour=(i % 24) + 1
        )
    
    forecast, err = pf.forecast()
    
    assert forecast.shape[0] == 24, f"Expected 24 forecasts, got {forecast.shape[0]}"
    assert np.all(np.isfinite(forecast)), "Forecast contains NaN/Inf"
    assert 0 <= err < 10, f"Error too large: {err}"
    print(f"✓ Power forecaster: mean={np.mean(forecast):.3f}, error={err:.3f}")

def test_solar_forecaster_weather_mask(data_dir):
    """Test CUFE solar forecaster with weather masking"""
    lr_file = os.path.join(data_dir, "solar.npy")
    beta_file = os.path.join(data_dir, "solar_beta.npy")

    sf = CUFESolarForecaster(lr_file, beta_file, rank_file=None)
    
    # Simulate solar generation (only during day)
    for i in range(24 * 7):
        hour = (i % 24) + 1
        # Solar only between 6am and 6pm
        solar = max(0, np.sin((hour - 6) * np.pi / 12)) if 6 <= hour <= 18 else 0.0
        sf.update_solar(
            solar=float(solar),
            month=1,
            day=(i // 24) + 1,
            hour=hour
        )
    
    forecast, err = sf.forecast()
    
    assert forecast.shape[0] == 24, f"Expected 24 forecasts, got {forecast.shape[0]}"
    assert (forecast >= 0).all(), "Solar forecast should be non-negative"
    assert np.all(np.isfinite(forecast)), "Forecast contains NaN/Inf"
    print(f"✓ Solar forecaster: mean={np.mean(forecast):.3f}, max={np.max(forecast):.3f}")

def test_carbon_forecaster_with_pretrained(data_dir):
    """Test CUFE carbon forecaster (NN only, no beta file)"""
    model_file = os.path.join(data_dir, "carbon_nn.sav")
    
    # CRITICAL FIX: No beta_file parameter!
    cf = CUFECarbonForecaster(model_file)
    
    # Feed realistic carbon intensity updates
    for _ in range(24):
        cf.update_carbon_intensity(np.random.uniform(0.2, 0.5))
    
    forecast = cf.forecast()
    
    assert forecast.shape[0] == 24, f"Expected 24 forecasts, got {forecast.shape[0]}"
    assert np.all(np.isfinite(forecast)), "Carbon forecast contains NaNs"
    assert np.mean(forecast) > 0, "Carbon forecast should be positive"
    assert np.mean(forecast) < 1.0, "Carbon forecast unrealistically high"
    print(f"✓ Carbon forecaster: mean={np.mean(forecast):.3f}")

# ---------------------------------------------------------
# MPC Solver Tests
# ---------------------------------------------------------

def test_mpcfluid_feasibility():
    """Test MPC optimizer produces valid actions"""
    mpc = MPCFluid()
    
    price = np.linspace(0.1, 0.3, 24)
    carbon = np.linspace(0.2, 0.6, 24)
    consumption = np.full(24, 2.0)
    generation = np.zeros(24)
    
    action = mpc.forecast(
        price, carbon, consumption, generation,
        battery=0.5, 
        net_consumption=1.5,
        error=0.1, 
        max_net_consumption=2.0,
        load_change_weight=0.2
    )
    
    assert isinstance(action, float), f"Action should be float, got {type(action)}"
    assert -1.0 <= action <= 1.0, f"Action {action} out of bounds"
    assert np.isfinite(action), "Action is NaN/Inf"
    print(f"✓ MPC optimizer: action={action:.4f}")

# ---------------------------------------------------------
# Full Agent Integration
# ---------------------------------------------------------

def test_mpc_forecast_algorithm_integration(dummy_env, data_dir):
    """Test full MPC agent integration"""
    cfg = {"data_dir": data_dir}
    agent = MPCForecastAlgorithm(env=dummy_env, config=cfg)

    # Single timestep observation [hour, price, net, soc, solar, carbon]
    obs = [[12, 0.25, 0.5, 0.5, 0.3, 0.4]]
    actions = agent.act(obs)
    
    assert isinstance(actions, list), "Actions should be list"
    assert len(actions) == 1, f"Expected 1 action, got {len(actions)}"
    assert isinstance(actions[0][0], float), f"Action should be float, got {type(actions[0][0])}"
    assert -1.0 <= actions[0][0] <= 1.0, f"Action {actions[0][0]} out of bounds"
    
    # CRITICAL FIX: Use plural attributes
    assert len(agent.power_forecasters) > 0, "Power forecasters not loaded"
    assert len(agent.solar_forecasters) > 0, "Solar forecasters not loaded"
    assert agent.carbon_forecaster is not None, "Carbon forecaster not loaded"
    assert agent.use_pretrained is True, "Should be using pretrained models"
    
    print(f"✓ Agent integration: action={actions[0][0]:.4f}")

# ---------------------------------------------------------
# Pretrained Model Usage Verification
# ---------------------------------------------------------

def test_pretrained_files_loaded_and_used(dummy_env, data_dir):
    """Ensure pretrained models are actually being loaded and used."""
    cfg = {"data_dir": data_dir}
    agent = MPCForecastAlgorithm(env=dummy_env, config=cfg)

    # CRITICAL FIX: Check plural attributes
    assert hasattr(agent, "power_forecasters"), "Missing power_forecasters attribute"
    assert hasattr(agent, "solar_forecasters"), "Missing solar_forecasters attribute"
    assert hasattr(agent, "carbon_forecaster"), "Missing carbon_forecaster attribute"
    
    assert len(agent.power_forecasters) > 0, "No power forecasters loaded"
    assert len(agent.solar_forecasters) > 0, "No solar forecasters loaded"
    assert agent.carbon_forecaster is not None, "No carbon forecaster loaded"
    
    # Check internal structure
    power_f = agent.power_forecasters[0]
    solar_f = agent.solar_forecasters[0]
    carbon_f = agent.carbon_forecaster
    
    assert hasattr(power_f, '_blender'), "Power forecaster missing blender"
    assert hasattr(power_f, 'beta'), "Power forecaster missing AR coefficients"
    assert power_f.beta.shape == (168, 24), f"Wrong beta shape: {power_f.beta.shape}"
    
    assert hasattr(solar_f, '_blender'), "Solar forecaster missing blender"
    assert hasattr(solar_f, 'beta'), "Solar forecaster missing AR coefficients"
    assert solar_f.beta.shape == (168, 24), f"Wrong beta shape: {solar_f.beta.shape}"
    
    assert hasattr(carbon_f, '_model'), "Carbon forecaster missing NN model"
    
    # Feed some data and test forecasts
    for i in range(24):
        power_f.update_power(1.5, 1, 1, i+1)
        solar_f.update_solar(0.5 if 8 <= i < 18 else 0.0, 1, 1, i+1)
        carbon_f.update_carbon_intensity(0.4)
    
    f_p, e_p = power_f.forecast()
    f_s, e_s = solar_f.forecast()
    f_c = carbon_f.forecast()

    assert f_p.shape == (24, 1), f"Power forecast wrong shape: {f_p.shape}"
    assert f_s.shape == (24, 1), f"Solar forecast wrong shape: {f_s.shape}"
    assert f_c.shape[0] == 24, f"Carbon forecast wrong shape: {f_c.shape}"
    
    assert np.all(np.isfinite(f_p)), "Power forecast contains NaN/Inf"
    assert np.all(np.isfinite(f_s)), "Solar forecast contains NaN/Inf"
    assert np.all(np.isfinite(f_c)), "Carbon forecast contains NaN/Inf"
    
    print(f"✓ Pretrained models validated:")
    print(f"  Power: mean={np.mean(f_p):.3f}, error={e_p:.3f}")
    print(f"  Solar: mean={np.mean(f_s):.3f}, error={e_s:.3f}")
    print(f"  Carbon: mean={np.mean(f_c):.3f}")

# ---------------------------------------------------------
# Action Scaling Verification (Supervisor's Critical Fix)
# ---------------------------------------------------------

def test_action_scaling_correct(dummy_env, data_dir):
    """Verify action scaling is correctly applied (capacity/power)"""
    cfg = {"data_dir": data_dir}
    agent = MPCForecastAlgorithm(env=dummy_env, config=cfg)
    
    # Test internal conversion
    cufe_action = 0.5  # CUFE wants to charge at 50% capacity
    current_soc = 0.5
    
    citylearn_action = agent._convert_cufe_action_to_citylearn(cufe_action, current_soc)
    
    # Expected: -0.5 * (6.4 / 5.0) = -0.64
    expected_scale = agent.battery_capacity_kwh / agent.battery_power_kw
    expected_action = -cufe_action * expected_scale
    expected_action = np.clip(expected_action, -1.0, 1.0)
    
    assert np.isclose(citylearn_action, expected_action, atol=0.01), \
        f"Action scaling wrong: got {citylearn_action:.4f}, expected {expected_action:.4f}"
    
    assert abs(citylearn_action) > abs(cufe_action), \
        "CityLearn action should be scaled up (capacity > power)"
    
    print(f"✓ Action scaling verified:")
    print(f"  CUFE action: {cufe_action:.4f}")
    print(f"  Scale factor: {expected_scale:.3f}")
    print(f"  CityLearn action: {citylearn_action:.4f}")
    print(f"  Expected: {expected_action:.4f}")

# ---------------------------------------------------------
# Run all tests
# ---------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])