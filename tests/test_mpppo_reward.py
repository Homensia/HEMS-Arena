import pytest
from hems.rewards.mp_ppo_reward import MPPPOReward

def _info(**k):
    # minimal helper with sane defaults
    d = dict(E_r_t=0.0, E_r_baseline_t=0.0, C_g_t=0.0, C_d_t=0.0, alpha=0.0, thermal_loads=[])
    d.update(k)
    return d

def test_requires_reset():
    r = MPPPOReward(zeta=0.2)
    with pytest.raises(AssertionError):
        r(_info(), soc_t=0.0)

def test_baseline_gap_only_when_worse():
    r = MPPPOReward(zeta=0.2)
    r.reset(soc0=1.0)
    
    # agent worse than baseline -> positive P1
    # E_r_t=10, E_r_baseline_t=7, C_g_t=0.3
    # R1 = 0.3 * (7-10) = -0.9
    # R2a = 0 (no discharge)
    # R2b = 0.2 * 0.3 * 10 = 0.6
    # Total = -0.9 - 0 - 0.6 = -1.5
    rew1 = r(_info(E_r_t=10, E_r_baseline_t=7, C_g_t=0.3), soc_t=1.0)
    assert abs(rew1 + 1.5) < 1e-9
    
    # agent better than baseline -> positive R1
    # R1 = 0.3 * (7-5) = 0.6
    # R2b = 0.2 * 0.3 * 5 = 0.3
    # Total = 0.6 - 0 - 0.3 = 0.3
    rew2 = r(_info(E_r_t=5, E_r_baseline_t=7, C_g_t=0.3), soc_t=1.0)
    assert abs(rew2 - 0.3) < 1e-12

def test_storage_usage_discharge_only_and_alpha_weight():
    r = MPPPOReward(zeta=0.25)  # (1 - zeta) = 0.75
    r.reset(soc0=4.0)
    
    # discharge 1.5 kWh, alpha=0.8 -> factor (1-alpha)=0.2
    info = _info(C_d_t=0.5, alpha=0.8)
    rew = r(info, soc_t=2.5)
    # R2a = (1-0.25)*0.5*1.5*0.2 = 0.75*0.5*1.5*0.2 = 0.1125
    # reward = -R2a = -0.1125
    assert abs(rew + 0.1125) < 1e-9
    
    # charge 1.0 kWh -> no discharge penalty
    rew2 = r(_info(C_d_t=0.5, alpha=0.8), soc_t=3.5)
    assert abs(rew2 - 0.0) < 1e-12

def test_future_residual_grid_term_with_zeta():
    r = MPPPOReward(zeta=0.2)
    r.reset(soc0=1.0)
    
    # Using thermal_loads: [3, 4, 5], sum = 12
    # E_r_t = 6, E_r_baseline_t = 0
    # R1 = 0.3 * (0 - 6) = -1.8
    # R2a = 0 (no discharge)
    # R2b = zeta * Cg * (E_r_t + sum(thermal)) = 0.2 * 0.3 * (6 + 12) = 0.2 * 0.3 * 18 = 1.08
    # Total = -1.8 - 0 - 1.08 = -2.88
    rew = r(_info(E_r_t=6, C_g_t=0.3, thermal_loads=[3, 4, 5]), soc_t=1.0)
    assert abs(rew + 2.88) < 1e-9

def test_combined_penalties_match_hand_calc():
    r = MPPPOReward(zeta=0.2)
    r.reset(soc0=5.0)  # prev_soc
    
    info = _info(
        E_r_t=10.0, E_r_baseline_t=8.0,  # worse by 2
        C_g_t=0.3,
        C_d_t=0.05,
        alpha=0.9,
        thermal_loads=[9.0, 7.0, 6.0]  # sum=22
    )
    soc_t = 4.0  # discharged 1.0
    
    # R1 = 0.3 * (8-10) = -0.6
    # R2a = (1-0.2) * 0.05 * 1.0 * (1-0.9) = 0.8*0.05*0.1 = 0.004
    # R2b = 0.2 * 0.3 * (10 + 22) = 0.06 * 32 = 1.92
    # total = -0.6 - 0.004 - 1.92 = -2.524
    rew = r(info, soc_t=soc_t)
    assert abs(rew + 2.524) < 1e-9

def test_zeta_extremes_gate_terms():
    # zeta=0 -> no future grid term; only R1 and R2a
    r0 = MPPPOReward(zeta=0.0)
    r0.reset(soc0=2.0)
    # R1 = 1.0 * (1-6) = -5
    # R2a = (1-0)*2*1*(1-0.5) = 1.0
    # R2b = 0 * ... = 0
    # total = -5 - 1.0 - 0 = -6
    rew0 = r0(_info(E_r_t=6, E_r_baseline_t=1, C_g_t=1.0, C_d_t=2.0, alpha=0.5, thermal_loads=[100, 100]), soc_t=1.0)
    assert abs(rew0 + 6.0) < 1e-9
    
    # zeta=1 -> no storage usage term; only R1 and R2b
    r1 = MPPPOReward(zeta=1.0)
    r1.reset(soc0=2.0)
    # R1 = 1.0 * (1-6) = -5
    # R2a = (1-1) * ... = 0
    # R2b = 1 * 1.0 * (6 + (3+4+5)) = 1.0 * 18 = 18
    # total = -5 - 0 - 18 = -23
    rew1 = r1(_info(E_r_t=6, E_r_baseline_t=1, C_g_t=1.0, C_d_t=2.0, alpha=0.5, thermal_loads=[3, 4, 5]), soc_t=1.0)
    assert abs(rew1 + 23.0) < 1e-9

def test_state_updates_between_calls():
    r = MPPPOReward(zeta=0.5)
    r.reset(soc0=4.0)
    
    # step 1: discharge 1.0
    rew1 = r(_info(C_d_t=1.0, alpha=0.0), soc_t=3.0)
    # R2a = (1-0.5)*1*1*(1-0)=0.5 -> reward=-0.5
    assert abs(rew1 + 0.5) < 1e-9
    
    # step 2: charge 0.5 -> no discharge penalty
    rew2 = r(_info(C_d_t=1.0, alpha=0.0), soc_t=3.5)
    assert abs(rew2 - 0.0) < 1e-12