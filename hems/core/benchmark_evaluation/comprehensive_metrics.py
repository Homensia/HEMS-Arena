"""
Comprehensive Metrics Calculator
Implements all 55+ KPIs from HEMS Benchmark KPI specification.

Categories:
1. Energy Economics (12 KPIs)
2. Renewable Energy (10 KPIs)
3. Battery Performance (15 KPIs)
4. AI/Learning Performance (10 KPIs)
5. Statistical/Research (8 KPIs)

Sign Convention (CityLearn):
- PV generation is NEGATIVE: must rectify with pv_gen = max(0, -pv)
- net > 0 = import, net < 0 = export
"""

import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ComprehensiveMetrics:
    """
    Calculate all KPIs following the definitive specification.
    """
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """Initialize calculator."""
        self.logger = logger_instance or logger
        self.eps = 1e-9  # Small epsilon for division protection
    
    def calculate_all_kpis(
        self,
        agent_data: Dict[str, Any],
        training_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate all KPIs from agent data.
        
        Args:
            agent_data: Testing/validation phase data with aggregated arrays
            training_data: Optional training phase data for learning metrics
            
        Returns:
            Dict with all KPIs organized by category
        """
        if 'aggregated' not in agent_data:
            self.logger.warning("No aggregated data available")
            return self._empty_kpis()
        
        agg = agent_data['aggregated']
        
        # Extract and rectify arrays
        net = np.array(agg.get('net_consumption', []))
        pv_raw = np.array(agg.get('pv_generation', []))
        pv = np.maximum(0, -pv_raw)  # RECTIFY: CityLearn PV is negative
        price = np.array(agg.get('electricity_price', []))
        soc = np.array(agg.get('battery_soc', []))
        imp = np.array(agg.get('import_grid', []))
        exp = np.array(agg.get('export_grid', []))
        
        # Get episode data for per-episode metrics
        episode_data = agent_data.get('episode_data', [])
        
        # Calculate consumption (building load)
        # consumption = net + pv (what was consumed from grid + local PV)
        # For now, approximate from available data
        consumption = np.abs(net) + pv  # Simplified
        
        # Training rewards
        training_rewards = None
        if training_data:
            training_rewards = self._extract_training_rewards(training_data)
        
        # Testing rewards
        test_rewards = agent_data.get('rewards', [])
        
        # Calculate all categories
        kpis = {}
        kpis['energy_economics'] = self._energy_economics(
            imp, exp, price, net, episode_data
        )
        kpis['renewable_energy'] = self._renewable_energy(
            pv, consumption, exp, episode_data
        )
        kpis['battery_performance'] = self._battery_performance(
            soc, episode_data, len(pv)
        )
        kpis['ai_learning'] = self._ai_learning(
            training_rewards, test_rewards, training_data
        )
        kpis['statistical_research'] = self._statistical_research(
            training_rewards, test_rewards
        )
        
        # Flatten for easy access
        flat_kpis = {}
        for category, metrics in kpis.items():
            flat_kpis.update(metrics)
        
        return {
            'by_category': kpis,
            'flat': flat_kpis
        }
    
    def _energy_economics(
        self,
        imp: np.ndarray,
        exp: np.ndarray,
        price: np.ndarray,
        net: np.ndarray,
        episode_data: List[Dict]
    ) -> Dict[str, float]:
        """
        Energy Economics KPIs (12 metrics).
        """
        # Handle missing prices
        if len(price) == 0 or np.all(price == 0):
            price = np.full_like(imp, 0.22)
        
        # Export price (80% of import)
        price_sell = price * 0.8
        
        # 1. Total Electricity Cost
        total_cost = np.sum(imp * price) - np.sum(exp * price_sell)
        
        # 2. Average Cost per Episode
        n_episodes = max(len(episode_data), 1)
        avg_cost_per_ep = total_cost / n_episodes
        
        # 3. Cost Volatility (CV of episode costs)
        episode_costs = self._calculate_episode_costs(episode_data, price)
        cost_volatility = np.std(episode_costs) / (np.abs(np.mean(episode_costs)) + self.eps)
        
        # 4. Peak Demand (average across episodes)
        peak_demands = []
        for ep in episode_data:
            td = ep.get('timestep_data', {})
            ep_imp = td.get('import_grid', [])
            if len(ep_imp) > 0:
                peak_demands.append(np.max(ep_imp))
        peak_demand_avg = np.mean(peak_demands) if peak_demands else np.max(imp)
        
        # 5. Max Peak Demand
        max_peak_demand = np.max([p for p in peak_demands]) if peak_demands else np.max(imp)
        
        # 6. Load Factor
        total_imp = np.sum(imp)
        T = len(imp)
        avg_power = total_imp / (T + self.eps)
        load_factor = avg_power / (np.max(imp) + self.eps)
        
        # 7. Total Consumption (from grid)
        total_import_kwh = np.sum(imp)
        
        # 8. Total Export (to grid)
        total_export_kwh = np.sum(exp)
        
        # 9. Net Consumption
        net_consumption = np.sum(net)
        
        # 10. Energy Efficiency Index (training improvement)
        # Requires training data - placeholder
        eei = 1.0
        
        # 11. Demand Response Score
        sigma_peak = np.std(peak_demands) if len(peak_demands) > 1 else 0
        peak_max = max_peak_demand
        dr_score = max(0, 1 - sigma_peak / (peak_max + self.eps))
        
        # 12. Grid Interaction Cost (per kWh)
        total_interaction = np.sum(imp) + np.sum(exp)
        gic = total_cost / (total_interaction + self.eps)
        
        return {
            'total_cost': float(total_cost),
            'avg_cost_per_episode': float(avg_cost_per_ep),
            'cost_volatility': float(cost_volatility),
            'peak_demand_avg': float(peak_demand_avg),
            'max_peak_demand': float(max_peak_demand),
            'load_factor': float(load_factor),
            'total_import_kwh': float(total_import_kwh),
            'total_export_kwh': float(total_export_kwh),
            'net_consumption': float(net_consumption),
            'energy_efficiency_index': float(eei),
            'demand_response_score': float(dr_score),
            'grid_interaction_cost': float(gic),
        }
    
    def _renewable_energy(
        self,
        pv: np.ndarray,
        consumption: np.ndarray,
        exp: np.ndarray,
        episode_data: List[Dict]
    ) -> Dict[str, float]:
        """
        Renewable Energy KPIs (10 metrics).
        """
        # 1. Total PV Generation
        pv_total = np.sum(pv)
        
        if pv_total < self.eps:
            return self._empty_renewable_kpis()
        
        # 2. Total PV Self-Consumed
        # PV_self = min(PV_gen, consumption)
        pv_self = np.sum(np.minimum(pv, np.maximum(0, consumption)))
        
        # 3. PV Self-Consumption Rate
        scr = 100 * pv_self / (pv_total + self.eps)
        
        # 4. PV Self-Sufficiency Ratio
        total_consumption = np.sum(consumption)
        ssr = 100 * pv_self / (total_consumption + self.eps)
        
        # 5. Renewable Fraction (same as SSR for PV-only)
        rf = ssr
        
        # 6. PV Curtailment Rate
        # Curtailment = PV not used and not exported
        pv_curtailed = np.sum(np.maximum(0, pv - pv_self - exp))
        curt_rate = 100 * pv_curtailed / (pv_total + self.eps)
        
        # 7. PV Utilization Efficiency
        pv_utilized = pv_self + np.sum(exp)
        pue = 100 * pv_utilized / (pv_total + self.eps)
        
        # 8. Grid Carbon Avoidance (assume 0.5 kg CO2/kWh)
        grid_intensity = 0.5
        co2_avoided = pv_self * grid_intensity
        
        # 9. Renewable Energy Value (€ from self-consumed PV)
        # Need prices - approximate with 0.22 €/kWh
        avg_price = 0.22
        vre = pv_self * avg_price
        
        # 10. Seasonal PV Performance
        episode_pv = []
        for ep in episode_data:
            td = ep.get('timestep_data', {})
            ep_pv_raw = td.get('pv_generation', [])
            if len(ep_pv_raw) > 0:
                ep_pv_rect = np.maximum(0, -np.array(ep_pv_raw))
                episode_pv.append(np.sum(ep_pv_rect))
        
        if len(episode_pv) > 1:
            seasonal_score = max(0, 1 - np.std(episode_pv) / (np.mean(episode_pv) + self.eps))
        else:
            seasonal_score = 0.5
        
        return {
            'pv_total_generation': float(pv_total),
            'pv_self_consumption': float(pv_self),
            'pv_self_consumption_rate': float(scr),
            'pv_self_sufficiency_ratio': float(ssr),
            'renewable_fraction': float(rf),
            'pv_curtailment_rate': float(curt_rate),
            'pv_utilization_efficiency': float(pue),
            'co2_avoided': float(co2_avoided),
            'renewable_energy_value': float(vre),
            'seasonal_pv_score': float(seasonal_score),
        }
    
    def _battery_performance(
        self,
        soc: np.ndarray,
        episode_data: List[Dict],
        total_steps: int
    ) -> Dict[str, float]:
        """
        Battery Performance KPIs (15 metrics).
        """
        if len(soc) == 0:
            return self._empty_battery_kpis()
        
        # Assume battery capacity (default 6.5 kWh if not specified)
        C_bat = 6.5
        
        # 1. Battery Cycles per Episode
        soc_changes = np.abs(np.diff(soc))
        total_throughput = np.sum(soc_changes) / 2.0
        n_episodes = max(len(episode_data), 1)
        cycles_per_ep = total_throughput / n_episodes
        
        # 2. Total Battery Cycles
        total_cycles = total_throughput
        
        # 3. Battery SoC Mean
        soc_mean = np.mean(soc) * 100
        
        # 4. Battery SoC Std
        soc_std = np.std(soc) * 100
        
        # 5. Battery Utilization Range
        util_range = (np.max(soc) - np.min(soc)) * 100
        
        # 6. Battery Throughput (SoC changes)
        throughput_soc = np.sum(np.abs(np.diff(soc)))
        
        # 7. Equivalent Full Cycles (from throughput)
        efc = throughput_soc / 2.0
        
        # 8. Depth of Discharge Mean (per episode)
        episode_dods = []
        for ep in episode_data:
            td = ep.get('timestep_data', {})
            ep_soc = td.get('battery_soc', [])
            if len(ep_soc) > 0:
                dod = np.max(ep_soc) - np.min(ep_soc)
                episode_dods.append(dod)
        dod_mean = np.mean(episode_dods) * 100 if episode_dods else 0
        
        # 9. Cycle Efficiency (assume 90% round-trip)
        cycle_efficiency = 90.0
        
        # 10. Charging Pattern Regularity
        regularity = 1.0 / (1.0 + soc_std / 100)
        
        # 11. Battery Capacity Utilization
        capacity_util = util_range
        
        # 12. Charge/Discharge Ratio
        # Count charge vs discharge cycles
        charge_ratio = 1.0  # Placeholder
        
        # 13. Battery Health Score
        ref_cycles = 5000
        health_score = (
            max(0, 1 - total_cycles / ref_cycles) +
            min(1, util_range / 60) +
            (1 - abs(dod_mean / 100 - 0.6))
        ) / 3.0
        
        # 14. Optimal SoC Usage (% time in 20-80% range)
        optimal_range = np.sum((soc >= 0.2) & (soc <= 0.8))
        optimal_pct = 100 * optimal_range / (len(soc) + self.eps)
        
        # 15. Battery Value Captured
        # Requires baseline comparison - placeholder
        battery_value = 0.0
        
        return {
            'battery_cycles_per_episode': float(cycles_per_ep),
            'battery_cycles_total': float(total_cycles),
            'battery_soc_mean': float(soc_mean),
            'battery_soc_std': float(soc_std),
            'battery_utilization_range': float(util_range),
            'battery_throughput_soc': float(throughput_soc),
            'equivalent_full_cycles': float(efc),
            'depth_of_discharge_mean': float(dod_mean),
            'cycle_efficiency': float(cycle_efficiency),
            'charging_pattern_regularity': float(regularity),
            'battery_capacity_utilization': float(capacity_util),
            'charge_discharge_ratio': float(charge_ratio),
            'battery_health_score': float(health_score),
            'optimal_soc_usage': float(optimal_pct),
            'battery_value_captured': float(battery_value),
        }
    
    def _ai_learning(
        self,
        training_rewards: Optional[np.ndarray],
        test_rewards: List[float],
        training_data: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        AI/Learning Performance KPIs (10 metrics).
        """
        # Use training rewards if available, otherwise use test rewards for basic metrics
        if training_rewards is None or len(training_rewards) == 0:
            # During testing/validation, use test rewards for basic metrics
            if test_rewards and len(test_rewards) > 0:
                R = np.array(test_rewards)
                return {
                    'avg_reward': float(np.mean(R)),
                    'reward_std': float(np.std(R)),
                    'learning_improvement': 0.0,
                    'convergence_speed': 0.0,
                    'training_time': 0.0,
                    'sample_efficiency': 0.0,
                    'action_consistency': 0.0,
                    'exploration_efficiency': 0.0,
                    'policy_stability': 0.0,
                    'training_stability_index': 0.0,
                }
            return self._empty_ai_kpis()
        
        R = training_rewards
        N = len(R)
        
        # When we have training data, we calculate training-specific metrics
        # but use test_rewards for avg_reward and reward_std (the actual test performance)
        test_reward_mean = np.mean(test_rewards) if test_rewards else 0.0
        test_reward_std = np.std(test_rewards) if test_rewards else 0.0
        
        # Training-based reward stats (for comparison)
        training_reward_mean = np.mean(R)
        training_reward_std = np.std(R)
        
        # 3. Learning Improvement
        window = min(10, N // 10)
        if N >= 2 * window:
            early = R[:window]
            late = R[-window:]
            learning_improvement = np.mean(late) - np.mean(early)
        else:
            learning_improvement = 0.0
        
        # 4. Convergence Speed (episodes to reach 90% of max)
        max_reward = np.max(R)
        threshold = 0.9 * max_reward
        converged_at = N
        for i, r in enumerate(R):
            if r >= threshold:
                converged_at = i + 1
                break
        convergence_speed = converged_at
        
        # 5. Training Time (if available)
        training_time = 0.0
        if training_data:
            # training_data structure: {'status': 'completed', 'results': {...}}
            results = training_data.get('results', {})
            training_time = results.get('training_time', 0.0)
        
        # 6. Sample Efficiency
        sample_efficiency = learning_improvement / max(1, N)
        
        # 7. Action Consistency
        if N >= window:
            late_rewards = R[-window:]
            late_std = np.std(late_rewards)
            late_mean = np.mean(late_rewards)
            action_consistency = 1.0 / (1.0 + late_std / (abs(late_mean) + self.eps))
        else:
            action_consistency = 0.5
        
        # 8. Exploration Efficiency
        kappa = 10
        reward_growth = abs(R[-1] - R[0]) / (N + self.eps)
        exploration_eff = min(1.0, reward_growth / kappa)
        
        # 9. Policy Stability
        if N >= window:
            var_late = np.var(R[-window:])
            policy_stability = 1.0 / (1.0 + var_late)
        else:
            policy_stability = 0.5
        
        # 10. Training Stability Index
        reward_range = np.max(R) - np.min(R)
        tsi = max(0, min(1, 1 - training_reward_std / (reward_range + self.eps)))
        
        return {
            'avg_reward': float(test_reward_mean),  # Use TEST rewards for actual performance
            'reward_std': float(test_reward_std),   # Use TEST rewards for actual performance
            'learning_improvement': float(learning_improvement),
            'convergence_speed': int(convergence_speed),
            'training_time': float(training_time),
            'sample_efficiency': float(sample_efficiency),
            'action_consistency': float(action_consistency),
            'exploration_efficiency': float(exploration_eff),
            'policy_stability': float(policy_stability),
            'training_stability_index': float(tsi),
        }
    
    def _statistical_research(
        self,
        training_rewards: Optional[np.ndarray],
        test_rewards: List[float]
    ) -> Dict[str, float]:
        """
        Statistical/Research KPIs (8 metrics).
        
        These metrics measure TEST performance statistics.
        Only generalization_gap compares training vs test.
        """
        # Use TEST rewards for all statistical KPIs (they measure test performance)
        R_test = np.array(test_rewards) if test_rewards and len(test_rewards) > 0 else np.array([0])
        
        # Training rewards only needed for generalization gap
        R_train = training_rewards if training_rewards is not None and len(training_rewards) > 0 else R_test
        
        n = len(R_test)
        
        # 1. 95% Confidence Interval (half-width) - TEST performance
        ci_95 = 1.96 * np.std(R_test) / np.sqrt(n + self.eps)
        
        # 2. Worst-Case Performance - TEST performance
        worst_case = np.min(R_test)
        
        # 3. Best-Case Performance - TEST performance
        best_case = np.max(R_test)
        
        # 4. Reward Variance - TEST performance
        reward_variance = np.var(R_test)
        
        # 5. Performance Consistency - TEST performance
        cv = np.std(R_test) / (abs(np.mean(R_test)) + self.eps)
        performance_consistency = 1.0 / (1.0 + cv)
        
        # 6. Generalization Gap - difference between TRAINING and TEST
        mean_train = np.mean(R_train)
        mean_test = np.mean(R_test)
        gen_gap = mean_test - mean_train
        gen_gap_pct = 100 * gen_gap / (abs(mean_train) + self.eps)
        
        # 7. Robustness Score - TEST performance
        robustness = np.min(R_test) / (np.mean(R_test) + self.eps)
        
        # 8. Stability Index (long-term) - TEST performance
        tail_size = min(20, n // 5)
        if n >= tail_size:
            R_tail = R_test[-tail_size:]
            tail_std = np.std(R_tail)
            tail_mean = np.mean(R_tail)
            stability_longterm = 1.0 / (1.0 + tail_std / (abs(tail_mean) + self.eps))
        else:
            stability_longterm = 0.5
        
        return {
            'ci_95_halfwidth': float(ci_95),
            'worst_case_performance': float(worst_case),
            'best_case_performance': float(best_case),
            'reward_variance': float(reward_variance),
            'performance_consistency': float(performance_consistency),
            'generalization_gap': float(gen_gap),
            'generalization_gap_percent': float(gen_gap_pct),
            'robustness_score': float(robustness),
            'stability_index_longterm': float(stability_longterm),
        }
    
    def _calculate_episode_costs(
        self,
        episode_data: List[Dict],
        price: np.ndarray
    ) -> np.ndarray:
        """Calculate cost for each episode."""
        costs = []
        for ep in episode_data:
            td = ep.get('timestep_data', {})
            imp = np.array(td.get('import_grid', []))
            exp = np.array(td.get('export_grid', []))
            
            if len(imp) > 0:
                ep_price = price[:len(imp)] if len(price) >= len(imp) else np.full(len(imp), 0.22)
                cost = np.sum(imp * ep_price) - np.sum(exp * ep_price * 0.8)
                costs.append(cost)
        
        return np.array(costs) if costs else np.array([0])
    
    def _extract_training_rewards(self, training_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract training rewards from training data."""
        if not training_data:
            return None
        
        results = training_data.get('results', {})
        
        # Sequential mode - buildings is a DICT with per-building results
        if 'buildings' in results and isinstance(results['buildings'], dict):
            all_rewards = []
            for building_results in results['buildings'].values():
                rewards = building_results.get('rewards', [])
                all_rewards.extend(rewards)
            return np.array(all_rewards) if all_rewards else None
        
        # Parallel/round-robin mode - rewards at top level
        if 'rewards' in results:
            return np.array(results['rewards'])
        
        return None
    
    def _empty_kpis(self) -> Dict[str, Any]:
        """Return empty KPI structure."""
        return {
            'by_category': {
                'energy_economics': self._empty_economics_kpis(),
                'renewable_energy': self._empty_renewable_kpis(),
                'battery_performance': self._empty_battery_kpis(),
                'ai_learning': self._empty_ai_kpis(),
                'statistical_research': self._empty_statistical_kpis(),
            },
            'flat': {}
        }
    
    def _empty_economics_kpis(self) -> Dict[str, float]:
        return {k: 0.0 for k in [
            'total_cost', 'avg_cost_per_episode', 'cost_volatility',
            'peak_demand_avg', 'max_peak_demand', 'load_factor',
            'total_import_kwh', 'total_export_kwh', 'net_consumption',
            'energy_efficiency_index', 'demand_response_score', 'grid_interaction_cost'
        ]}
    
    def _empty_renewable_kpis(self) -> Dict[str, float]:
        return {k: 0.0 for k in [
            'pv_total_generation', 'pv_self_consumption', 'pv_self_consumption_rate',
            'pv_self_sufficiency_ratio', 'renewable_fraction', 'pv_curtailment_rate',
            'pv_utilization_efficiency', 'co2_avoided', 'renewable_energy_value',
            'seasonal_pv_score'
        ]}
    
    def _empty_battery_kpis(self) -> Dict[str, float]:
        return {k: 0.0 for k in [
            'battery_cycles_per_episode', 'battery_cycles_total', 'battery_soc_mean',
            'battery_soc_std', 'battery_utilization_range', 'battery_throughput_soc',
            'equivalent_full_cycles', 'depth_of_discharge_mean', 'cycle_efficiency',
            'charging_pattern_regularity', 'battery_capacity_utilization',
            'charge_discharge_ratio', 'battery_health_score', 'optimal_soc_usage',
            'battery_value_captured'
        ]}
    
    def _empty_ai_kpis(self) -> Dict[str, float]:
        return {k: 0.0 for k in [
            'avg_reward', 'reward_std', 'learning_improvement', 'convergence_speed',
            'training_time', 'sample_efficiency', 'action_consistency',
            'exploration_efficiency', 'policy_stability', 'training_stability_index'
        ]}
    
    def _empty_statistical_kpis(self) -> Dict[str, float]:
        return {k: 0.0 for k in [
            'ci_95_halfwidth', 'worst_case_performance', 'best_case_performance',
            'reward_variance', 'performance_consistency', 'generalization_gap',
            'generalization_gap_percent', 'robustness_score', 'stability_index_longterm'
        ]}