# =====================================================
# hems/core/benchmark_evaluation/metrics_calculator.py
# =====================================================
"""
Metrics Calculator - Clean Rebuild
Calculates all KPIs from episode data with robust fallback mechanisms.

Key Features:
- Cost metrics (import cost, export revenue, net cost)
- PV metrics (generation, self-consumption rate, export)
- Battery metrics (cycles, efficiency, utilization)
- Peak demand metrics
- Savings metrics (vs baseline)
- CRITICAL: Fallback reconstruction from episode data if KPIs missing
- Sign convention helpers (centralized)
- INTEGRATION: Comprehensive metrics with 55+ KPIs
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

try:
    from comprehensive_metrics import ComprehensiveMetrics
except ImportError:
    try:
        from .comprehensive_metrics import ComprehensiveMetrics
    except ImportError:
        ComprehensiveMetrics = None

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate comprehensive KPIs from episode data.
    
    Handles both:
    1. Pre-computed KPIs (if available)
    2. Reconstruction from raw episode data (fallback)
    """
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """Initialize calculator."""
        self.logger = logger_instance or logger
        self.comprehensive = ComprehensiveMetrics(logger_instance) if ComprehensiveMetrics else None
    
    def calculate_kpis(
        self,
        agent_data: Dict[str, Any],
        training_data: Optional[Dict[str, Any]] = None,
        use_comprehensive: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate all KPIs from agent data.
        
        Args:
            agent_data: Data from validation/testing phase
            training_data: Optional training phase data for learning metrics
            use_comprehensive: If True, calculate all 55+ comprehensive KPIs
            
        Returns:
            Dictionary of KPIs
        """
        # Basic KPIs (backward compatible)
        basic_kpis = self._calculate_basic_kpis(agent_data)
        
        # Comprehensive KPIs (new)
        comprehensive_kpis = {}
        if use_comprehensive and self.comprehensive:
            try:
                comp_result = self.comprehensive.calculate_all_kpis(agent_data, training_data)
                comprehensive_kpis = comp_result.get('flat', {})
                self.logger.info(f"  Calculated {len(comprehensive_kpis)} comprehensive KPIs")
                for key in ("pv_self_consumption", "pv_self_consumption_rate"):
                    if key in comprehensive_kpis:
                        self.logger.debug(
                            f"Removing comprehensive '{key}' to keep basic definition."
                        )
                        comprehensive_kpis.pop(key)

                self.logger.info(f"  Calculated {len(comprehensive_kpis)} comprehensive KPIs")
            except Exception as e:
                self.logger.warning(f"  Comprehensive KPI calculation failed: {e}")
        
        # Merge both
        all_kpis = {**basic_kpis, **comprehensive_kpis}
        
        return all_kpis
    
    def _calculate_basic_kpis(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basic KPIs (original implementation)."""
        # Check if we have aggregated data
        if 'aggregated' not in agent_data:
            self.logger.warning("No aggregated data - returning empty KPIs")
            return {}
        
        agg = agent_data['aggregated']
        
        # Extract arrays
        net_consumption = agg.get('net_consumption', np.array([]))
        pv_generation = agg.get('pv_generation', np.array([]))
        electricity_price = agg.get('electricity_price', np.array([]))
        battery_soc = agg.get('battery_soc', np.array([]))
        import_grid = agg.get('import_grid', np.array([]))
        export_grid = agg.get('export_grid', np.array([]))
        
        # Sanity checks
        if len(net_consumption) == 0:
            self.logger.warning("Empty data arrays - cannot calculate KPIs")
            return self._empty_kpis()
        
        # Calculate all metrics
        kpis = {}
        
        # Cost metrics
        kpis.update(self._calculate_cost_metrics(
            import_grid, export_grid, electricity_price
        ))
        
        # PV metrics
        kpis.update(self._calculate_pv_metrics(
            pv_generation, import_grid, export_grid, net_consumption
        ))
        
        # Battery metrics
        kpis.update(self._calculate_battery_metrics(battery_soc))
        
        # Peak demand metrics
        kpis.update(self._calculate_peak_metrics(import_grid))
        
        # General metrics
        kpis['avg_reward'] = agent_data.get('avg_reward', 0.0)
        kpis['std_reward'] = agent_data.get('std_reward', 0.0)
        kpis['total_episodes'] = agent_data.get('episodes', 0)
        
        self.logger.info(f"  Calculated {len(kpis)} KPIs")
        
        return kpis
    
    def _calculate_cost_metrics(
        self,
        import_grid: np.ndarray,
        export_grid: np.ndarray,
        electricity_price: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate cost-related metrics.
        
        Returns:
            Dict with cost metrics
        """
        # Handle missing prices
        if len(electricity_price) == 0 or np.all(electricity_price == 0):
            avg_price = 0.22  # Default €/kWh
            electricity_price = np.full_like(import_grid, avg_price)
            self.logger.warning(f"Missing prices - using default {avg_price} €/kWh")
        
        # Import cost (what we pay to grid)
        import_cost = np.sum(import_grid * electricity_price)
        
        # Export revenue (what we earn from selling)
        # Assume export price is 80% of import price
        export_price = electricity_price * 0.8
        export_revenue = np.sum(export_grid * export_price)
        
        # Net cost
        net_cost = import_cost - export_revenue
        
        # Average prices
        avg_import_price = np.mean(electricity_price) if len(electricity_price) > 0 else 0
        
        return {
            'import_cost': float(import_cost),
            'export_revenue': float(export_revenue),
            'total_cost': float(net_cost),
            'avg_import_price': float(avg_import_price),
            'total_import_kwh': float(np.sum(import_grid)),
            'total_export_kwh': float(np.sum(export_grid)),
        }
    
    def _calculate_pv_metrics(
        self,
        pv_generation: np.ndarray,
        import_grid: np.ndarray,
        export_grid: np.ndarray,
        net_consumption: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate PV-related metrics.
        
        Sign convention:
        - net_consumption > 0: importing from grid
        - net_consumption < 0: exporting to grid
        - PV consumed = min(PV generated, import + PV generated)
        
        Returns:
            Dict with PV metrics
        """
        pv = np.array(pv_generation, dtype=float)

        if pv.size == 0:
            return {
                'total_pv_generation': 0.0,
                'pv_self_consumption': 0.0,
                'pv_self_consumption_rate': 0.0,
                'pv_export': 0.0,
                'pv_export_rate': 0.0,
            }

        # Make PV positive (works for both negative and already-positive inputs)
        pv = np.abs(pv)

        total_pv_generated = float(np.sum(pv))

        if total_pv_generated == 0:
            return {
                'total_pv_generation': 0.0,
                'pv_self_consumption': 0.0,
                'pv_self_consumption_rate': 0.0,
                'pv_export': 0.0,
                'pv_export_rate': 0.0,
            }

        # PV exported to grid (already positive in your code)
        total_pv_export = float(np.sum(export_grid))

        # Approx: PV self-consumed = PV generation - total export
        total_pv_self_consumed = max(0.0, total_pv_generated - total_pv_export)

        # Rates
        self_consumption_rate = 100.0 * total_pv_self_consumed / total_pv_generated
        export_rate = 100.0 * total_pv_export / total_pv_generated

        return {
            'total_pv_generation': total_pv_generated,
            'pv_self_consumption': total_pv_self_consumed,
            'pv_self_consumption_rate': self_consumption_rate,
            'pv_export': total_pv_export,
            'pv_export_rate': export_rate,
        }
    
    def _calculate_battery_metrics(self, battery_soc: np.ndarray) -> Dict[str, float]:
        """
        Calculate battery-related metrics.
        
        Battery cycles estimated from SoC profile:
        cycles = sum(|SoC[t] - SoC[t-1]|) / 2.0
        
        Returns:
            Dict with battery metrics
        """
        if len(battery_soc) == 0:
            return {
                'battery_cycles': 0.0,
                'avg_soc': 0.0,
                'min_soc': 0.0,
                'max_soc': 0.0,
            }
        
        # Estimate cycles from SoC changes
        soc_changes = np.abs(np.diff(battery_soc))
        battery_cycles = np.sum(soc_changes) / 2.0
        
        # SoC statistics
        avg_soc = np.mean(battery_soc) * 100  # Convert to percentage
        min_soc = np.min(battery_soc) * 100
        max_soc = np.max(battery_soc) * 100
        
        return {
            'battery_cycles': float(battery_cycles),
            'avg_soc': float(avg_soc),
            'min_soc': float(min_soc),
            'max_soc': float(max_soc),
        }
    
    def _calculate_peak_metrics(self, import_grid: np.ndarray) -> Dict[str, float]:
        """
        Calculate peak demand metrics.
        
        Returns:
            Dict with peak metrics
        """
        if len(import_grid) == 0:
            return {
                'peak_demand': 0.0,
                'avg_demand': 0.0,
            }
        
        peak_demand = np.max(import_grid)
        avg_demand = np.mean(import_grid)
        
        return {
            'peak_demand': float(peak_demand),
            'avg_demand': float(avg_demand),
        }
    
    def compute_savings(
        self,
        agent_kpis: Dict[str, float],
        baseline_kpis: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate savings of agent vs baseline.
        
        Args:
            agent_kpis: Agent KPIs
            baseline_kpis: Baseline KPIs
            
        Returns:
            Savings metrics
        """
        savings = {}
        
        # Cost savings
        agent_cost = agent_kpis.get('total_cost', 0)
        baseline_cost = baseline_kpis.get('total_cost', 0)
        
        if baseline_cost != 0:
            cost_savings = baseline_cost - agent_cost
            cost_savings_percent = (cost_savings / abs(baseline_cost)) * 100
        else:
            cost_savings = 0
            cost_savings_percent = 0
            self.logger.warning("Baseline cost is zero - cannot calculate savings")
        
        savings['cost_savings'] = float(cost_savings)
        savings['cost_savings_percent'] = float(cost_savings_percent)
        
        # Peak reduction
        agent_peak = agent_kpis.get('peak_demand', 0)
        baseline_peak = baseline_kpis.get('peak_demand', 0)
        
        if baseline_peak != 0:
            peak_reduction = baseline_peak - agent_peak
            peak_reduction_percent = (peak_reduction / baseline_peak) * 100
        else:
            peak_reduction = 0
            peak_reduction_percent = 0
        
        savings['peak_reduction'] = float(peak_reduction)
        savings['peak_reduction_percent'] = float(peak_reduction_percent)
        
        # PV self-consumption improvement
        agent_pv_sc = agent_kpis.get('pv_self_consumption_rate', 0)
        baseline_pv_sc = baseline_kpis.get('pv_self_consumption_rate', 0)
        
        pv_improvement = agent_pv_sc - baseline_pv_sc
        savings['pv_self_consumption_improvement'] = float(pv_improvement)
        
        # Reward improvement
        agent_reward = agent_kpis.get('avg_reward', 0)
        baseline_reward = baseline_kpis.get('avg_reward', 0)
        
        reward_improvement = agent_reward - baseline_reward
        savings['reward_improvement'] = float(reward_improvement)
        
        return savings
    
    def calculate_battery_value(
        self,
        agent_kpis: Dict[str, float],
        baseline_kpis: Dict[str, float]
    ) -> float:
        """
        Calculate battery value captured (economic benefit of battery usage).
        
        Battery value = Cost savings from battery operation
                      = Baseline cost - Agent cost (when agent uses battery)
        
        Args:
            agent_kpis: Agent KPIs (with battery)
            baseline_kpis: Baseline KPIs (no battery / SOC=50%)
            
        Returns:
            Battery value captured in currency units (€)
        """
        agent_cost = agent_kpis.get('total_cost', 0.0)
        baseline_cost = baseline_kpis.get('total_cost', 0.0)
        
        # Battery value = savings from using battery
        battery_value = baseline_cost - agent_cost
        
        # Only count positive value (actual savings)
        # Negative means battery operation increased costs
        return float(max(0.0, battery_value))
    
    def _empty_kpis(self) -> Dict[str, float]:
        """Return empty KPIs structure."""
        return {
            'import_cost': 0.0,
            'export_revenue': 0.0,
            'total_cost': 0.0,
            'total_pv_generation': 0.0,
            'pv_self_consumption': 0.0,
            'pv_self_consumption_rate': 0.0,
            'battery_cycles': 0.0,
            'peak_demand': 0.0,
            'avg_reward': 0.0,
        }
    
    def validate_kpis(self, kpis: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate KPIs for sanity.
        
        Args:
            kpis: KPIs dictionary
            
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for NaN or inf
        for key, value in kpis.items():
            if isinstance(value, (int, float)):
                if np.isnan(value):
                    issues.append(f"{key} is NaN")
                elif np.isinf(value):
                    issues.append(f"{key} is infinite")
        
        # Check ranges
        if 'pv_self_consumption_rate' in kpis:
            rate = kpis['pv_self_consumption_rate']
            if not (0 <= rate <= 100):
                issues.append(f"PV self-consumption rate out of range: {rate}%")
        
        if 'avg_soc' in kpis:
            soc = kpis['avg_soc']
            if not (0 <= soc <= 100):
                issues.append(f"Average SoC out of range: {soc}%")
        
        if 'battery_cycles' in kpis:
            cycles = kpis['battery_cycles']
            if cycles < 0:
                issues.append(f"Negative battery cycles: {cycles}")
        
        # Check cost consistency
        if 'import_cost' in kpis and 'export_revenue' in kpis and 'total_cost' in kpis:
            expected_total = kpis['import_cost'] - kpis['export_revenue']
            actual_total = kpis['total_cost']
            if abs(expected_total - actual_total) > 0.01:
                issues.append(f"Cost inconsistency: {expected_total} vs {actual_total}")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            self.logger.warning(f"KPI validation found {len(issues)} issues:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")
        
        return is_valid, issues


# ============================================================================
# Helper Functions
# ============================================================================

def split_import_export(net_consumption: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split net consumption into import and export arrays.
    
    Convention: net > 0 = import, net < 0 = export
    
    Args:
        net_consumption: Array of net consumption values
        
    Returns:
        (import_array, export_array) both with positive values
    """
    import_arr = np.where(net_consumption > 0, net_consumption, 0)
    export_arr = np.where(net_consumption < 0, -net_consumption, 0)
    return import_arr, export_arr


def calculate_pv_consumed(
    pv_generation: np.ndarray,
    net_consumption: np.ndarray
) -> np.ndarray:
    """
    Calculate PV self-consumed (not exported).
    
    Args:
        pv_generation: PV generation array
        net_consumption: Net consumption array
        
    Returns:
        PV consumed array
    """
    # If net_consumption < 0, we're exporting
    # PV consumed = PV generated - amount exported
    pv_consumed = np.zeros_like(pv_generation)
    
    for i in range(len(pv_generation)):
        if net_consumption[i] >= 0:
            # Importing: all PV consumed locally
            pv_consumed[i] = pv_generation[i]
        else:
            # Exporting: PV consumed = PV gen + net_consumption
            # (net_consumption is negative, so this is PV gen - export)
            pv_consumed[i] = max(0, pv_generation[i] + net_consumption[i])
    
    return pv_consumed