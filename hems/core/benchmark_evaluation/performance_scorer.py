# =====================================================
# hems/core/benchmark_evaluation/performance_scorer.py
# =====================================================

"""
Operational Performance Index (OPI) - MAUT-Based Scorer
========================================================

A scientifically grounded multi-attribute performance evaluation system for HEMS agents
based on Multi-Attribute Utility Theory (MAUT) with baseline-oracle normalization.

Theoretical Foundations
-----------------------
This implementation follows established decision analysis frameworks:

1. **Multi-Attribute Utility Theory (MAUT)**:
   - Keeney, R. L., & Raiffa, H. (1976). Decisions with Multiple Objectives: 
     Preferences and Value Trade-Offs. Wiley.
   - Additive value model: S(A) = Î£_c w_c Â· U_c(A)
   
2. **Desirability Functions**:
   - Derringer, G., & Suich, R. (1980). Simultaneous Optimization of Several 
     Response Variables. Journal of Quality Technology, 12(4), 214-219.
   - Maps heterogeneous KPIs to [0,1] utility scale
   
3. **Baseline-Oracle Normalization**:
   - Scenario-intrinsic reference points (no arbitrary thresholds)
   - Utility = fraction of achievable improvement closed
   - Aligned with ideal/anti-ideal concepts in MCDM
   
4. **Integration with MCDM**:
   - Compatible with TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
   - Compatible with DEA (Data Envelopment Analysis)
   - Rank correlation validation (Spearman's Ï, Kendall's Ï„)

Key Features
------------
- Five weighted categories (Energy Economics, Renewable, Battery, AI, Statistical)
- Three utility function types (benefit, cost, target)
- Multi-building aggregation with statistical validation
- Weight sensitivity analysis and robustness testing
- Regression detection (agents worse than baseline)
- Comprehensive reporting with confidence intervals

Mathematical Formulation
------------------------
For agent A, category c, and KPI j:

**Single-Attribute Utility** (benefit-type, maximize):
    u_j(x) = (x - x_baseline) / (x_oracle - x_baseline)
    
**Single-Attribute Utility** (cost-type, minimize):
    u_j(x) = (x_baseline - x) / (x_baseline - x_oracle)
    
**Single-Attribute Utility** (target-type, moderate):
    u_j(x) = max(0, 1 - |x - m_j| / Î”_j)
    
**Category Utility**:
    U_c(A) = Î£_{j K_c} Î±_{c,j} Â· u_j(A)
    where Î£_j Î±_{c,j} = 1 (intra-category weights)
    
**Operational Performance Index (OPI)**:
    S(A) = Î£_c w_c Â· U_c(A)
    where Î£_c w_c = 1 (category weights)

Implementation Notes
--------------------
- Utilities can be negative (agent worse than baseline) - reported separately
- Clipped utilities [0,1] used for aggregation to prevent negative OPI
- Per-building utilities computed then aggregated for multi-building scenarios
- Statistical testing uses paired tests across buildings
- Sensitivity analysis perturbs weights ±50% to test rank stability

"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from itertools import combinations

logger = logging.getLogger(__name__)


class UtilityType(Enum):
    """Types of utility functions for KPI normalization."""
    BENEFIT = "benefit"  # Maximize (higher is better)
    COST = "cost"        # Minimize (lower is better)
    TARGET = "target"    # Moderate (target range is best)


@dataclass
class KPIDefinition:
    """Definition of a KPI with its utility function type and parameters."""
    name: str
    utility_type: UtilityType
    category: str
    target_value: Optional[float] = None      # For target-type utilities
    target_tolerance: Optional[float] = None  # For target-type utilities
    oracle_value: Optional[float] = None      # Domain-knowledge oracle target
    metric_bounds: Optional[Tuple[float, float]] = None  # (min, max) for bounded metrics
    description: str = ""


@dataclass
class UtilityResult:
    """Result of utility calculation for a KPI."""
    kpi_name: str
    raw_value: float
    baseline_value: float
    oracle_value: float
    raw_utility: float      # Can be negative (worse than baseline)
    clipped_utility: float  # Clipped to [0,1] for aggregation
    is_regression: bool     # True if worse than baseline


@dataclass
class CategoryScore:
    """Score for a KPI category."""
    category_name: str
    utilities: List[UtilityResult]
    category_utility: float  # Weighted average of clipped utilities
    weight: float           # Category weight in overall OPI


@dataclass
class OPIResult:
    """Complete OPI result for an agent."""
    agent_name: str
    opi_score: float                    # Overall Performance Index [0,1]
    category_scores: List[CategoryScore]
    total_kpis: int
    regression_count: int               # Number of KPIs worse than baseline
    regression_kpis: List[str]         # Names of regressed KPIs
    weighted_contribution: Dict[str, float]  # Category contributions to OPI


@dataclass
class StatisticalComparison:
    """Statistical comparison between two agents."""
    agent_a: str
    agent_b: str
    opi_diff_mean: float       # Mean OPI difference (A - B)
    opi_diff_std: float        # Std of OPI difference
    p_value: float             # p-value from paired test
    effect_size: float         # Cohen's d
    is_significant: bool       # p < 0.05
    confidence_interval: Tuple[float, float]  # 95% CI


class PerformanceScorer:
    """
    MAUT-based Operational Performance Index (OPI) calculator.
    
    Implements baseline-oracle normalization with multi-attribute utility theory
    for scientifically rigorous HEMS agent evaluation.
    
    Usage
    -----
    >>> scorer = PerformanceScorer()
    >>> results = scorer.calculate_scores(
    ...     agent_kpis={'DQN': {...}, 'SAC': {...}},
    ...     baseline_kpis={...},
    ...     oracle_kpis={...}
    ... )
    >>> print(results['overall_scores']['DQN'].opi_score)
    """
    
    # =========================================================================
    # CATEGORY STRUCTURE AND WEIGHTS
    # =========================================================================
    # Based on normative priorities in residential energy management:
    # 1. Economic performance (30%) - primary homeowner concern
    # 2. Renewable utilization (25%) - sustainability priority
    # 3. Battery health (20%) - asset protection
    # 4. AI/Learning (15%) - algorithm quality
    # 5. Statistical robustness (10%) - research validity
    
    CATEGORY_WEIGHTS = {
        'energy_economics': 0.30,
        'renewable_energy': 0.25,
        'battery_performance': 0.20,
        'ai_learning': 0.15,
        'statistical_research': 0.10,
    }
    
    # =========================================================================
    # KPI DEFINITIONS WITH UTILITY TYPES
    # =========================================================================
    # Each KPI is assigned a utility function type that defines how it should
    # be normalized relative to baseline and oracle.
    
    KPI_DEFINITIONS = {
        # Energy Economics Category
        'total_cost': KPIDefinition(
            name='total_cost',
            utility_type=UtilityType.COST,
            category='energy_economics',
            description='Total electricity cost ($)'
        ),
        'cost_volatility': KPIDefinition(
            name='cost_volatility',
            utility_type=UtilityType.COST,
            category='energy_economics',
            description='Standard deviation of daily costs'
        ),
        'peak_demand_avg': KPIDefinition(
            name='peak_demand_avg',
            utility_type=UtilityType.COST,
            category='energy_economics',
            description='Average peak demand (kW)'
        ),
        'load_factor': KPIDefinition(
            name='load_factor',
            utility_type=UtilityType.BENEFIT,
            category='energy_economics',
            description='Load factor (avg/peak demand)'
        ),
        'demand_response_score': KPIDefinition(
            name='demand_response_score',
            utility_type=UtilityType.BENEFIT,
            category='energy_economics',
            description='Demand response capability score'
        ),
        'grid_interaction_cost': KPIDefinition(
            name='grid_interaction_cost',
            utility_type=UtilityType.COST,
            category='energy_economics',
            description='Cost of grid interactions'
        ),
        
        # Renewable Energy Category
        'pv_self_consumption_rate': KPIDefinition(
            name='pv_self_consumption_rate',
            utility_type=UtilityType.BENEFIT,
            category='renewable_energy',
            description='Fraction of PV generation consumed locally'
        ),
        'pv_self_sufficiency_ratio': KPIDefinition(
            name='pv_self_sufficiency_ratio',
            utility_type=UtilityType.BENEFIT,
            category='renewable_energy',
            description='Fraction of consumption met by PV'
        ),
        'renewable_fraction': KPIDefinition(
            name='renewable_fraction',
            utility_type=UtilityType.BENEFIT,
            category='renewable_energy',
            description='Fraction of consumption from renewables'
        ),
        'pv_curtailment_rate': KPIDefinition(
            name='pv_curtailment_rate',
            utility_type=UtilityType.COST,
            category='renewable_energy',
            description='Fraction of PV generation curtailed'
        ),
        'pv_utilization_efficiency': KPIDefinition(
            name='pv_utilization_efficiency',
            utility_type=UtilityType.BENEFIT,
            category='renewable_energy',
            description='Efficiency of PV utilization'
        ),
        
        # Battery Performance Category
        'battery_cycles': KPIDefinition(
            name='battery_cycles',
            utility_type=UtilityType.TARGET,
            category='battery_performance',
            target_value=500.0,  # cycles/year (adjust based on warranty)
            target_tolerance=100.0,  # ±100 cycles/year acceptable
            description='Annual battery cycles'
        ),
        'cycle_efficiency': KPIDefinition(
            name='cycle_efficiency',
            utility_type=UtilityType.BENEFIT,
            category='battery_performance',
            description='Round-trip efficiency (%)'
        ),
        'battery_capacity_utilization': KPIDefinition(
            name='battery_capacity_utilization',
            utility_type=UtilityType.BENEFIT,
            category='battery_performance',
            description='Fraction of capacity utilized'
        ),
        'battery_health_score': KPIDefinition(
            name='battery_health_score',
            utility_type=UtilityType.BENEFIT,
            category='battery_performance',
            description='Battery health score'
        ),
        'optimal_soc_usage': KPIDefinition(
            name='optimal_soc_usage',
            utility_type=UtilityType.BENEFIT,
            category='battery_performance',
            description='Optimal SOC range utilization'
        ),
        
        # AI/Learning Performance Category
        'avg_reward': KPIDefinition(
            name='avg_reward',
            utility_type=UtilityType.BENEFIT,
            category='ai_learning',
            description='Average episode reward'
        ),
        'learning_improvement': KPIDefinition(
            name='learning_improvement',
            utility_type=UtilityType.BENEFIT,
            category='ai_learning',
            description='Improvement from start to end of training'
        ),
        'sample_efficiency': KPIDefinition(
            name='sample_efficiency',
            utility_type=UtilityType.BENEFIT,
            category='ai_learning',
            description='Sample efficiency (performance per timestep)'
        ),
        'policy_stability': KPIDefinition(
            name='policy_stability',
            utility_type=UtilityType.BENEFIT,
            category='ai_learning',
            description='Stability of learned policy'
        ),
        'training_stability_index': KPIDefinition(
            name='training_stability_index',
            utility_type=UtilityType.BENEFIT,
            category='ai_learning',
            description='Training stability index'
        ),
        
        # Statistical/Research Category
        'performance_consistency': KPIDefinition(
            name='performance_consistency',
            utility_type=UtilityType.BENEFIT,
            category='statistical_research',
            description='Consistency across test buildings'
        ),
        'generalization_gap': KPIDefinition(
            name='generalization_gap',
            utility_type=UtilityType.COST,
            category='statistical_research',
            description='Gap between validation and test performance'
        ),
        'robustness_score': KPIDefinition(
            name='robustness_score',
            utility_type=UtilityType.BENEFIT,
            category='statistical_research',
            description='Robustness to distribution shifts'
        ),
        'stability_index_longterm': KPIDefinition(
            name='stability_index_longterm',
            utility_type=UtilityType.BENEFIT,
            category='statistical_research',
            description='Long-term stability index'
        ),
    }
    
    def __init__(
        self,
        category_weights: Optional[Dict[str, float]] = None,
        logger_instance: Optional[logging.Logger] = None,
        enable_statistical_tests: bool = True,
        enable_sensitivity_analysis: bool = True,
        confidence_level: float = 0.95,
        sensitivity_perturbation: float = 0.30  # ±30% weight variation
    ):
        """
        Initialize the performance scorer.
        
        Args:
            category_weights: Optional custom category weights (must sum to 1.0)
            logger_instance: Logger instance
            enable_statistical_tests: Enable statistical hypothesis testing
            enable_sensitivity_analysis: Enable weight sensitivity analysis
            confidence_level: Confidence level for statistical tests (default: 0.95)
            sensitivity_perturbation: Weight perturbation for sensitivity (default: 0.30)
        """
        self.logger = logger_instance or logger
        
        # Category weights
        if category_weights is not None:
            self._validate_weights(category_weights)
            self.category_weights = category_weights
        else:
            self.category_weights = self.CATEGORY_WEIGHTS.copy()
        
        # Configuration
        self.enable_statistical_tests = enable_statistical_tests
        self.enable_sensitivity_analysis = enable_sensitivity_analysis
        self.confidence_level = confidence_level
        self.sensitivity_perturbation = sensitivity_perturbation
        
        # Build intra-category weights (equal weights within category)
        self.intra_category_weights = self._build_intra_category_weights()
        
        self.logger.info("[OPI] Performance Scorer initialized (MAUT-based)")
        self.logger.info(f"[OPI] Category weights: {self.category_weights}")
        self.logger.info(f"[OPI] Statistical tests: {enable_statistical_tests}")
        self.logger.info(f"[OPI] Sensitivity analysis: {enable_sensitivity_analysis}")
    
    def _validate_weights(self, weights: Dict[str, float]) -> None:
        """Validate that weights sum to 1.0 and are non-negative."""
        total = sum(weights.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Category weights must sum to 1.0, got {total}")
        if any(w < 0 for w in weights.values()):
            raise ValueError("Category weights must be non-negative")
    
    def _build_intra_category_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Build intra-category weights (Î±_{c,j}).
        
        Default: Equal weights within each category.
        Can be overridden for specific applications.
        
        Returns:
            Dict mapping category -> {kpi_name: weight}
        """
        weights = {}
        for category in self.category_weights.keys():
            kpis_in_category = [
                kpi.name for kpi in self.KPI_DEFINITIONS.values()
                if kpi.category == category
            ]
            n_kpis = len(kpis_in_category)
            if n_kpis > 0:
                equal_weight = 1.0 / n_kpis
                weights[category] = {kpi: equal_weight for kpi in kpis_in_category}
            else:
                weights[category] = {}
        
        return weights
    
    # =========================================================================
    # MAIN SCORING METHODS
    # =========================================================================
    
    def calculate_scores(
        self,
        agent_kpis: Dict[str, Dict[str, float]],
        baseline_kpis: Dict[str, float],
        oracle_kpis: Dict[str, float],
        per_building_data: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None
    ) -> Dict[str, Any]:
        """
        Calculate Operational Performance Index (OPI) for all agents.
        
        This is the main entry point for OPI calculation.
        
        Args:
            agent_kpis: Dict mapping agent_name -> aggregated_kpis
                       (aggregated across all test buildings)
            baseline_kpis: Baseline KPIs (NoControl or simple RBC)
            oracle_kpis: Oracle KPIs (theoretical optimum or best MPC)
            per_building_data: Optional dict mapping agent_name -> building_name -> kpis
                              (for multi-building statistical analysis)
        
        Returns:
            Dict containing:
                - 'opi_results': Dict[agent_name, OPIResult]
                - 'rankings': List of (agent_name, opi_score, rank)
                - 'statistical_comparisons': List[StatisticalComparison] (if enabled)
                - 'sensitivity_analysis': Dict (if enabled)
                - 'correlation_analysis': Dict (KPI correlations)
        """
        if not agent_kpis:
            self.logger.warning("[OPI] No agent data provided")
            return {}
        
        self.logger.info(f"[OPI] Calculating scores for {len(agent_kpis)} agents")
        self.logger.info(f"[OPI] Baseline KPIs: {len(baseline_kpis)} metrics")
        self.logger.info(f"[OPI] Oracle KPIs: {len(oracle_kpis)} metrics")
        
        # Validate inputs
        self._validate_reference_kpis(baseline_kpis, oracle_kpis)
        
        # Calculate OPI for each agent
        opi_results = {}
        for agent_name, agent_kpi_dict in agent_kpis.items():
            try:
                opi_result = self._calculate_agent_opi(
                    agent_name=agent_name,
                    agent_kpis=agent_kpi_dict,
                    baseline_kpis=baseline_kpis,
                    oracle_kpis=oracle_kpis
                )
                opi_results[agent_name] = opi_result
                
                self.logger.info(
                    f"[OPI] {agent_name}: OPI={opi_result.opi_score:.4f}, "
                    f"Regressions={opi_result.regression_count}/{opi_result.total_kpis}"
                )
                
            except Exception as e:
                self.logger.error(f"[OPI] Failed to calculate OPI for {agent_name}: {e}")
                continue
        
        if not opi_results:
            self.logger.error("[OPI] No valid OPI results computed")
            return {}
        
        # Rank agents
        rankings = self._rank_agents(opi_results)
        
        # Statistical analysis
        statistical_comparisons = []
        if self.enable_statistical_tests and per_building_data:
            statistical_comparisons = self._perform_statistical_tests(
                per_building_data,
                baseline_kpis,
                oracle_kpis
            )
        
        # Sensitivity analysis
        sensitivity_results = {}
        if self.enable_sensitivity_analysis:
            sensitivity_results = self._perform_sensitivity_analysis(
                agent_kpis,
                baseline_kpis,
                oracle_kpis
            )
        
        # KPI correlation analysis
        correlation_analysis = self._analyze_kpi_correlations(agent_kpis)
        
        results = {
            'opi_results': opi_results,
            'rankings': rankings,
            'statistical_comparisons': statistical_comparisons,
            'sensitivity_analysis': sensitivity_results,
            'correlation_analysis': correlation_analysis,
            'metadata': {
                'n_agents': len(opi_results),
                'category_weights': self.category_weights,
                'confidence_level': self.confidence_level,
                'timestamp': np.datetime64('now').astype(str)
            }
        }
        
        return results
    
    def _validate_reference_kpis(
        self,
        baseline_kpis: Dict[str, float],
        oracle_kpis: Dict[str, float]
    ) -> None:
        """Validate baseline and oracle KPIs."""
        if not baseline_kpis:
            raise ValueError("Baseline KPIs cannot be empty")
        if not oracle_kpis:
            raise ValueError("Oracle KPIs cannot be empty")
        
        # Check for common KPIs
        baseline_keys = set(baseline_kpis.keys())
        oracle_keys = set(oracle_kpis.keys())
        common_keys = baseline_keys & oracle_keys
        
        if len(common_keys) == 0:
            raise ValueError("Baseline and oracle must have at least one common KPI")
        
        self.logger.debug(f"[OPI] Common KPIs: {len(common_keys)}")
    
    def _calculate_agent_opi(
        self,
        agent_name: str,
        agent_kpis: Dict[str, float],
        baseline_kpis: Dict[str, float],
        oracle_kpis: Dict[str, float]
    ) -> OPIResult:
        """
        Calculate OPI for a single agent.
        
        Implements the full MAUT calculation:
        1. Compute single-attribute utilities u_j for each KPI
        2. Aggregate within categories: U_c = Î£_j Î±_j Â· u_j
        3. Aggregate across categories: OPI = Î£_c w_c Â· U_c
        
        Args:
            agent_name: Name of the agent
            agent_kpis: Agent's KPIs
            baseline_kpis: Baseline reference KPIs
            oracle_kpis: Oracle reference KPIs
        
        Returns:
            OPIResult object with complete scoring breakdown
        """
        # Step 1: Calculate single-attribute utilities for all KPIs
        all_utilities = {}
        for kpi_name, kpi_def in self.KPI_DEFINITIONS.items():
            # Check if all required values are available
            if (kpi_name in agent_kpis and 
                kpi_name in baseline_kpis and 
                kpi_name in oracle_kpis):
                
                utility_result = self._compute_utility(
                    kpi_name=kpi_name,
                    kpi_def=kpi_def,
                    agent_value=agent_kpis[kpi_name],
                    baseline_value=baseline_kpis[kpi_name],
                    oracle_value=oracle_kpis[kpi_name]
                )
                all_utilities[kpi_name] = utility_result
        
        if not all_utilities:
            raise ValueError(f"No valid utilities computed for {agent_name}")
        
        # Step 2: Aggregate within categories
        category_scores = []
        for category_name, category_weight in self.category_weights.items():
            # Get utilities for this category
            category_utilities = [
                util for kpi_name, util in all_utilities.items()
                if self.KPI_DEFINITIONS[kpi_name].category == category_name
            ]
            
            if not category_utilities:
                continue
            
            # Compute weighted average of clipped utilities within category
            intra_weights = self.intra_category_weights[category_name]
            category_utility = 0.0
            total_weight = 0.0
            
            for util in category_utilities:
                weight = intra_weights.get(util.kpi_name, 0.0)
                category_utility += weight * util.clipped_utility
                total_weight += weight
            
            if total_weight > 0:
                category_utility /= total_weight
            
            category_score = CategoryScore(
                category_name=category_name,
                utilities=category_utilities,
                category_utility=category_utility,
                weight=category_weight
            )
            category_scores.append(category_score)
        
        # Step 3: Compute overall OPI
        opi_score = sum(
            cat_score.category_utility * cat_score.weight
            for cat_score in category_scores
        )
        
        # Collect regression information
        regression_kpis = [
            util.kpi_name for util in all_utilities.values()
            if util.is_regression
        ]
        
        # Compute weighted contributions
        weighted_contribution = {
            cat.category_name: cat.category_utility * cat.weight
            for cat in category_scores
        }
        
        return OPIResult(
            agent_name=agent_name,
            opi_score=opi_score,
            category_scores=category_scores,
            total_kpis=len(all_utilities),
            regression_count=len(regression_kpis),
            regression_kpis=regression_kpis,
            weighted_contribution=weighted_contribution
        )
    
    # =========================================================================
    # UTILITY COMPUTATION (BASELINE-ORACLE NORMALIZATION)
    # =========================================================================
    
    def _compute_utility(
        self,
        kpi_name: str,
        kpi_def: KPIDefinition,
        agent_value: float,
        baseline_value: float,
        oracle_value: float
    ) -> UtilityResult:
        """
        Compute single-attribute utility using baseline-oracle normalization.
        
        Mathematical formulations:
        
        **Benefit-Type (maximize)**:
            u_j = (x - x_baseline) / (x_oracle - x_baseline)
            u_j = 0 if x x_baseline (no improvement)
            u_j = 1 if x x_oracle (oracle-level performance)
        
        **Cost-Type (minimize)**:
            u_j = (x_baseline - x) / (x_baseline - x_oracle)
            u_j = 0 if x x_baseline (no improvement)
            u_j = 1 if x x_oracle (oracle-level performance)
        
        **Target-Type (moderate)**:
            u_j = max(0, 1 - |x - m_j| / Î”_j)
            where m_j = target value, Î”_j = tolerance
        
        Args:
            kpi_name: Name of KPI
            kpi_def: KPI definition with utility type
            agent_value: Agent's KPI value
            baseline_value: Baseline reference value
            oracle_value: Oracle reference value
        
        Returns:
            UtilityResult with raw and clipped utilities
        """
        # Validate inputs
        if not np.isfinite(agent_value):
            self.logger.warning(f"[OPI] {kpi_name}: Invalid agent value {agent_value}")
            return self._create_invalid_utility(kpi_name, agent_value, baseline_value, oracle_value)
        
        if not np.isfinite(baseline_value) or not np.isfinite(oracle_value):
            self.logger.warning(
                f"[OPI] {kpi_name}: Invalid reference values "
                f"(baseline={baseline_value}, oracle={oracle_value})"
            )
            return self._create_invalid_utility(kpi_name, agent_value, baseline_value, oracle_value)
        
        # Compute utility based on type
        if kpi_def.utility_type == UtilityType.BENEFIT:
            raw_utility = self._compute_benefit_utility(
                agent_value, baseline_value, oracle_value
            )
        elif kpi_def.utility_type == UtilityType.COST:
            raw_utility = self._compute_cost_utility(
                agent_value, baseline_value, oracle_value
            )
        elif kpi_def.utility_type == UtilityType.TARGET:
            raw_utility = self._compute_target_utility(
                agent_value,
                target=kpi_def.target_value,
                tolerance=kpi_def.target_tolerance
            )
        else:
            raise ValueError(f"Unknown utility type: {kpi_def.utility_type}")
        
        # Clip for aggregation
        clipped_utility = np.clip(raw_utility, 0.0, 1.0)
        
        # Check for regression
        is_regression = raw_utility < 0.0
        
        return UtilityResult(
            kpi_name=kpi_name,
            raw_value=agent_value,
            baseline_value=baseline_value,
            oracle_value=oracle_value,
            raw_utility=raw_utility,
            clipped_utility=clipped_utility,
            is_regression=is_regression
        )
    
    def _compute_benefit_utility(
        self,
        x: float,
        x_baseline: float,
        x_oracle: float
    ) -> float:
        """
        Compute utility for benefit-type KPI (maximize).
        
        u = (x - x_baseline) / (x_oracle - x_baseline)
        
        With 50% improvement target:
        - u = 0: Agent at baseline
        - u = 1: Agent achieved 50% improvement  
        - u < 0: Agent worse than baseline
        - u > 1: Agent exceeded 50% improvement (excellent!)
        """
        denominator = x_oracle - x_baseline
        
        if abs(denominator) < 1e-9:
            # Oracle = baseline (shouldn't happen with 50% target)
            return 0.5
        
        utility = (x - x_baseline) / denominator
        return utility
    
    def _compute_cost_utility(
        self,
        x: float,
        x_baseline: float,
        x_oracle: float
    ) -> float:
        """
        Compute utility for cost-type KPI (minimize).
        
        u = (x_baseline - x) / (x_baseline - x_oracle)
        
        With 50% reduction target:
        - u = 0: Agent at baseline
        - u = 1: Agent achieved 50% cost reduction
        - u < 0: Agent worse than baseline  
        - u > 1: Agent exceeded 50% reduction (excellent!)
        """
        denominator = x_baseline - x_oracle
        
        if abs(denominator) < 1e-9:
            # Oracle = baseline (shouldn't happen with 50% target)
            return 0.5
        
        utility = (x_baseline - x) / denominator
        return utility
    
    def _compute_target_utility(
        self,
        x: float,
        target: float,
        tolerance: float
    ) -> float:
        """
        Compute utility for target-type KPI (moderate).
        
        u = max(0, 1 - |x - target| / tolerance)
        
        Interpretation:
        - u = 1: Agent hits target value
        - u = 0: Agent deviates by tolerance or more
        - Linear decay from target
        
        Used for metrics like battery cycles where moderate usage is optimal.
        """
        if target is None or tolerance is None:
            self.logger.warning("[OPI] Target utility requires target and tolerance")
            return 0.5
        
        if tolerance <= 0:
            self.logger.warning(f"[OPI] Invalid tolerance {tolerance}, using default")
            tolerance = abs(target * 0.2)  # 20% of target
        
        deviation = abs(x - target)
        utility = max(0.0, 1.0 - deviation / tolerance)
        
        return utility
    
    def _create_invalid_utility(
        self,
        kpi_name: str,
        agent_value: float,
        baseline_value: float,
        oracle_value: float
    ) -> UtilityResult:
        """Create a neutral utility result for invalid inputs."""
        return UtilityResult(
            kpi_name=kpi_name,
            raw_value=agent_value,
            baseline_value=baseline_value,
            oracle_value=oracle_value,
            raw_utility=0.0,
            clipped_utility=0.0,
            is_regression=False
        )
    
    # =========================================================================
    # RANKING
    # =========================================================================
    
    def _rank_agents(
        self,
        opi_results: Dict[str, OPIResult]
    ) -> List[Tuple[str, float, int]]:
        """
        Rank agents by OPI score (descending).
        
        Returns:
            List of (agent_name, opi_score, rank) tuples
        """
        sorted_agents = sorted(
            [(name, result.opi_score) for name, result in opi_results.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        rankings = []
        for rank, (agent_name, opi_score) in enumerate(sorted_agents, 1):
            rankings.append((agent_name, opi_score, rank))
        
        return rankings
    
    # =========================================================================
    # STATISTICAL TESTING
    # =========================================================================
    
    def _perform_statistical_tests(
        self,
        per_building_data: Dict[str, Dict[str, Dict[str, float]]],
        baseline_kpis: Dict[str, float],
        oracle_kpis: Dict[str, float]
    ) -> List[StatisticalComparison]:
        """
        Perform pairwise statistical tests between agents.
        
        Uses paired t-test (or Wilcoxon signed-rank if normality violated)
        across test buildings to determine if OPI differences are significant.
        
        Args:
            per_building_data: Dict[agent_name][building_name][kpi_name] = value
            baseline_kpis: Baseline KPIs (aggregated or per-building)
            oracle_kpis: Oracle KPIs (aggregated or per-building)
        
        Returns:
            List of StatisticalComparison objects for all agent pairs
        """
        self.logger.info("[OPI] Performing statistical hypothesis tests")
        
        # Calculate OPI for each agent on each building
        agent_building_opis = {}
        for agent_name, building_data in per_building_data.items():
            agent_building_opis[agent_name] = {}
            for building_name, building_kpis in building_data.items():
                try:
                    opi_result = self._calculate_agent_opi(
                        agent_name=f"{agent_name}_{building_name}",
                        agent_kpis=building_kpis,
                        baseline_kpis=baseline_kpis,
                        oracle_kpis=oracle_kpis
                    )
                    agent_building_opis[agent_name][building_name] = opi_result.opi_score
                except Exception as e:
                    self.logger.warning(
                        f"[OPI] Failed to compute OPI for {agent_name} on {building_name}: {e}"
                    )
        
        # Perform pairwise comparisons
        comparisons = []
        agent_names = list(agent_building_opis.keys())
        
        for agent_a, agent_b in combinations(agent_names, 2):
            # Get OPI scores across buildings for both agents
            buildings_a = set(agent_building_opis[agent_a].keys())
            buildings_b = set(agent_building_opis[agent_b].keys())
            common_buildings = buildings_a & buildings_b
            
            if len(common_buildings) < 2:
                self.logger.warning(
                    f"[OPI] Insufficient common buildings for {agent_a} vs {agent_b}"
                )
                continue
            
            opis_a = [agent_building_opis[agent_a][b] for b in common_buildings]
            opis_b = [agent_building_opis[agent_b][b] for b in common_buildings]
            
            # Perform paired test
            comparison = self._paired_statistical_test(
                agent_a=agent_a,
                agent_b=agent_b,
                opis_a=opis_a,
                opis_b=opis_b
            )
            comparisons.append(comparison)
        
        self.logger.info(f"[OPI] Completed {len(comparisons)} pairwise comparisons")
        
        # Apply Holm-Bonferroni correction for multiple comparisons
        if len(comparisons) > 1:
            comparisons = self._apply_bonferroni_correction(comparisons)
        
        return comparisons
    
    def _paired_statistical_test(
        self,
        agent_a: str,
        agent_b: str,
        opis_a: List[float],
        opis_b: List[float]
    ) -> StatisticalComparison:
        """
        Perform paired statistical test between two agents.
        
        Uses paired t-test with fallback to Wilcoxon signed-rank test
        if normality assumption is violated.
        
        Args:
            agent_a: Name of first agent
            agent_b: Name of second agent
            opis_a: OPI scores for agent A across buildings
            opis_b: OPI scores for agent B across buildings
        
        Returns:
            StatisticalComparison object
        """
        opis_a = np.array(opis_a)
        opis_b = np.array(opis_b)
        differences = opis_a - opis_b
        
        # Summary statistics
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        n = len(differences)
        
        # Effect size (Cohen's d)
        if std_diff > 0:
            cohens_d = mean_diff / std_diff
        else:
            cohens_d = 0.0
        
        # Perform paired t-test
        try:
            t_stat, p_value = stats.ttest_rel(opis_a, opis_b)
            
            # Check normality of differences (Shapiro-Wilk test)
            if n >= 3:
                _, p_normal = stats.shapiro(differences)
                if p_normal < 0.05:
                    # Use Wilcoxon signed-rank test instead
                    self.logger.debug(
                        f"[OPI] Non-normal differences for {agent_a} vs {agent_b}, "
                        "using Wilcoxon test"
                    )
                    _, p_value = stats.wilcoxon(differences)
        except Exception as e:
            self.logger.warning(f"[OPI] Statistical test failed: {e}")
            p_value = 1.0  # Conservative: assume no difference
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        se = std_diff / np.sqrt(n)
        ci_margin = stats.t.ppf(1 - alpha/2, df=n-1) * se
        ci_lower = mean_diff - ci_margin
        ci_upper = mean_diff + ci_margin
        
        # Significance
        is_significant = p_value < 0.05
        
        return StatisticalComparison(
            agent_a=agent_a,
            agent_b=agent_b,
            opi_diff_mean=mean_diff,
            opi_diff_std=std_diff,
            p_value=p_value,
            effect_size=cohens_d,
            is_significant=is_significant,
            confidence_interval=(ci_lower, ci_upper)
        )
    
    def _apply_bonferroni_correction(
        self,
        comparisons: List[StatisticalComparison]
    ) -> List[StatisticalComparison]:
        """
        Apply Holm-Bonferroni correction for multiple comparisons.
        
        Adjusts significance thresholds to control family-wise error rate.
        
        Args:
            comparisons: List of statistical comparisons
        
        Returns:
            Updated list with corrected significance flags
        """
        n_comparisons = len(comparisons)
        alpha = 0.05
        
        # Sort by p-value
        sorted_comparisons = sorted(comparisons, key=lambda x: x.p_value)
        
        # Apply Holm-Bonferroni correction
        for i, comp in enumerate(sorted_comparisons):
            adjusted_alpha = alpha / (n_comparisons - i)
            comp.is_significant = comp.p_value < adjusted_alpha
        
        self.logger.info(
            f"[OPI] Applied Holm-Bonferroni correction for {n_comparisons} comparisons"
        )
        
        return comparisons
    
    # =========================================================================
    # SENSITIVITY ANALYSIS
    # =========================================================================
    
    def _perform_sensitivity_analysis(
        self,
        agent_kpis: Dict[str, Dict[str, float]],
        baseline_kpis: Dict[str, float],
        oracle_kpis: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Perform weight sensitivity analysis.
        
        Perturbs category weights by ±perturbation% and computes:
        - Rank stability (Spearman's Ï)
        - OPI score volatility
        - Robust ranking (agents that maintain rank)
        
        Args:
            agent_kpis: Agent KPIs
            baseline_kpis: Baseline KPIs
            oracle_kpis: Oracle KPIs
        
        Returns:
            Dict with sensitivity analysis results
        """
        self.logger.info("[OPI] Performing weight sensitivity analysis")
        
        # Baseline ranking
        baseline_results = self.calculate_scores(
            agent_kpis=agent_kpis,
            baseline_kpis=baseline_kpis,
            oracle_kpis=oracle_kpis
        )
        baseline_ranking = baseline_results['rankings']
        baseline_ranks = {name: rank for name, _, rank in baseline_ranking}
        
        # Generate perturbed weight vectors
        perturbed_rankings = []
        n_perturbations = 20  # Test with 20 random perturbations
        
        for i in range(n_perturbations):
            perturbed_weights = self._generate_perturbed_weights()
            
            # Temporarily update weights
            original_weights = self.category_weights.copy()
            self.category_weights = perturbed_weights
            
            # Recalculate OPI
            perturbed_results = self.calculate_scores(
                agent_kpis=agent_kpis,
                baseline_kpis=baseline_kpis,
                oracle_kpis=oracle_kpis
            )
            perturbed_ranking = perturbed_results['rankings']
            perturbed_ranks = {name: rank for name, _, rank in perturbed_ranking}
            
            # Compute rank correlation
            common_agents = set(baseline_ranks.keys()) & set(perturbed_ranks.keys())
            if len(common_agents) >= 2:
                baseline_rank_vector = [baseline_ranks[a] for a in common_agents]
                perturbed_rank_vector = [perturbed_ranks[a] for a in common_agents]
                
                spearman_rho, _ = stats.spearmanr(baseline_rank_vector, perturbed_rank_vector)
                kendall_tau, _ = stats.kendalltau(baseline_rank_vector, perturbed_rank_vector)
            else:
                spearman_rho = 1.0
                kendall_tau = 1.0
            
            perturbed_rankings.append({
                'perturbation_id': i,
                'weights': perturbed_weights,
                'ranking': perturbed_ranking,
                'spearman_rho': spearman_rho,
                'kendall_tau': kendall_tau
            })
            
            # Restore original weights
            self.category_weights = original_weights
        
        # Compute sensitivity metrics
        spearman_scores = [pr['spearman_rho'] for pr in perturbed_rankings]
        kendall_scores = [pr['kendall_tau'] for pr in perturbed_rankings]
        
        sensitivity_results = {
            'n_perturbations': n_perturbations,
            'perturbation_magnitude': self.sensitivity_perturbation,
            'baseline_ranking': baseline_ranking,
            'perturbed_rankings': perturbed_rankings,
            'rank_correlation_stats': {
                'spearman_mean': np.mean(spearman_scores),
                'spearman_std': np.std(spearman_scores),
                'spearman_min': np.min(spearman_scores),
                'kendall_mean': np.mean(kendall_scores),
                'kendall_std': np.std(kendall_scores),
                'kendall_min': np.min(kendall_scores)
            },
            'interpretation': self._interpret_sensitivity(np.mean(spearman_scores))
        }
        
        self.logger.info(
            f"[OPI] Sensitivity: Spearman  = {np.mean(spearman_scores):.3f} ± "
            f"{np.std(spearman_scores):.3f}"
        )
        
        return sensitivity_results
    
    def _generate_perturbed_weights(self) -> Dict[str, float]:
        """
        Generate perturbed category weights.
        
        Perturbs each weight by ±perturbation% and renormalizes.
        
        Returns:
            Dict of perturbed weights (sum to 1.0)
        """
        perturbed = {}
        for category, weight in self.category_weights.items():
            # Random perturbation in [-perturbation, +perturbation]
            perturbation = np.random.uniform(
                -self.sensitivity_perturbation,
                self.sensitivity_perturbation
            )
            perturbed[category] = max(0.0, weight * (1 + perturbation))
        
        # Renormalize
        total = sum(perturbed.values())
        if total > 0:
            perturbed = {k: v / total for k, v in perturbed.items()}
        else:
            # Fallback to equal weights
            perturbed = {k: 1.0 / len(perturbed) for k in perturbed.keys()}
        
        return perturbed
    
    def _interpret_sensitivity(self, mean_spearman: float) -> str:
        """Interpret sensitivity analysis results."""
        if mean_spearman >= 0.9:
            return "Excellent: Rankings are highly stable under weight variations"
        elif mean_spearman >= 0.7:
            return "Good: Rankings show moderate stability"
        elif mean_spearman >= 0.5:
            return "Fair: Rankings show some sensitivity to weights"
        else:
            return "Poor: Rankings are highly sensitive to weight choices"
    
    # =========================================================================
    # KPI CORRELATION ANALYSIS
    # =========================================================================
    
    def _analyze_kpi_correlations(
        self,
        agent_kpis: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Analyze correlations between KPIs across agents.
        
        Identifies highly correlated KPIs that may be redundant.
        
        Args:
            agent_kpis: Agent KPIs
        
        Returns:
            Dict with correlation matrix and warnings
        """
        if len(agent_kpis) < 3:
            return {'note': 'Insufficient agents for correlation analysis (need 3)'}
        
        # Build KPI matrix (agents x KPIs)
        kpi_names = list(set(k for kpis in agent_kpis.values() for k in kpis.keys()))
        kpi_matrix = []
        
        for agent_name in agent_kpis.keys():
            agent_vector = [agent_kpis[agent_name].get(kpi, np.nan) for kpi in kpi_names]
            kpi_matrix.append(agent_vector)
        
        kpi_matrix = np.array(kpi_matrix)
        
        # Remove KPIs with too many NaNs
        valid_kpi_mask = np.sum(~np.isnan(kpi_matrix), axis=0) >= 3
        valid_kpi_names = [kpi for kpi, valid in zip(kpi_names, valid_kpi_mask) if valid]
        kpi_matrix = kpi_matrix[:, valid_kpi_mask]
        
        if kpi_matrix.shape[1] < 2:
            return {'note': 'Insufficient valid KPIs for correlation analysis'}
        
        # Compute correlation matrix
        correlation_matrix = np.corrcoef(kpi_matrix.T)
        
        # Find highly correlated pairs (|Ï| > 0.8)
        high_correlations = []
        n_kpis = len(valid_kpi_names)
        for i in range(n_kpis):
            for j in range(i + 1, n_kpis):
                if abs(correlation_matrix[i, j]) > 0.8:
                    high_correlations.append({
                        'kpi_1': valid_kpi_names[i],
                        'kpi_2': valid_kpi_names[j],
                        'correlation': correlation_matrix[i, j]
                    })
        
        results = {
            'kpi_names': valid_kpi_names,
            'correlation_matrix': correlation_matrix.tolist(),
            'high_correlations': high_correlations,
            'n_high_correlations': len(high_correlations)
        }
        
        if high_correlations:
            self.logger.warning(
                f"[OPI] Found {len(high_correlations)} highly correlated KPI pairs (|Ï|>0.8)"
            )
            for hc in high_correlations[:5]:  # Show first 5
                self.logger.warning(
                    f"[OPI]   {hc['kpi_1']} <-> {hc['kpi_2']}: Ï={hc['correlation']:.3f}"
                )
        
        return results
    
    # =========================================================================
    # REPORTING AND FORMATTING
    # =========================================================================
    
    def format_results(self, results: Dict[str, Any]) -> str:
        """
        Format OPI results as human-readable text.
        
        Args:
            results: Output from calculate_scores()
        
        Returns:
            Formatted string report
        """
        if not results or 'opi_results' not in results:
            return "No OPI results available."
        
        lines = []
        lines.append("=" * 100)
        lines.append("OPERATIONAL PERFORMANCE INDEX (OPI) - MAUT-Based Evaluation")
        lines.append("=" * 100)
        lines.append("")
        
        # Metadata
        metadata = results.get('metadata', {})
        lines.append("Configuration:")
        lines.append(f"  Agents evaluated: {metadata.get('n_agents', 0)}")
        lines.append(f"  Confidence level: {metadata.get('confidence_level', 0.95):.2%}")
        lines.append(f"  Category weights: {metadata.get('category_weights', {})}")
        lines.append("")
        
        # Rankings
        lines.append("Overall Rankings (by OPI):")
        lines.append("-" * 100)
        rankings = results.get('rankings', [])
        for agent_name, opi_score, rank in rankings:
            opi_result = results['opi_results'][agent_name]
            regression_info = f"({opi_result.regression_count} regressions)" if opi_result.regression_count > 0 else ""
            lines.append(
                f"  {rank}. {agent_name:30s} | OPI: {opi_score:.4f} | "
                f"KPIs: {opi_result.total_kpis} {regression_info}"
            )
        lines.append("")
        
        # Detailed breakdown for each agent
        for agent_name, opi_score, rank in rankings:
            opi_result = results['opi_results'][agent_name]
            lines.append(f"Agent: {agent_name} (Rank #{rank})")
            lines.append("-" * 100)
            lines.append(f"  Overall OPI: {opi_result.opi_score:.4f}")
            lines.append("")
            
            lines.append("  Category Breakdown:")
            for cat_score in opi_result.category_scores:
                weighted_contrib = opi_result.weighted_contribution[cat_score.category_name]
                lines.append(
                    f"    {cat_score.category_name:30s} | Utility: {cat_score.category_utility:.4f} | "
                    f"Weight: {cat_score.weight:.2f} | Contribution: {weighted_contrib:.4f}"
                )
            lines.append("")
            
            # Show regressions if any
            if opi_result.regression_count > 0:
                lines.append(f"  Regressions ({opi_result.regression_count} KPIs worse than baseline):")
                for regressed_kpi in opi_result.regression_kpis[:5]:  # Show first 5
                    lines.append(f"    - {regressed_kpi}")
                if len(opi_result.regression_kpis) > 5:
                    lines.append(f"    ... and {len(opi_result.regression_kpis) - 5} more")
                lines.append("")
        
        # Statistical comparisons
        if results.get('statistical_comparisons'):
            lines.append("Statistical Comparisons (Paired Tests):")
            lines.append("-" * 100)
            for comp in results['statistical_comparisons'][:10]:  # Show first 10
                sig_marker = "***" if comp.is_significant else ""
                lines.append(
                    f"  {comp.agent_a} vs {comp.agent_b}: "
                    f"Î”={comp.opi_diff_mean:+.4f} (p={comp.p_value:.4f}) "
                    f"d={comp.effect_size:.2f} {sig_marker}"
                )
            lines.append("")
        
        # Sensitivity analysis
        if results.get('sensitivity_analysis'):
            sens = results['sensitivity_analysis']
            if 'rank_correlation_stats' in sens:
                stats_dict = sens['rank_correlation_stats']
                lines.append("Weight Sensitivity Analysis:")
                lines.append("-" * 100)
                lines.append(
                    f"  Rank stability (Spearman Ï): {stats_dict['spearman_mean']:.3f} ± "
                    f"{stats_dict['spearman_std']:.3f}"
                )
                lines.append(
                    f"  Rank stability (Kendall Ï„): {stats_dict['kendall_mean']:.3f} ± "
                    f"{stats_dict['kendall_std']:.3f}"
                )
                lines.append(f"  Interpretation: {sens.get('interpretation', 'N/A')}")
                lines.append("")
        
        # Correlation warnings
        if results.get('correlation_analysis'):
            corr = results['correlation_analysis']
            if corr.get('n_high_correlations', 0) > 0:
                lines.append(" KPI Correlation Warnings:")
                lines.append("-" * 100)
                for hc in corr['high_correlations'][:5]:
                    lines.append(
                        f"  {hc['kpi_1']} <-> {hc['kpi_2']}: Ï={hc['correlation']:.3f}"
                    )
                lines.append("")
        
        lines.append("=" * 100)
        lines.append("End of OPI Report")
        lines.append("=" * 100)
        
        return "\n".join(lines)
    
    def export_to_dict(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export OPI results to a JSON-serializable dictionary.
        
        Args:
            results: Output from calculate_scores()
        
        Returns:
            JSON-serializable dict
        """
        if not results or 'opi_results' not in results:
            return {}
        
        export = {
            'metadata': results.get('metadata', {}),
            'rankings': [
                {
                    'rank': rank,
                    'agent_name': agent_name,
                    'opi_score': opi_score
                }
                for agent_name, opi_score, rank in results['rankings']
            ],
            'agents': {}
        }
        
        # Export detailed agent results
        for agent_name, opi_result in results['opi_results'].items():
            export['agents'][agent_name] = {
                'opi_score': opi_result.opi_score,
                'total_kpis': opi_result.total_kpis,
                'regression_count': opi_result.regression_count,
                'regression_kpis': opi_result.regression_kpis,
                'category_scores': {
                    cat.category_name: {
                        'utility': cat.category_utility,
                        'weight': cat.weight,
                        'contribution': opi_result.weighted_contribution[cat.category_name]
                    }
                    for cat in opi_result.category_scores
                }
            }
        
        # Export statistical comparisons
        if results.get('statistical_comparisons'):
            export['statistical_comparisons'] = [
                {
                    'agent_a': comp.agent_a,
                    'agent_b': comp.agent_b,
                    'opi_diff_mean': comp.opi_diff_mean,
                    'p_value': comp.p_value,
                    'effect_size': comp.effect_size,
                    'is_significant': comp.is_significant
                }
                for comp in results['statistical_comparisons']
            ]
        
        # Export sensitivity analysis
        if results.get('sensitivity_analysis'):
            sens = results['sensitivity_analysis']
            if 'rank_correlation_stats' in sens:
                export['sensitivity_analysis'] = sens['rank_correlation_stats']
        
        return export


# =============================================================================
# BACKWARD COMPATIBILITY WRAPPER
# =============================================================================

def create_legacy_compatible_scorer(logger_instance=None):
    """
    Create a PerformanceScorer instance that can work with legacy code.
    
    This function provides a compatibility layer for existing code that may
    expect the old min-max normalization interface.
    
    Returns:
        PerformanceScorer instance
    """
    return PerformanceScorer(logger_instance=logger_instance)


if __name__ == "__main__":
    # Example usage
    print("MAUT-based Operational Performance Index (OPI)")
    print("=" * 60)