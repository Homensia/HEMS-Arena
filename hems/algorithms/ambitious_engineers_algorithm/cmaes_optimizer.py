# ==================================================================
# hems/agents/ambitious_engineers_algorithm.py - cmaes_optimizer.py
# ==================================================================

"""
CMA-ES Optimizer for Policy Network Training
=============================================
Wrapper for training Phase 1 and Phase 2/3 policies using CMA-ES.

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a
black-box optimization algorithm that:
- Doesn't require gradients (derivative-free)
- Handles non-convex, noisy objectives
- Adapts the search distribution based on successful solutions
- Highly parallelizable (evaluates population in parallel)

Training Configuration (from their paper):
- Phase 1: 3000 iterations, population=50, sigma=0.05, 5 seeds
- Phase 2/3: 10000 iterations, population=50, sigma=0.005, 1 seed

Reference: Hansen & Ostermeier (2001), CMA-ES algorithm
Team ambitiousengineers used the pycma library
"""

import numpy as np
import cma
import multiprocessing
from typing import Dict, List, Callable, Optional, Tuple
from functools import partial
from tqdm import tqdm
import time
import pickle
from pathlib import Path


class CMAESOptimizer:
    """
    CMA-ES optimizer for policy network training.
    
    Handles:
    - Parallel evaluation of candidate solutions
    - Checkpointing and resume capability
    - Multiple random seeds for robustness
    - L2 regularization penalty
    """
    
    def __init__(
        self,
        objective_function: Callable,
        n_params: int,
        sigma0: float = 0.05,
        population_size: int = 50,
        max_iterations: int = 3000,
        n_jobs: int = 1,
        l2_penalty: float = 0.0,
        seed: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize CMA-ES optimizer.
        
        Args:
            objective_function: Function to minimize, signature: f(params) -> cost
            n_params: Number of parameters to optimize
            sigma0: Initial step size (standard deviation)
            population_size: Number of candidate solutions per iteration
            max_iterations: Maximum number of CMA-ES iterations
            n_jobs: Number of parallel workers
            l2_penalty: L2 regularization strength
            seed: Random seed for reproducibility
            checkpoint_dir: Directory for saving checkpoints
            verbose: Print progress information
        """
        self.objective_function = objective_function
        self.n_params = n_params
        self.sigma0 = sigma0
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.n_jobs = min(n_jobs, multiprocessing.cpu_count())
        self.l2_penalty = l2_penalty
        self.seed = seed
        self.verbose = verbose
        
        # Checkpoint management
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None
        
        # Training history
        self.history = {
            'best_params': [],
            'best_costs': [],
            'mean_costs': [],
            'iterations': []
        }
        
        # Best solution tracking
        self.best_params = None
        self.best_cost = np.inf
    
    def _objective_with_penalty(self, params: np.ndarray) -> float:
        """
        Evaluate objective with L2 regularization.
        
        Args:
            params: Parameter vector
        
        Returns:
            Total cost (objective + penalty)
        """
        # Base objective
        cost = self.objective_function(params)
        
        # Add L2 penalty
        if self.l2_penalty > 0:
            penalty = self.l2_penalty * np.sum(params ** 2)
            cost += penalty
        
        return cost
    
    def _evaluate_population(
        self,
        population: List[np.ndarray]
    ) -> List[float]:
        """
        Evaluate a population of candidate solutions.
        
        Uses multiprocessing if n_jobs > 1.
        
        Args:
            population: List of parameter vectors
        
        Returns:
            List of costs for each candidate
        """
        if self.n_jobs > 1:
            # Parallel evaluation
            with multiprocessing.Pool(self.n_jobs) as pool:
                costs = pool.map(self._objective_with_penalty, population)
        else:
            # Sequential evaluation
            costs = [self._objective_with_penalty(params) for params in population]
        
        return costs
    
    def optimize(
        self,
        x0: Optional[np.ndarray] = None,
        callback: Optional[Callable] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Run CMA-ES optimization.
        
        Args:
            x0: Initial parameter vector (default: zeros)
            callback: Optional callback function called after each iteration
                     Signature: callback(iteration, best_cost, best_params)
        
        Returns:
            Tuple of (best_params, best_cost)
        """
        # Initialize starting point
        if x0 is None:
            x0 = np.zeros(self.n_params)
        
        if self.verbose:
            print("=" * 80)
            print("CMA-ES Optimization")
            print("=" * 80)
            print(f"Parameters: {self.n_params}")
            print(f"Population size: {self.population_size}")
            print(f"Max iterations: {self.max_iterations}")
            print(f"Initial sigma: {self.sigma0}")
            print(f"L2 penalty: {self.l2_penalty}")
            print(f"Parallel workers: {self.n_jobs}")
            print(f"Random seed: {self.seed}")
            print("=" * 80)
        
        # Initialize CMA-ES
        es = cma.CMAEvolutionStrategy(
            x0=x0,
            sigma0=self.sigma0,
            inopts={
                'seed': self.seed,
                'popsize': self.population_size,
                'verb_disp': 1 if self.verbose else 0
            }
        )
        
        # Optimization loop
        start_time = time.time()
        iteration = 0
        
        iterator = range(self.max_iterations)
        if self.verbose:
            iterator = tqdm(iterator, desc="CMA-ES Optimization")
        
        for iteration in iterator:
            # Ask for new candidate solutions
            population = es.ask()
            
            # Evaluate population
            costs = self._evaluate_population(population)
            
            # Tell CMA-ES the results
            es.tell(population, costs)
            
            # Track best solution
            best_idx = np.argmin(costs)
            if costs[best_idx] < self.best_cost:
                self.best_cost = costs[best_idx]
                self.best_params = population[best_idx].copy()
            
            # Update history
            self.history['iterations'].append(iteration)
            self.history['best_costs'].append(self.best_cost)
            self.history['mean_costs'].append(np.mean(costs))
            self.history['best_params'].append(self.best_params.copy())
            
            # Callback
            if callback:
                callback(iteration, self.best_cost, self.best_params)
            
            # Checkpointing
            if self.checkpoint_dir and (iteration + 1) % 100 == 0:
                self._save_checkpoint(iteration)
            
            # Check stopping criteria
            if es.stop():
                if self.verbose:
                    print(f"\nStopping criteria met at iteration {iteration}")
                break
        
        elapsed_time = time.time() - start_time
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("Optimization Complete")
            print("=" * 80)
            print(f"Total iterations: {iteration + 1}")
            print(f"Best cost: {self.best_cost:.6f}")
            print(f"Time elapsed: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
            print("=" * 80)
        
        return self.best_params, self.best_cost
    
    def _save_checkpoint(self, iteration: int):
        """Save optimization checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_iter{iteration}.pkl"
        
        checkpoint = {
            'iteration': iteration,
            'best_params': self.best_params,
            'best_cost': self.best_cost,
            'history': self.history,
            'config': {
                'n_params': self.n_params,
                'sigma0': self.sigma0,
                'population_size': self.population_size,
                'l2_penalty': self.l2_penalty,
                'seed': self.seed
            }
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load optimization checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.best_params = checkpoint['best_params']
        self.best_cost = checkpoint['best_cost']
        self.history = checkpoint['history']
        
        return checkpoint['iteration']
    
    def get_training_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get training curves.
        
        Returns:
            Tuple of (iterations, best_costs, mean_costs)
        """
        return (
            np.array(self.history['iterations']),
            np.array(self.history['best_costs']),
            np.array(self.history['mean_costs'])
        )


def train_multiple_seeds(
    objective_function: Callable,
    n_params: int,
    n_seeds: int = 5,
    sigma0: float = 0.05,
    population_size: int = 50,
    max_iterations: int = 3000,
    n_jobs: int = 1,
    l2_penalty: float = 0.01,
    checkpoint_dir: Optional[str] = None,
    save_all_models: bool = True,
    verbose: bool = True
) -> List[Dict]:
    """
    Train with multiple random seeds for robustness.
    
    This is what Team ambitiousengineers did for Phase 1:
    - 5 different random seeds
    - Saved checkpoints at iterations 2500, 2600, 2700, 2800, 2900, 3000
    - Total: 30 models (5 seeds × 6 checkpoints each)
    
    Args:
        objective_function: Objective to minimize
        n_params: Number of parameters
        n_seeds: Number of random seeds
        sigma0: Initial step size
        population_size: CMA-ES population size
        max_iterations: Max iterations per seed
        n_jobs: Parallel workers
        l2_penalty: L2 regularization
        checkpoint_dir: Directory for checkpoints
        save_all_models: Save all intermediate models
        verbose: Print progress
    
    Returns:
        List of results dictionaries for each seed
    """
    results = []
    
    for seed_idx in range(n_seeds):
        if verbose:
            print(f"\n{'='*80}")
            print(f"Training with Seed {seed_idx + 1}/{n_seeds}")
            print(f"{'='*80}\n")
        
        # Create seed-specific checkpoint directory
        if checkpoint_dir:
            seed_checkpoint_dir = Path(checkpoint_dir) / f"seed_{seed_idx}"
        else:
            seed_checkpoint_dir = None
        
        # Initialize optimizer
        optimizer = CMAESOptimizer(
            objective_function=objective_function,
            n_params=n_params,
            sigma0=sigma0,
            population_size=population_size,
            max_iterations=max_iterations,
            n_jobs=n_jobs,
            l2_penalty=l2_penalty,
            seed=seed_idx,
            checkpoint_dir=str(seed_checkpoint_dir) if seed_checkpoint_dir else None,
            verbose=verbose
        )
        
        # Run optimization
        best_params, best_cost = optimizer.optimize()
        
        # Save result
        result = {
            'seed': seed_idx,
            'best_params': best_params,
            'best_cost': best_cost,
            'history': optimizer.history
        }
        results.append(result)
        
        # Save intermediate checkpoints if requested
        if save_all_models and checkpoint_dir:
            checkpoint_iterations = [2500, 2600, 2700, 2800, 2900, max_iterations]
            for checkpoint_iter in checkpoint_iterations:
                if checkpoint_iter <= len(optimizer.history['iterations']):
                    idx = min(checkpoint_iter - 1, len(optimizer.history['best_params']) - 1)
                    params = optimizer.history['best_params'][idx]
                    
                    model_path = Path(checkpoint_dir) / f"model_seed{seed_idx}_iter{checkpoint_iter}.npy"
                    np.save(model_path, params)
                    
                    if verbose:
                        print(f"  Saved checkpoint: {model_path.name}")
    
    if verbose:
        print(f"\n{'='*80}")
        print("Multi-Seed Training Complete")
        print(f"{'='*80}")
        print(f"Seeds trained: {n_seeds}")
        print("Best costs per seed:")
        for i, result in enumerate(results):
            print(f"  Seed {i}: {result['best_cost']:.6f}")
        print(f"Overall best: {min(r['best_cost'] for r in results):.6f}")
        print(f"{'='*80}\n")
    
    return results


def select_best_model(
    results: List[Dict],
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, float, int]:
    """
    Select best model from multiple seed runs.
    
    Args:
        results: List of training results from train_multiple_seeds
        save_path: Optional path to save best model
    
    Returns:
        Tuple of (best_params, best_cost, best_seed)
    """
    # Find best result
    best_idx = np.argmin([r['best_cost'] for r in results])
    best_result = results[best_idx]
    
    best_params = best_result['best_params']
    best_cost = best_result['best_cost']
    best_seed = best_result['seed']
    
    # Save if requested
    if save_path:
        np.save(save_path, best_params)
        print(f"Best model saved to: {save_path}")
    
    return best_params, best_cost, best_seed


if __name__ == "__main__":
    print("Testing CMA-ES Optimizer")
    print("=" * 80)
    
    # Test 1: Simple quadratic function
    print("\nTest 1: Quadratic Function Optimization")
    print("-" * 80)
    
    def quadratic_objective(params):
        """Simple quadratic: f(x) = ||x - target||^2"""
        target = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        return np.sum((params[:5] - target) ** 2)
    
    optimizer = CMAESOptimizer(
        objective_function=quadratic_objective,
        n_params=5,
        sigma0=1.0,
        population_size=10,
        max_iterations=50,
        n_jobs=1,
        verbose=True
    )
    
    best_params, best_cost = optimizer.optimize()
    
    print(f"\nOptimization Results:")
    print(f"  Best parameters: {best_params[:5]}")
    print(f"  Target: [1.0, 2.0, 3.0, 4.0, 5.0]")
    print(f"  Final cost: {best_cost:.6f}")
    print(f"  Expected: ~0.0")
    
    # Test 2: Rosenbrock function (harder)
    print("\nTest 2: Rosenbrock Function")
    print("-" * 80)
    
    def rosenbrock_objective(params):
        """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
        x, y = params[0], params[1]
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    optimizer2 = CMAESOptimizer(
        objective_function=rosenbrock_objective,
        n_params=2,
        sigma0=0.5,
        population_size=20,
        max_iterations=100,
        n_jobs=1,
        verbose=True
    )
    
    best_params2, best_cost2 = optimizer2.optimize()
    
    print(f"\nOptimization Results:")
    print(f"  Best parameters: [{best_params2[0]:.4f}, {best_params2[1]:.4f}]")
    print(f"  Target: [1.0, 1.0]")
    print(f"  Final cost: {best_cost2:.6f}")
    print(f"  Expected: ~0.0")
    
    # Test 3: Training curves
    print("\nTest 3: Training Curves")
    print("-" * 80)
    
    iterations, best_costs, mean_costs = optimizer2.get_training_curve()
    
    print(f"Training history:")
    print(f"  Total iterations: {len(iterations)}")
    print(f"  Initial cost: {best_costs[0]:.6f}")
    print(f"  Final cost: {best_costs[-1]:.6f}")
    print(f"  Improvement: {((best_costs[0] - best_costs[-1])/best_costs[0]*100):.1f}%")
    
    # Test 4: L2 regularization
    print("\nTest 4: L2 Regularization")
    print("-" * 80)
    
    def simple_objective(params):
        """Objective that prefers large parameters without regularization"""
        return -np.sum(params**2)  # Wants params -> infinity
    
    # Without regularization
    optimizer3a = CMAESOptimizer(
        objective_function=simple_objective,
        n_params=5,
        sigma0=1.0,
        population_size=10,
        max_iterations=20,
        l2_penalty=0.0,
        verbose=False
    )
    params_no_reg, _ = optimizer3a.optimize()
    
    # With regularization
    optimizer3b = CMAESOptimizer(
        objective_function=simple_objective,
        n_params=5,
        sigma0=1.0,
        population_size=10,
        max_iterations=20,
        l2_penalty=0.1,
        verbose=False
    )
    params_with_reg, _ = optimizer3b.optimize()
    
    print(f"Without L2 penalty:")
    print(f"  Param norm: {np.linalg.norm(params_no_reg):.4f}")
    print(f"With L2 penalty (0.1):")
    print(f"  Param norm: {np.linalg.norm(params_with_reg):.4f}")
    print(f"  (L2 penalty keeps parameters small)")
    
    # Test 5: Multi-seed training (small scale)
    print("\nTest 5: Multi-Seed Training")
    print("-" * 80)
    
    results = train_multiple_seeds(
        objective_function=quadratic_objective,
        n_params=5,
        n_seeds=3,
        sigma0=1.0,
        population_size=10,
        max_iterations=30,
        n_jobs=1,
        l2_penalty=0.0,
        save_all_models=False,
        verbose=False
    )
    
    print(f"Trained {len(results)} models with different seeds")
    for i, result in enumerate(results):
        print(f"  Seed {i}: cost = {result['best_cost']:.6f}")
    
    best_params, best_cost, best_seed = select_best_model(results)
    print(f"\nBest model: Seed {best_seed} with cost {best_cost:.6f}")
    
    print("\n" + "=" * 80)
    print("✅ CMA-ES Optimizer Tests Completed!")
    print("\nThis optimizer will be used to train:")
    print("  - Phase 1 policy (184 params, 3000 iters, 5 seeds)")
    print("  - Phase 2/3 policy (465 params, 10000 iters, 1 seed)")