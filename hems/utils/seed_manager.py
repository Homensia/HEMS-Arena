"""
Comprehensive Seed Management for HEMS Benchmark Reproducibility
Ensures deterministic behavior across all components.
"""

import random
import numpy as np
import torch
import os
from typing import Optional


class SeedManager:
    """
    Centralized seed management for perfect reproducibility.
    
    Usage:
        SeedManager.set_seed(42)  # At start of benchmark
        SeedManager.set_seed(42, context="training")  # Before training
        SeedManager.set_seed(42, context="validation")  # Before validation
    """
    
    _global_seed: Optional[int] = None
    _seed_history: list = []
    
    @classmethod
    def set_seed(cls, seed: int, context: str = "global", verbose: bool = True):
        """
        Set all random seeds comprehensively for perfect reproducibility.
        
        Args:
            seed: Random seed value
            context: Context description for logging (e.g., "training", "validation")
            verbose: Whether to print seed setting confirmation
        """
        cls._global_seed = seed
        cls._seed_history.append({'seed': seed, 'context': context})
        
        # Python built-in random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # PyTorch deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Python hash seed (for dictionary iteration, etc.)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # TensorFlow (if used by any agents)
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
        
        if verbose:
            print(f"🎲 Seed set to {seed} (context: {context})")
    
    @classmethod
    def get_current_seed(cls) -> Optional[int]:
        """Get the current global seed."""
        return cls._global_seed
    
    @classmethod
    def get_seed_history(cls) -> list:
        """Get history of all seed settings."""
        return cls._seed_history.copy()
    
    @classmethod
    def reset(cls):
        """Reset seed manager state."""
        cls._global_seed = None
        cls._seed_history = []


def set_all_random_seeds(seed: int, context: str = "global", verbose: bool = True):
    """
    Convenience function for setting all random seeds.
    
    This is the main function to call throughout the codebase.
    
    Args:
        seed: Random seed value
        context: Context description for logging
        verbose: Whether to print confirmation
    
    Example:
        >>> from hems.utils.seed_manager import set_all_random_seeds
        >>> set_all_random_seeds(42, context="training_phase")
    """
    SeedManager.set_seed(seed, context, verbose)


def get_reproducibility_report() -> dict:
    """
    Generate a report on reproducibility settings.
    
    Returns:
        Dictionary with reproducibility status
    """
    report = {
        'current_seed': SeedManager.get_current_seed(),
        'seed_history': SeedManager.get_seed_history(),
        'pytorch_deterministic': torch.backends.cudnn.deterministic,
        'pytorch_benchmark': torch.backends.cudnn.benchmark,
        'cuda_available': torch.cuda.is_available(),
        'pythonhashseed': os.environ.get('PYTHONHASHSEED', 'not set')
    }
    
    return report


# Auto-set seed from environment variable if available
if 'HEMS_SEED' in os.environ:
    try:
        env_seed = int(os.environ['HEMS_SEED'])
        SeedManager.set_seed(env_seed, context="environment_variable", verbose=True)
    except ValueError:
        pass