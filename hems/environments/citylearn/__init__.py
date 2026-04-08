"""
CityLearn environments module.
"""

from .citylearn_wrapper import CityLearnWrapper, CityLearnEnvironmentManager, CityLearnEnvironmentManagerAdapter

# Initialize dataset loaders
#if self.dataset_type == 'synthetic' and SYNTHETIC_AVAILABLE:
#    self.synthetic_loader = SyntheticDatasetLoader(config.datasets_root)
#elif self.dataset_type == 'synthetic':
#    raise ImportError("Synthetic dataset loader not available")

__all__ = [
    'CityLearnHEMSEnvironment',
    'CityLearnEnvironmentManager', 
    'SyntheticCityLearnEnv'
]