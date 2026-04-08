# ==============================================================
# hems/agents/ambitious_engineers_algorithm.py - fprecasting.py
# ==============================================================

"""
Forecasting Models - Demand and Solar Generation Prediction
============================================================
Faithful reproduction of Team ambitiousengineers' MLP-based forecasting approach.

Two separate models:
1. Demand Forecasting: Predicts future electricity demand
   - Features: Lags + Exponential smoothing + Time embeddings
   - Time-of-day and day-of-week embeddings (learned)
   - Forecasts 10 steps ahead
   
2. Solar Forecasting: Predicts future PV generation
   - Features: Lags of target + Leads of phase 1 average generation
   - No time embeddings (solar is purely weather-dependent)
   - Forecasts 10 steps ahead

Both models use:
- Multi-layer perceptrons with skip connections
- Bayesian optimization for hyperparameter tuning (leave-one-building-out CV)
- Training on historical data from Phase 1

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings


# ============================================================================
# Neural Network Architectures
# ============================================================================

class MLP(nn.Module):
    """
    Standard multi-layer perceptron.
    
    Architecture: Input -> Hidden (ReLU + Dropout) -> Output
    """
    
    def __init__(self, n_in: int, n_hidden: int, n_out: int, dropout: float = 0.1):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(n_in, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_out)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.linear2(self.dropout(x))
        return x


class SkipMLP(nn.Module):
    """
    MLP with skip connection (residual connection).
    
    Architecture: x -> Hidden (ReLU + Dropout) -> Output + x
    
    Skip connections help with gradient flow and training stability.
    """
    
    def __init__(self, n_in: int, n_hidden: int, dropout: float = 0.1):
        super(SkipMLP, self).__init__()
        self.linear1 = nn.Linear(n_in, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_in)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x0):
        x = self.activation(self.linear1(x0))
        x = self.linear2(self.dropout(x))
        return x + x0  # Skip connection


# ============================================================================
# Demand Forecasting Model
# ============================================================================

class DemandModel(nn.Module):
    """
    Demand forecasting model with time embeddings.
    
    Key Features:
    - Time-of-day embeddings (24 hours)
    - Day-of-week embeddings (7 days)
    - Historical features (lags + exponential smoothing)
    - Multi-horizon prediction (10 steps ahead)
    
    Architecture:
    - Embedding layer for temporal features
    - 3-layer MLP with skip connections
    - Clamps output to non-negative (demand >= 0)
    """
    
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        n_hidden: int,
        dropout: float = 0.1,
        emb_dim: int = 8
    ):
        """
        Args:
            n_features: Number of input features (lags + exponential smoothing)
            n_targets: Number of forecast horizons (default: 10)
            n_hidden: Hidden layer size
            dropout: Dropout probability
            emb_dim: Embedding dimension for time features
        """
        super(DemandModel, self).__init__()
        
        # Time embeddings
        self.tod_embedding = nn.Embedding(24, emb_dim)  # Time of day
        self.dow_embedding = nn.Embedding(7, emb_dim)   # Day of week
        
        # MLP layers
        self.mlp1 = MLP(n_features + 2 * emb_dim, n_hidden, n_hidden, dropout)
        self.mlp2 = SkipMLP(n_hidden, n_hidden, dropout)
        self.mlp3 = MLP(n_hidden, n_hidden, n_targets, dropout)
    
    def forward(self, time_of_day, day_of_week, features):
        """
        Forward pass.
        
        Args:
            time_of_day: Hour of day [0-23], shape (batch_size,)
            day_of_week: Day of week [0-6], shape (batch_size,)
            features: Historical features, shape (batch_size, n_features)
        
        Returns:
            Demand forecasts, shape (batch_size, n_targets)
        """
        # Get embeddings
        tod = self.tod_embedding(time_of_day)
        dow = self.dow_embedding(day_of_week)
        
        # Concatenate all features
        x = torch.cat([tod, dow, features], dim=1)
        
        # Pass through MLP
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        
        # Ensure non-negative demand
        return x.clamp_min(0)


class DemandDataset(Dataset):
    """PyTorch dataset for demand forecasting."""
    
    def __init__(
        self,
        time_of_day: np.ndarray,
        day_of_week: np.ndarray,
        features: np.ndarray,
        targets: np.ndarray,
        device: str = 'cpu'
    ):
        self.time_of_day = torch.tensor(time_of_day, dtype=torch.long, device=device)
        self.day_of_week = torch.tensor(day_of_week, dtype=torch.long, device=device)
        self.features = torch.tensor(features, dtype=torch.float32, device=device)
        self.targets = torch.tensor(targets, dtype=torch.float32, device=device)
        self.device = device
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        return (
            self.time_of_day[index],
            self.day_of_week[index],
            self.features[index],
            self.targets[index]
        )


# ============================================================================
# Solar Forecasting Model
# ============================================================================

class SolarModel(nn.Module):
    """
    Solar generation forecasting model.
    
    Key Features:
    - No time embeddings (solar is weather-dependent, not time-dependent)
    - Uses lags of target + leads of average generation
    - Multi-horizon prediction (10 steps ahead)
    
    Architecture:
    - 3-layer MLP with skip connections
    - Clamps output to non-negative (generation >= 0)
    """
    
    def __init__(
        self,
        n_features: int,
        n_horizon: int,
        n_hidden: int,
        dropout: float = 0.1
    ):
        """
        Args:
            n_features: Number of input features
            n_horizon: Number of forecast horizons
            n_hidden: Hidden layer size
            dropout: Dropout probability
        """
        super(SolarModel, self).__init__()
        
        self.mlp1 = MLP(n_features, n_hidden, n_hidden, dropout)
        self.mlp2 = SkipMLP(n_hidden, n_hidden, dropout)
        self.mlp3 = MLP(n_hidden, n_hidden, n_horizon, dropout)
    
    def forward(self, features):
        """
        Forward pass.
        
        Args:
            features: Historical features, shape (batch_size, n_features)
        
        Returns:
            Solar forecasts, shape (batch_size, n_horizon)
        """
        x = self.mlp1(features)
        x = self.mlp2(x)
        x = self.mlp3(x)
        
        # Ensure non-negative generation
        return x.clamp_min(0)


class SolarDataset(Dataset):
    """PyTorch dataset for solar forecasting."""
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        device: str = 'cpu'
    ):
        self.features = torch.tensor(features, dtype=torch.float32, device=device)
        self.targets = torch.tensor(targets, dtype=torch.float32, device=device)
        self.device = device
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        return self.features[index], self.targets[index]


# ============================================================================
# Exponential Smoothing for Demand Features
# ============================================================================

class ExponentialSmoother:
    """
    Exponential smoothing feature extractor.
    
    Maintains daily and weekly smoothed estimates that adapt over time.
    Uses multiple decay rates to capture different timescales.
    
    This is used as input features for the demand forecasting model.
    """
    
    def __init__(
        self,
        daily_baseline: np.ndarray,
        weekly_baseline: np.ndarray,
        n_lags: int = 10,
        n_leads: int = 10,
        decays: Optional[List[float]] = None
    ):
        """
        Args:
            daily_baseline: Daily pattern baseline, shape (24,)
            weekly_baseline: Weekly pattern baseline, shape (168,)
            n_lags: Number of historical values to track
            n_leads: Number of future values to predict
            decays: Decay rates for exponential smoothing
        """
        self.daily_baseline = daily_baseline
        self.weekly_baseline = weekly_baseline
        self.n_lags = n_lags
        self.n_leads = n_leads
        
        if decays is None:
            decays = [0.9, 0.95, 0.99]  # Multiple timescales
        self.decays = decays
        
        # Initialize state
        self.daily_b = daily_baseline.copy()
        self.daily_x = daily_baseline.copy()
        self.daily_n = np.ones(24)
        
        self.weekly_b = weekly_baseline.copy()
        self.weekly_x = weekly_baseline.copy()
        self.weekly_n = np.ones(168)
        
        self.time_index = 0
    
    def update(self, value: float) -> np.ndarray:
        """
        Update smoothed estimates with new observation.
        
        Args:
            value: New demand observation
        
        Returns:
            Feature vector with smoothed estimates
        """
        # Update daily state
        i = self.time_index % 24
        self.daily_b[i] = self._uniform_update(value, self.daily_b[i], self.daily_n[i])
        self.daily_x[i] = self._average_update(
            value, self.daily_x[i], self.daily_b[i], self.daily_n[i], self.decays
        )
        self.daily_n[i] += 1
        
        # Update weekly state
        j = self.time_index % 168
        self.weekly_b[j] = self._uniform_update(value, self.weekly_b[j], self.weekly_n[j])
        self.weekly_x[j] = self._average_update(
            value, self.weekly_x[j], self.weekly_b[j], self.weekly_n[j], self.decays
        )
        self.weekly_n[j] += 1
        
        # Extract features
        i_range = np.arange(self.time_index - self.n_lags, self.time_index + self.n_leads) % 24
        j_range = np.arange(self.time_index - self.n_lags, self.time_index + self.n_leads) % 168
        features = np.concatenate([self.daily_x[i_range], self.weekly_x[j_range]])
        
        self.time_index += 1
        return features
    
    @staticmethod
    def _uniform_update(x: float, b: float, n: float) -> float:
        """Uniform average update."""
        return (n * b + x) / (n + 1)
    
    @staticmethod
    def _average_update(x: float, prev: float, b: float, n: float, decays: List[float]) -> float:
        """Exponential moving average update with multiple decay rates."""
        alpha = np.array([1.0 / (n + 1)] + [1.0 - d for d in decays])
        values = np.array([b] + [prev] * len(decays))
        return np.dot(alpha, values) / alpha.sum()


# ============================================================================
# Forecaster Wrapper Classes
# ============================================================================

class DemandForecaster:
    """
    High-level demand forecasting wrapper.
    
    Handles:
    - Feature engineering (lags + exponential smoothing)
    - Model training with proper train/val split
    - Online forecasting during evaluation
    """
    
    def __init__(
        self,
        n_lags: int = 10,
        n_targets: int = 10,
        n_hidden: int = 256,
        dropout: float = 0.1,
        emb_dim: int = 8,
        device: str = 'cpu'
    ):
        """Initialize demand forecaster."""
        self.n_lags = n_lags
        self.n_targets = n_targets
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.emb_dim = emb_dim
        self.device = device
        
        self.model = None
        self.feature_mean = None
        self.feature_std = None
    
    def prepare_features(
        self,
        data: pd.DataFrame,
        target_col: str = 'non_shiftable_load'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features for training.
        
        Creates:
        - Time-of-day features
        - Day-of-week features
        - Lag features
        - Exponential smoothing features
        - Target forecasts
        
        Args:
            data: DataFrame with demand time series
            target_col: Name of target column
        
        Returns:
            Tuple of (time_of_day, day_of_week, features, targets)
        """
        # Implementation would go here
        # This is a placeholder - full implementation follows the same pattern
        # as their 2_create_demand_model.py
        pass
    
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        n_epochs: int = 25,
        batch_size: int = 256,
        learning_rate: float = 3e-4,
        verbose: bool = True
    ):
        """Train the demand forecasting model."""
        # Training implementation
        pass
    
    def forecast(self, current_state: Dict) -> np.ndarray:
        """
        Generate forecast for next n_targets steps.
        
        Args:
            current_state: Dict with current demand, time, etc.
        
        Returns:
            Demand forecasts, shape (n_targets,)
        """
        # Forecasting implementation
        pass


class SolarForecaster:
    """
    High-level solar forecasting wrapper.
    
    Similar to DemandForecaster but for solar generation.
    """
    
    def __init__(
        self,
        n_lags: int = 216,  # 9 days * 24 hours
        n_targets: int = 10,
        n_hidden: int = 2048,
        dropout: float = 0.0,
        device: str = 'cpu'
    ):
        """Initialize solar forecaster."""
        self.n_lags = n_lags
        self.n_targets = n_targets
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.device = device
        
        self.model = None
    
    def train(
        self,
        train_data: pd.DataFrame,
        phase1_avg_generation: np.ndarray,
        n_epochs: int = 25,
        batch_size: int = 256,
        verbose: bool = True
    ):
        """Train the solar forecasting model."""
        pass
    
    def forecast(self, current_state: Dict) -> np.ndarray:
        """Generate solar forecast."""
        pass


if __name__ == "__main__":
    print("Testing Forecasting Models")
    print("=" * 80)
    
    # Test neural network architectures
    print("\nTest 1: Neural Network Architectures")
    print("-" * 80)
    
    # Test MLP
    mlp = MLP(n_in=10, n_hidden=64, n_out=5, dropout=0.1)
    x = torch.randn(32, 10)
    y = mlp(x)
    print(f"MLP output shape: {y.shape} (expected: torch.Size([32, 5]))")
    
    # Test SkipMLP
    skip_mlp = SkipMLP(n_in=64, n_hidden=128, dropout=0.1)
    x = torch.randn(32, 64)
    y = skip_mlp(x)
    print(f"SkipMLP output shape: {y.shape} (expected: torch.Size([32, 64]))")
    
    # Test Demand Model
    demand_model = DemandModel(n_features=50, n_targets=10, n_hidden=256, emb_dim=8)
    tod = torch.randint(0, 24, (32,))
    dow = torch.randint(0, 7, (32,))
    features = torch.randn(32, 50)
    forecasts = demand_model(tod, dow, features)
    print(f"Demand model output shape: {forecasts.shape} (expected: torch.Size([32, 10]))")
    print(f"All forecasts non-negative: {(forecasts >= 0).all().item()}")
    
    # Test Solar Model
    solar_model = SolarModel(n_features=442, n_horizon=10, n_hidden=2048)
    features = torch.randn(32, 442)
    forecasts = solar_model(features)
    print(f"Solar model output shape: {forecasts.shape} (expected: torch.Size([32, 10]))")
    print(f"All forecasts non-negative: {(forecasts >= 0).all().item()}")
    
    # Test Exponential Smoother
    print("\nTest 2: Exponential Smoother")
    print("-" * 80)
    daily_baseline = np.random.rand(24) * 2 + 1
    weekly_baseline = np.random.rand(168) * 2 + 1
    smoother = ExponentialSmoother(daily_baseline, weekly_baseline, n_lags=5, n_leads=5)
    
    # Simulate some updates
    for i in range(50):
        value = 2.0 + np.sin(i * 2 * np.pi / 24) + np.random.randn() * 0.1
        features = smoother.update(value)
    
    print(f"Smoothed features shape: {features.shape}")
    print(f"Feature statistics: mean={features.mean():.4f}, std={features.std():.4f}")
    
    print("\n" + "=" * 80)
    print("✅ Forecasting Model Tests Completed!")
    print("\nNext Steps:")
    print("1. Implement full training pipeline for demand forecaster")
    print("2. Implement full training pipeline for solar forecaster")
    print("3. These forecasters will be used in Phase 2/3 policy networks")