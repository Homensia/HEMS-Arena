# =====================================================================
# hems/agents/ambitious_engineers_algorithm.py - create_solar_model.py
# =====================================================================


"""
3_create_solar_model.py - Train Solar Generation Forecasting Model
===================================================================
Expected runtime: ~5 minutes

This script trains the solar forecasting MLP with:
- Lags of target variable (solar generation)
- Leads of Phase 1 average generation
- NO time embeddings (solar is weather-dependent, not time-dependent)
- Leave-one-building-out cross-validation
- Larger hidden layer (2048 units) due to more complex patterns

The model predicts future solar generation 10 steps ahead.

Output:
- data/models/solar_forecaster.pth
- data/models/solar_forecaster_config.json
- data/models/solar_training_stats.json
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from hems.algorithms.ambitious_engineers_algorithm.forecasting import SolarModel


class SolarForecasterTrainer:
    """
    Trainer for solar generation forecasting model.
    
    Key differences from demand forecaster:
    - NO time embeddings (weather-dependent)
    - Uses lags of target + leads of average generation
    - Larger network (2048 hidden units)
    - Different feature engineering
    """
    
    def __init__(
        self,
        observations_path: str = "data/external/observations.csv",
        output_dir: str = "data/models",
        n_lags: int = 216,  # 9 days * 24 hours
        n_targets: int = 10,
        n_hidden: int = 2048,
        dropout: float = 0.0  # No dropout for solar
    ):
        self.observations_path = observations_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_lags = n_lags
        self.n_targets = n_targets
        self.n_hidden = n_hidden
        self.dropout = dropout
        
        # Load data
        print("Loading observations...")
        self.observations = pd.read_csv(observations_path)
        print(f"  Loaded {len(self.observations)} observations")
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Using device: {self.device}")
    
    def compute_phase1_average_generation(self) -> np.ndarray:
        """
        Compute average solar generation across all buildings.
        
        This is used as a feature (leads of average generation).
        
        Returns:
            Array of average generation values
        """
        # Group by timestep and compute mean solar generation
        avg_generation = self.observations.groupby('timestep')['solar_generation'].mean().values
        
        print(f"  Computed Phase 1 average generation: {len(avg_generation)} timesteps")
        print(f"    Mean: {avg_generation.mean():.2f} kW")
        print(f"    Max: {avg_generation.max():.2f} kW")
        
        return avg_generation
    
    def prepare_features_and_targets(
        self,
        building_data: pd.DataFrame,
        avg_generation: np.ndarray
    ) -> tuple:
        """
        Prepare features and targets for a single building.
        
        Features:
        - Lags of target (n_lags)
        - Leads of average generation (n_targets)
        
        Args:
            building_data: DataFrame for one building
            avg_generation: Average generation across all buildings
        
        Returns:
            (features, targets)
        """
        # Target: solar generation
        solar = building_data['solar_generation'].values
        
        # Feature 1: Lags of target
        lag_features = []
        for i in range(self.n_lags):
            lagged = np.roll(solar, i+1)
            lagged[:i+1] = 0.0  # Pad with zeros (no generation at start)
            lag_features.append(lagged)
        lag_features = np.column_stack(lag_features)
        
        # Feature 2: Leads of average generation
        lead_features = []
        for i in range(self.n_targets):
            lead = np.roll(avg_generation[:len(solar)], -(i+1))
            lead[-i-1:] = avg_generation[-1]  # Pad with last value
            lead_features.append(lead)
        lead_features = np.column_stack(lead_features)
        
        # Combine features
        features = np.column_stack([lag_features, lead_features])
        
        # Create targets (next n_targets steps)
        targets = []
        for i in range(len(solar)):
            target = solar[i:i+self.n_targets]
            if len(target) < self.n_targets:
                target = np.pad(target, (0, self.n_targets - len(target)), 
                               constant_values=0.0)
            targets.append(target)
        targets = np.array(targets)
        
        # Remove last n_targets samples (incomplete targets)
        n_valid = len(solar) - self.n_targets
        features = features[:n_valid]
        targets = targets[:n_valid]
        
        return features, targets
    
    def train_with_cv(self, n_epochs: int = 25, batch_size: int = 256, lr: float = 3e-4):
        """
        Train model with leave-one-building-out cross-validation.
        
        Args:
            n_epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
        
        Returns:
            CV results and average validation loss
        """
        print("\nTraining with leave-one-building-out CV...")
        print("-"*80)
        
        # Compute average generation (used for all folds)
        avg_generation = self.compute_phase1_average_generation()
        
        cv_results = []
        
        # Leave-one-building-out
        for test_building in range(1, 6):
            print(f"\nFold {test_building}: Testing on Building {test_building}")
            print("-"*60)
            
            # Prepare training data (all buildings except test)
            train_data = []
            for building_id in range(1, 6):
                if building_id == test_building:
                    continue
                
                building_obs = self.observations[
                    self.observations['building_num'] == building_id
                ].copy()
                
                print(f"  Processing Building {building_id} (train)...")
                feats, targs = self.prepare_features_and_targets(
                    building_obs, avg_generation
                )
                train_data.append((feats, targs))
            
            # Concatenate training data
            train_feats = np.concatenate([d[0] for d in train_data])
            train_targs = np.concatenate([d[1] for d in train_data])
            
            # Normalize features
            scaler = StandardScaler()
            train_feats = scaler.fit_transform(train_feats)
            
            # Prepare test data
            test_obs = self.observations[
                self.observations['building_num'] == test_building
            ].copy()
            
            print(f"  Processing Building {test_building} (test)...")
            test_feats, test_targs = self.prepare_features_and_targets(
                test_obs, avg_generation
            )
            test_feats = scaler.transform(test_feats)
            
            # Create datasets
            train_dataset = TensorDataset(
                torch.tensor(train_feats, dtype=torch.float32),
                torch.tensor(train_targs, dtype=torch.float32)
            )
            
            test_dataset = TensorDataset(
                torch.tensor(test_feats, dtype=torch.float32),
                torch.tensor(test_targs, dtype=torch.float32)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model
            n_features = train_feats.shape[1]
            model = SolarModel(
                n_features=n_features,
                n_horizon=self.n_targets,
                n_hidden=self.n_hidden,
                dropout=self.dropout
            ).to(self.device)
            
            # Optimizer and loss
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            # Training loop
            print(f"\n  Training for {n_epochs} epochs...")
            best_val_loss = float('inf')
            
            for epoch in range(n_epochs):
                # Train
                model.train()
                train_loss = 0.0
                for feats, targs in train_loader:
                    feats = feats.to(self.device)
                    targs = targs.to(self.device)
                    
                    optimizer.zero_grad()
                    preds = model(feats)
                    loss = criterion(preds, targs)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validate
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for feats, targs in test_loader:
                        feats = feats.to(self.device)
                        targs = targs.to(self.device)
                        
                        preds = model(feats)
                        loss = criterion(preds, targs)
                        val_loss += loss.item()
                
                val_loss /= len(test_loader)
                
                if (epoch + 1) % 5 == 0:
                    print(f"    Epoch {epoch+1}/{n_epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            
            print(f"  Best validation loss: {best_val_loss:.6f}")
            
            cv_results.append({
                'test_building': test_building,
                'val_loss': best_val_loss,
                'scaler': scaler
            })
        
        # Calculate average CV performance
        avg_val_loss = np.mean([r['val_loss'] for r in cv_results])
        print(f"\n{'='*80}")
        print(f"Average CV Loss: {avg_val_loss:.6f}")
        print(f"{'='*80}")
        
        return cv_results, avg_val_loss, avg_generation
    
    def train_final_model(
        self,
        avg_generation: np.ndarray,
        n_epochs: int = 25,
        batch_size: int = 256,
        lr: float = 3e-4
    ):
        """
        Train final model on all buildings.
        
        Args:
            avg_generation: Pre-computed average generation
            n_epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
        
        Returns:
            Trained model, scaler, and validation loss
        """
        print("\nTraining final model on all buildings...")
        print("-"*80)
        
        # Prepare data for all buildings
        all_data = []
        for building_id in range(1, 6):
            building_obs = self.observations[
                self.observations['building_num'] == building_id
            ].copy()
            
            print(f"  Processing Building {building_id}...")
            feats, targs = self.prepare_features_and_targets(
                building_obs, avg_generation
            )
            all_data.append((feats, targs))
        
        # Concatenate all data
        all_feats = np.concatenate([d[0] for d in all_data])
        all_targs = np.concatenate([d[1] for d in all_data])
        
        # Normalize features
        scaler = StandardScaler()
        all_feats = scaler.fit_transform(all_feats)
        
        # Create dataset
        dataset = TensorDataset(
            torch.tensor(all_feats, dtype=torch.float32),
            torch.tensor(all_targs, dtype=torch.float32)
        )
        
        # Split into train/val (80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        n_features = all_feats.shape[1]
        model = SolarModel(
            n_features=n_features,
            n_horizon=self.n_targets,
            n_hidden=self.n_hidden,
            dropout=self.dropout
        ).to(self.device)
        
        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Training loop
        print(f"\nTraining for {n_epochs} epochs...")
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(n_epochs):
            # Train
            model.train()
            train_loss = 0.0
            for feats, targs in train_loader:
                feats = feats.to(self.device)
                targs = targs.to(self.device)
                
                optimizer.zero_grad()
                preds = model(feats)
                loss = criterion(preds, targs)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for feats, targs in val_loader:
                    feats = feats.to(self.device)
                    targs = targs.to(self.device)
                    
                    preds = model(feats)
                    loss = criterion(preds, targs)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        print(f"\nFinal model trained. Best validation loss: {best_val_loss:.6f}")
        
        return model, scaler, best_val_loss
    
    def save_model(self, model, scaler, avg_generation):
        """Save trained model, scaler, and average generation."""
        # Save model weights
        model_path = self.output_dir / "solar_forecaster.pth"
        torch.save(model.state_dict(), model_path)
        print(f"  Saved model: {model_path}")
        
        # Save scaler
        import pickle
        scaler_path = self.output_dir / "solar_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"  Saved scaler: {scaler_path}")
        
        # Save average generation
        avg_gen_path = self.output_dir / "phase1_avg_generation.npy"
        np.save(avg_gen_path, avg_generation)
        print(f"  Saved avg generation: {avg_gen_path}")
        
        # Save configuration
        config = {
            'n_lags': self.n_lags,
            'n_targets': self.n_targets,
            'n_hidden': self.n_hidden,
            'dropout': self.dropout,
            'device': str(self.device)
        }
        
        config_path = self.output_dir / "solar_forecaster_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  Saved config: {config_path}")


def main():
    print("="*80)
    print("  STEP 3: Train Solar Generation Forecasting Model")
    print("="*80)
    print("\nThis script trains an MLP to forecast solar generation 10 steps ahead")
    print("Expected runtime: ~5 minutes\n")
    
    start_time = time.time()
    
    try:
        # Initialize trainer
        trainer = SolarForecasterTrainer(
            observations_path="data/external/observations.csv",
            output_dir="data/models",
            n_lags=216,  # 9 days
            n_targets=10,
            n_hidden=2048,
            dropout=0.0
        )
        
        # Cross-validation
        print("\nPhase 1: Leave-One-Building-Out Cross-Validation")
        print("="*80)
        cv_results, avg_cv_loss, avg_generation = trainer.train_with_cv(
            n_epochs=25,
            batch_size=256,
            lr=3e-4
        )
        
        # Train final model
        print("\n\nPhase 2: Train Final Model on All Buildings")
        print("="*80)
        model, scaler, final_loss = trainer.train_final_model(
            avg_generation=avg_generation,
            n_epochs=25,
            batch_size=256,
            lr=3e-4
        )
        
        # Save model
        print("\nSaving model...")
        trainer.save_model(model, scaler, avg_generation)
        
        # Save training statistics
        stats = {
            'cv_results': [
                {'test_building': r['test_building'], 'val_loss': float(r['val_loss'])}
                for r in cv_results
            ],
            'avg_cv_loss': float(avg_cv_loss),
            'final_val_loss': float(final_loss),
            'training_time_seconds': time.time() - start_time
        }
        
        stats_path = Path("data/models") / "solar_training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved stats: {stats_path}")
        
        # Summary
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("  SOLAR MODEL TRAINING COMPLETE")
        print("="*80)
        print(f"\nElapsed time: {elapsed_time/60:.1f} minutes")
        print(f"\nCross-Validation Results:")
        for r in cv_results:
            print(f"  Building {r['test_building']}: {r['val_loss']:.6f}")
        print(f"  Average: {avg_cv_loss:.6f}")
        print(f"\nFinal Model Validation Loss: {final_loss:.6f}")
        
        print("\n" + "="*80)
        print("  NEXT STEP: Run 4_create_training_single_agents.py")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())