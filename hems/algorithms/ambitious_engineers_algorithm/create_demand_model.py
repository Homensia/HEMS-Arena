# ======================================================================
# hems/agents/ambitious_engineers_algorithm.py - create_demand_model.py
# ======================================================================


"""
2_create_demand_model.py - Train Demand Forecasting Model
==========================================================
Expected runtime: ~20 minutes

This script trains the demand forecasting MLP with:
- Exponential smoothing features
- Time-of-day and day-of-week embeddings
- Leave-one-building-out cross-validation
- Bayesian optimization for hyperparameters

The model predicts future demand 10 steps ahead.

Output:
- data/models/demand_forecaster.pth
- data/models/demand_forecaster_config.json
- data/models/demand_training_stats.json
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from hems.algorithms.ambitious_engineers_algorithm.forecasting import (
    DemandModel, ExponentialSmoother
)


class DemandForecasterTrainer:
    """
    Trainer for demand forecasting model.
    
    Implements the exact approach from Team ambitiousengineers:
    - Exponential smoothing features
    - Time embeddings
    - Leave-one-building-out CV
    """
    
    def __init__(
        self,
        observations_path: str = "data/external/observations.csv",
        output_dir: str = "data/models",
        n_lags: int = 10,
        n_targets: int = 10,
        n_hidden: int = 256,
        dropout: float = 0.1,
        emb_dim: int = 8
    ):
        self.observations_path = observations_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_lags = n_lags
        self.n_targets = n_targets
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.emb_dim = emb_dim
        
        # Load data
        print("Loading observations...")
        self.observations = pd.read_csv(observations_path)
        print(f"  ✓ Loaded {len(self.observations)} observations")
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Using device: {self.device}")
    
    def create_exponential_smoothing_features(
        self,
        data: pd.Series,
        n_lags: int = 10,
        n_leads: int = 10
    ) -> np.ndarray:
        """
        Create exponential smoothing features for demand data.
        
        Args:
            data: Time series of demand values
            n_lags: Number of lag steps
            n_leads: Number of lead steps
        
        Returns:
            Array of shape (T, feature_dim)
        """
        # Compute daily and weekly baselines
        data_array = data.values
        
        # Daily baseline (24 hours)
        daily_baseline = np.zeros(24)
        for h in range(24):
            daily_baseline[h] = np.mean(data_array[h::24])
        
        # Weekly baseline (168 hours = 7 days * 24 hours)
        weekly_baseline = np.zeros(168)
        for h in range(168):
            weekly_baseline[h] = np.mean(data_array[h::168])
        
        # Initialize exponential smoother
        smoother = ExponentialSmoother(
            daily_baseline=daily_baseline,
            weekly_baseline=weekly_baseline,
            n_lags=n_lags,
            n_leads=n_leads
        )
        
        # Generate features
        features_list = []
        for value in data_array:
            features = smoother.update(value)
            features_list.append(features)
        
        return np.array(features_list)
    
    def prepare_features_and_targets(
        self,
        building_data: pd.DataFrame
    ) -> tuple:
        """
        Prepare features and targets for a single building.
        
        Args:
            building_data: DataFrame for one building
        
        Returns:
            (time_of_day, day_of_week, features, targets)
        """
        # Extract time features with proper bounds and type conversion
        time_of_day = (building_data['hour'].values % 24).astype(np.int64)
        day_of_week = (building_data['day_of_week'].values % 7).astype(np.int64)
        
        # Validate ranges before continuing
        assert time_of_day.min() >= 0 and time_of_day.max() <= 23, \
            f"Hour out of range: [{time_of_day.min()}, {time_of_day.max()}]"
        assert day_of_week.min() >= 0 and day_of_week.max() <= 6, \
            f"Day of week out of range: [{day_of_week.min()}, {day_of_week.max()}]"
        
        # Target: demand
        demand = building_data['non_shiftable_load'].values
        
        # Create exponential smoothing features
        print("    Creating exponential smoothing features...")
        exp_features = self.create_exponential_smoothing_features(
            building_data['non_shiftable_load'],
            n_lags=self.n_lags,
            n_leads=self.n_targets
        )
        
        # Create lag features
        lag_features = []
        for i in range(self.n_lags):
            lagged = np.roll(demand, i+1)
            lagged[:i+1] = demand[0]  # Pad with first value
            lag_features.append(lagged)
        lag_features = np.column_stack(lag_features)
        
        # Combine all features
        features = np.column_stack([lag_features, exp_features])
        
        # Create targets (next n_targets steps)
        targets = []
        for i in range(len(demand)):
            target = demand[i:i+self.n_targets]
            if len(target) < self.n_targets:
                # Pad with last value
                target = np.pad(target, (0, self.n_targets - len(target)), 
                            constant_values=demand[-1])
            targets.append(target)
        targets = np.array(targets)
        
        # Remove last n_targets samples (incomplete targets)
        n_valid = len(demand) - self.n_targets
        time_of_day = time_of_day[:n_valid]
        day_of_week = day_of_week[:n_valid]
        features = features[:n_valid]
        targets = targets[:n_valid]
        
        return time_of_day, day_of_week, features, targets
    
    def train_with_cv(self, n_epochs: int = 25, batch_size: int = 256, lr: float = 3e-4):
        """
        Train model with leave-one-building-out cross-validation.
        
        Args:
            n_epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
        
        Returns:
            Trained model and statistics
        """
        print("\nTraining with leave-one-building-out CV...")
        print("-"*80)
        
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
                tod, dow, feats, targs = self.prepare_features_and_targets(building_obs)
                train_data.append((tod, dow, feats, targs))
            
            # Concatenate training data
            train_tod = np.concatenate([d[0] for d in train_data])
            train_dow = np.concatenate([d[1] for d in train_data])
            train_feats = np.concatenate([d[2] for d in train_data])
            train_targs = np.concatenate([d[3] for d in train_data])
            
            # Normalize features
            scaler = StandardScaler()
            train_feats = scaler.fit_transform(train_feats)
            
            # Prepare test data
            test_obs = self.observations[
                self.observations['building_num'] == test_building
            ].copy()
            
            print(f"  Processing Building {test_building} (test)...")
            test_tod, test_dow, test_feats, test_targs = \
                self.prepare_features_and_targets(test_obs)
            test_feats = scaler.transform(test_feats)
            
            # Create datasets
            train_dataset = TensorDataset(
                torch.tensor(train_tod, dtype=torch.long),
                torch.tensor(train_dow, dtype=torch.long),
                torch.tensor(train_feats, dtype=torch.float32),
                torch.tensor(train_targs, dtype=torch.float32)
            )
            
            test_dataset = TensorDataset(
                torch.tensor(test_tod, dtype=torch.long),
                torch.tensor(test_dow, dtype=torch.long),
                torch.tensor(test_feats, dtype=torch.float32),
                torch.tensor(test_targs, dtype=torch.float32)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model
            n_features = train_feats.shape[1]
            model = DemandModel(
                n_features=n_features,
                n_targets=self.n_targets,
                n_hidden=self.n_hidden,
                dropout=self.dropout,
                emb_dim=self.emb_dim
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
                for tod, dow, feats, targs in train_loader:
                    tod = tod.to(self.device)
                    dow = dow.to(self.device)
                    feats = feats.to(self.device)
                    targs = targs.to(self.device)
                    
                    optimizer.zero_grad()
                    preds = model(tod, dow, feats)
                    loss = criterion(preds, targs)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validate
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for tod, dow, feats, targs in test_loader:
                        tod = tod.to(self.device)
                        dow = dow.to(self.device)
                        feats = feats.to(self.device)
                        targs = targs.to(self.device)
                        
                        preds = model(tod, dow, feats)
                        loss = criterion(preds, targs)
                        val_loss += loss.item()
                
                val_loss /= len(test_loader)
                
                if (epoch + 1) % 5 == 0:
                    print(f"    Epoch {epoch+1}/{n_epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            
            print(f"  ✓ Best validation loss: {best_val_loss:.6f}")
            
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
        
        return cv_results, avg_val_loss
    
    def train_final_model(self, n_epochs: int = 25, batch_size: int = 256, lr: float = 3e-4):
        """
        Train final model on all buildings.
        
        Args:
            n_epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
        
        Returns:
            Trained model and scaler
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
            tod, dow, feats, targs = self.prepare_features_and_targets(building_obs)
            all_data.append((tod, dow, feats, targs))
        
        # Concatenate all data
        all_tod = np.concatenate([d[0] for d in all_data])
        all_dow = np.concatenate([d[1] for d in all_data])
        all_feats = np.concatenate([d[2] for d in all_data])
        all_targs = np.concatenate([d[3] for d in all_data])
        
        # Normalize features
        scaler = StandardScaler()
        all_feats = scaler.fit_transform(all_feats)
        
        # Create dataset
        dataset = TensorDataset(
            torch.tensor(all_tod, dtype=torch.long),
            torch.tensor(all_dow, dtype=torch.long),
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
        model = DemandModel(
            n_features=n_features,
            n_targets=self.n_targets,
            n_hidden=self.n_hidden,
            dropout=self.dropout,
            emb_dim=self.emb_dim
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
            for tod, dow, feats, targs in train_loader:
                tod = tod.to(self.device)
                dow = dow.to(self.device)
                feats = feats.to(self.device)
                targs = targs.to(self.device)
                
                optimizer.zero_grad()
                preds = model(tod, dow, feats)
                loss = criterion(preds, targs)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for tod, dow, feats, targs in val_loader:
                    tod = tod.to(self.device)
                    dow = dow.to(self.device)
                    feats = feats.to(self.device)
                    targs = targs.to(self.device)
                    
                    preds = model(tod, dow, feats)
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
        
        print(f"\n✓ Final model trained. Best validation loss: {best_val_loss:.6f}")
        
        return model, scaler, best_val_loss
    
    def save_model(self, model, scaler):
        """Save trained model and configuration."""
        # Save model weights
        model_path = self.output_dir / "demand_forecaster.pth"
        torch.save(model.state_dict(), model_path)
        print(f"  ✓ Saved model: {model_path}")
        
        # Save scaler
        import pickle
        scaler_path = self.output_dir / "demand_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"  ✓ Saved scaler: {scaler_path}")
        
        # Save configuration
        config = {
            'n_lags': self.n_lags,
            'n_targets': self.n_targets,
            'n_hidden': self.n_hidden,
            'dropout': self.dropout,
            'emb_dim': self.emb_dim,
            'device': str(self.device)
        }
        
        config_path = self.output_dir / "demand_forecaster_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  ✓ Saved config: {config_path}")


def main():
    print("="*80)
    print("  STEP 2: Train Demand Forecasting Model")
    print("="*80)
    print("\nThis script trains an MLP to forecast demand 10 steps ahead")
    print("Expected runtime: ~20 minutes\n")
    
    start_time = time.time()
    
    try:
        # Initialize trainer
        trainer = DemandForecasterTrainer(
            observations_path="data/external/observations.csv",
            output_dir="data/models",
            n_lags=10,
            n_targets=10,
            n_hidden=256,
            dropout=0.1,
            emb_dim=8
        )
        
        # Cross-validation
        print("\nPhase 1: Leave-One-Building-Out Cross-Validation")
        print("="*80)
        cv_results, avg_cv_loss = trainer.train_with_cv(
            n_epochs=25,
            batch_size=256,
            lr=3e-4
        )
        
        # Train final model
        print("\n\nPhase 2: Train Final Model on All Buildings")
        print("="*80)
        model, scaler, final_loss = trainer.train_final_model(
            n_epochs=25,
            batch_size=256,
            lr=3e-4
        )
        
        # Save model
        print("\nSaving model...")
        trainer.save_model(model, scaler)
        
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
        
        stats_path = Path("data/models") / "demand_training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  ✓ Saved stats: {stats_path}")
        
        # Summary
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("  DEMAND MODEL TRAINING COMPLETE")
        print("="*80)
        print(f"\nElapsed time: {elapsed_time/60:.1f} minutes")
        print(f"\nCross-Validation Results:")
        for r in cv_results:
            print(f"  Building {r['test_building']}: {r['val_loss']:.6f}")
        print(f"  Average: {avg_cv_loss:.6f}")
        print(f"\nFinal Model Validation Loss: {final_loss:.6f}")
        
        print("\n" + "="*80)
        print("  NEXT STEP: Run 3_create_solar_model.py")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())