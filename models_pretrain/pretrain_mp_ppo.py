#====================================
# models_pretrain/pretrain_mp_ppo.py
#====================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from pathlib import Path
from collections import deque
from hems.algorithms.mp_ppo import MPPPO, create_temporal_features, create_future_temporal_features, student_t_nll
from datetime import datetime
import time

def extract_net_load(obs, building_idx=0, obs_indices=None):
    """Extract net load (L̂_t = L_t - E^pv_t) from observation."""
    try:
        obs_flat = np.array(obs[building_idx], dtype=np.float32) if isinstance(obs, list) else obs
        
        if obs_indices is None:
            obs_indices = {'load': 10, 'pv': 11}
        
        load_idx = obs_indices['load']
        pv_idx = obs_indices['pv']
        
        if len(obs_flat) > max(load_idx, pv_idx):
            load = obs_flat[load_idx]
            solar = obs_flat[pv_idx]
            net_load = load - solar
            return float(net_load)
        else:
            return 0.0
    except Exception as e:
        print(f"Warning: Error extracting net load: {e}")
        return 0.0

def auto_detect_observation_indices(env):
    """Auto-detect observation indices from environment metadata."""
    if hasattr(env, 'observation_names') and len(env.observation_names) > 0:
        obs_names = env.observation_names[0]
        indices = {'load': None, 'pv': None}
        
        for i, name in enumerate(obs_names):
            name_lower = name.lower()
            if 'non_shiftable_load' in name_lower or 'non shiftable load' in name_lower:
                indices['load'] = i
            elif 'solar_gen' in name_lower or 'solar generation' in name_lower:
                indices['pv'] = i
        
        if indices['load'] is not None and indices['pv'] is not None:
            print(f"Auto-detected indices: load={indices['load']}, pv={indices['pv']}")
            return indices
    
    print("Using default indices: load=10, pv=11")
    return {'load': 10, 'pv': 11}

def create_pretraining_dataset(env, num_episodes=100, ctx=48, horizon=24, obs_indices=None):
    """Generate pretraining data from environment interactions."""
    dataset = []
    start_time = time.time()
    
    if obs_indices is None:
        obs_indices = auto_detect_observation_indices(env)
    
    print(f"\n{'='*60}")
    print(f"DATA COLLECTION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"Target episodes: {num_episodes}")
    print(f"Buildings: {len(env.buildings)}")
    print(f"Context window: {ctx}, Horizon: {horizon}")
    
    for ep in range(num_episodes):
        ep_start = time.time()
        obs = env.reset()
        done = False
        timestep = 0
        ep_samples = 0
        
        episode_net_loads = []
        episode_timesteps = []
        
        while not done:
            actions = [[float(np.random.uniform(-1, 1))] for _ in range(len(env.buildings))]
            next_obs, reward, done, truncated, info = env.step(actions)
            
            net_load = extract_net_load(obs, building_idx=0, obs_indices=obs_indices)
            episode_net_loads.append(net_load)
            episode_timesteps.append(timestep)
            
            obs = next_obs
            timestep += 1
            
            if done or truncated:
                break
        
        for t in range(len(episode_net_loads)):
            hist_consumption = np.zeros(ctx, dtype=np.float32)
            for i in range(ctx):
                hist_idx = t - (ctx - 1 - i)
                if hist_idx >= 0 and hist_idx < len(episode_net_loads):
                    hist_consumption[i] = episode_net_loads[hist_idx]
            
            hist = create_temporal_features(
                timestep=episode_timesteps[t],
                ctx=ctx,
                consumption_history=hist_consumption
            )
            
            tgt = create_future_temporal_features(
                timestep=episode_timesteps[t],
                horizon=horizon
            )
            
            y_target = np.zeros(horizon, dtype=np.float32)
            for h in range(horizon):
                future_idx = t + h + 1
                if future_idx < len(episode_net_loads):
                    y_target[h] = episode_net_loads[future_idx]
                else:
                    y_target[h] = episode_net_loads[-1] if episode_net_loads else 0.0
            
            hist = np.asarray(hist, dtype=np.float32)
            tgt = np.asarray(tgt, dtype=np.float32)
            y_target = np.asarray(y_target, dtype=np.float32)
            
            dataset.append((hist, tgt, y_target))
            ep_samples += 1
        
        ep_time = time.time() - ep_start
        progress = (ep + 1) / num_episodes * 100
        
        if (ep + 1) % 10 == 0 or ep == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (ep + 1)) * (num_episodes - ep - 1)
            print(f"Episode {ep + 1:3d}/{num_episodes} | "
                  f"Progress: {progress:5.1f}% | "
                  f"Samples: {ep_samples:4d} | "
                  f"Total: {len(dataset):6d} | "
                  f"Time: {ep_time:.2f}s | "
                  f"ETA: {eta/60:.1f}min")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DATA COLLECTION COMPLETED")
    print(f"Total samples: {len(dataset)}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Avg samples/episode: {len(dataset)/num_episodes:.1f}")
    
    all_targets = np.array([y for _, _, y in dataset])
    print(f"\nNet Load Statistics:")
    print(f"  Mean: {all_targets.mean():.4f}")
    print(f"  Std:  {all_targets.std():.4f}")
    print(f"  Min:  {all_targets.min():.4f}")
    print(f"  Max:  {all_targets.max():.4f}")
    print(f"{'='*60}\n")
    
    return dataset, obs_indices

def batch_generator(dataset, batch_size=64, shuffle=True):
    """Generate batches for training with optional shuffling."""
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)
    
    for start in range(0, len(dataset), batch_size):
        batch_idx = indices[start:start + batch_size]
        hist_batch = np.stack([dataset[i][0] for i in batch_idx])
        tgt_batch = np.stack([dataset[i][1] for i in batch_idx])
        y_batch = np.stack([dataset[i][2] for i in batch_idx])
        yield hist_batch, tgt_batch, y_batch

def validate_dataset(dataset, num_samples=3):
    """Validate dataset samples for debugging."""
    print(f"\n{'='*60}")
    print(f"DATASET VALIDATION")
    print(f"{'='*60}")
    print(f"Total samples: {len(dataset)}")
    
    for i in range(min(num_samples, len(dataset))):
        hist, tgt, y = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"  Historical features shape: {hist.shape}")
        print(f"  Target queries shape: {tgt.shape}")
        print(f"  Future targets shape: {y.shape}")
        print(f"  Historical consumption (last 5): {hist[-5:, -1]}")
        print(f"  Future net loads (first 5): {y[:5]}")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    from citylearn.citylearn import CityLearnEnv
    
    print(f"\n{'='*60}")
    print(f"MP-PPO PREDICTOR PRETRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    print("Creating CityLearn environment...")
    env = CityLearnEnv(
        schema='citylearn_challenge_2022_phase_all',
        central_agent=False
    )
    print(f"Environment created: {len(env.buildings)} buildings")
    print(f"Observation space dims: {[s.shape[0] for s in env.observation_space]}\n")
    
    config = {
        'horizon': 24,
        'ctx': 48,
        'in_dim': 5,
        'd_model': 32,
        'nhead': 2,
        'layers': 4,
        'ff': 32,
        'lr_pred': 3e-4,
        'buffer_steps': 8000,
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    obs_dim = sum([s.shape[0] for s in env.observation_space])
    act_dim = 31
    
    print(f"Creating MP-PPO agent...")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {act_dim}")
    print(f"  Device: {config['device']}")
    print(f"  Horizon: {config['horizon']}, Context: {config['ctx']}\n")
    
    agent = MPPPO(obs_dim, act_dim, config)
    
    dataset, obs_indices = create_pretraining_dataset(
        env, 
        num_episodes=100,
        ctx=config['ctx'],
        horizon=config['horizon']
    )
    
    validate_dataset(dataset, num_samples=3)
    
    print(f"{'='*60}")
    print(f"PREDICTOR TRAINING STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    epochs = 100
    batch_size = 64
    num_batches = len(dataset) // batch_size
    
    print(f"Training configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches per epoch: {num_batches}")
    print(f"  Total updates: {epochs * num_batches}\n")
    
    agent.predictor.train()
    train_start = time.time()
    
    best_loss = float('inf')
    losses = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        batch_count = 0
        
        for hist, tgt, y in batch_generator(dataset, batch_size, shuffle=True):
            hist_t = torch.as_tensor(hist, dtype=torch.float32, device=config['device'])
            tgt_t = torch.as_tensor(tgt, dtype=torch.float32, device=config['device'])
            y_t = torch.as_tensor(y, dtype=torch.float32, device=config['device'])
            
            nu, xi, sigma, _ = agent.predictor(hist_t, tgt_t)
            nll = student_t_nll(y_t, nu, xi, sigma)
            
            agent.opt_pred.zero_grad(set_to_none=True)
            nll.backward()
            torch.nn.utils.clip_grad_norm_(agent.predictor.parameters(), 1.0)
            agent.opt_pred.step()
            
            epoch_loss += nll.item()
            batch_count += 1
        
        avg_loss = epoch_loss / max(1, batch_count)
        losses.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            Path("models").mkdir(exist_ok=True)
            torch.save({
                'predictor': agent.predictor.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
                'obs_indices': obs_indices
            }, "models/mp_ppo_predictor_best.pt")
        
        epoch_time = time.time() - epoch_start
        progress = (epoch + 1) / epochs * 100
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - train_start
            eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
            print(f"Epoch {epoch + 1:3d}/{epochs} | "
                  f"Progress: {progress:5.1f}% | "
                  f"NLL Loss: {avg_loss:.4f} | "
                  f"Best: {best_loss:.4f} | "
                  f"Time: {epoch_time:.2f}s | "
                  f"ETA: {eta/60:.1f}min")
    
    total_train_time = time.time() - train_start
    print(f"\n{'='*60}")
    print(f"PREDICTOR TRAINING COMPLETED")
    print(f"Total training time: {total_train_time/60:.2f} minutes")
    print(f"Avg time per epoch: {total_train_time/epochs:.2f} seconds")
    print(f"Best NLL Loss: {best_loss:.4f}")
    print(f"Final NLL Loss: {losses[-1]:.4f}")
    print(f"{'='*60}\n")
    
    save_path = "models/mp_ppo_pretrained.pt"
    Path("models").mkdir(exist_ok=True)
    agent.save(save_path)
    
    torch.save({
        'predictor': agent.predictor.state_dict(),
        'config': config,
        'final_loss': losses[-1],
        'best_loss': best_loss,
        'epochs': epochs,
        'obs_indices': obs_indices
    }, "models/mp_ppo_predictor_only.pt")
    
    print(f"{'='*60}")
    print(f"PRETRAINING COMPLETE")
    print(f"Full model saved to: {save_path}")
    print(f"Predictor weights saved to: models/mp_ppo_predictor_only.pt")
    print(f"Best checkpoint saved to: models/mp_ppo_predictor_best.pt")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label='Training NLL Loss')
        plt.axhline(y=best_loss, color='r', linestyle='--', label=f'Best Loss: {best_loss:.4f}')
        plt.xlabel('Epoch')
        plt.ylabel('Negative Log-Likelihood')
        plt.title('MP-PPO Predictor Pretraining')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('models/pretraining_curve.png', dpi=150, bbox_inches='tight')
        print(f"Training curve saved to: models/pretraining_curve.png\n")
    except ImportError:
        print("matplotlib not available, skipping training curve plot\n")