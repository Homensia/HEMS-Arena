"""
Simple HEMS Logger - Compact version showing only key info every 50 steps
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path

class SimpleLogger:
    def __init__(self, log_file="agent_log.txt"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Clear previous log
        with open(self.log_file, 'w') as f:
            f.write(f"=== HEMS Agent Log Started: {datetime.now()} ===\n\n")
    
    def log_step(self, step, episode, observations, actions, rewards, env=None):
        """Log everything - show observations, actions, rewards, environment"""
        
        # Format all the data safely
        obs_str = self._format_data_safe(observations, "O")
        action_str = self._format_data_safe(actions, "A")
        reward_str = self._format_reward_safe(rewards)
        env_str = self._get_env_info_safe(env)
        
        # Create one line with everything
        log_line = f"E{episode:03d}S{step:04d} | {obs_str} → {action_str} → R={reward_str} | {env_str}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_line)
        
        # Print to console every 50 steps (much less frequent)
        if step % 50 == 0:
            print(f"E{episode:03d}S{step:04d} | {obs_str} → {action_str} → R={reward_str} | {env_str}")
    
    def _get_env_info_safe(self, env):
        """Get environment info with full error handling"""
        if env is None:
            return "No-env"
        
        info_parts = []
        
        try:
            # Try to get building info
            if hasattr(env, 'buildings') and env.buildings:
                building = env.buildings[0]
                
                # Battery state
                if hasattr(building, 'electrical_storage_soc'):
                    try:
                        soc = building.electrical_storage_soc
                        if isinstance(soc, (list, np.ndarray)):
                            soc_val = float(soc[0]) if len(soc) > 0 else 0.0
                        else:
                            soc_val = float(soc)
                        info_parts.append(f"Bat:{soc_val:.2f}")
                    except:
                        pass
                
                # Net consumption
                if hasattr(building, 'net_electricity_consumption'):
                    try:
                        net = building.net_electricity_consumption
                        if isinstance(net, (list, np.ndarray)):
                            net_val = float(net[0]) if len(net) > 0 else 0.0
                        else:
                            net_val = float(net)
                        info_parts.append(f"Net:{net_val:.1f}")
                    except:
                        pass
                
                # Solar generation
                if hasattr(building, 'solar_generation'):
                    try:
                        solar = building.solar_generation
                        if isinstance(solar, (list, np.ndarray)):
                            solar_val = float(solar[0]) if len(solar) > 0 else 0.0
                        else:
                            solar_val = float(solar)
                        if solar_val > 0.01:  # Only show if significant
                            info_parts.append(f"Sol:{solar_val:.1f}")
                    except:
                        pass
            
            # Time info
            if hasattr(env, 'time_step'):
                try:
                    step_val = env.time_step
                    if isinstance(step_val, (list, np.ndarray)):
                        step_num = int(step_val[0]) if len(step_val) > 0 else 0
                    else:
                        step_num = int(step_val)
                    hour = step_num % 24
                    info_parts.append(f"H:{hour}")
                except:
                    pass
            
        except Exception as e:
            info_parts.append(f"Err:{str(e)[:10]}")
        
        return " | ".join(info_parts) if info_parts else "No-data"
    
    def _format_reward_safe(self, rewards):
        """Safely format rewards"""
        try:
            if rewards is None:
                return "0.00"
            elif isinstance(rewards, (list, tuple, np.ndarray)):
                if len(rewards) == 0:
                    return "0.00"
                elif len(rewards) == 1:
                    return f"{float(rewards[0]):.2f}"
                else:
                    total = sum(float(r) for r in rewards)
                    return f"{total:.2f}"
            else:
                return f"{float(rewards):.2f}"
        except:
            return "ERR"
    
    def _format_data_safe(self, data, prefix):
        """Safely format observations or actions"""
        try:
            if data is None:
                return f"{prefix}=None"
            elif isinstance(data, (list, tuple, np.ndarray)):
                # Convert to flat list if nested
                flat_data = []
                self._flatten_data(data, flat_data)
                
                if len(flat_data) == 0:
                    return f"{prefix}=[]"
                elif len(flat_data) == 1:
                    return f"{prefix}=[{self._safe_float(flat_data[0]):.2f}]"
                elif len(flat_data) <= 5:
                    values = [f"{self._safe_float(x):.2f}" for x in flat_data]
                    return f"{prefix}=[{','.join(values)}]"
                else:
                    # Show first 3 and last 2 for longer arrays
                    first_vals = [f"{self._safe_float(x):.2f}" for x in flat_data[:3]]
                    last_vals = [f"{self._safe_float(x):.2f}" for x in flat_data[-2:]]
                    return f"{prefix}=[{','.join(first_vals)}..{','.join(last_vals)}]"
            else:
                return f"{prefix}={self._safe_float(data):.2f}"
        except Exception as e:
            return f"{prefix}=ERR"
    
    def _flatten_data(self, data, result):
        """Recursively flatten nested data structures"""
        if isinstance(data, (list, tuple, np.ndarray)):
            for item in data:
                self._flatten_data(item, result)
        else:
            result.append(data)
    
    def _safe_float(self, value):
        """Safely convert any value to float"""
        try:
            if isinstance(value, (list, np.ndarray)):
                if len(value) > 0:
                    return float(value[0])
                else:
                    return 0.0
            else:
                return float(value)
        except:
            return 0.0