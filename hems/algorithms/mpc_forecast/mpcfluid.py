#=========================================
#hems/algorithms/mpc_forecast/mpcfluid.py
#=========================================


import numpy as np
from scipy.optimize import linprog

class MPCFluid:
   
    
    def __init__(self):
        self._battery_capacity = 6.4
        self._bounds = [(-1.0,1.0)]*24 + [(0.0,1.0)]*24 + [(0.0,None)]*73
        self._A_eq = []
        
        # Energy balance: -Xi*6.4 + N+i - N-i = Ci-Gi
        for i in range(24):
            row = [0.0]*121
            row[i] = -self._battery_capacity
            row[i+48] = 1.0
            row[i+72] = -1.0
            self._A_eq.append(row)
        
        # Battery dynamics: -Xi + S{i+1} - Si = 0
        row = [0.0]*121
        row[0] = -1.0
        row[24] = 1.0
        self._A_eq.append(row)
        
        for i in range(1,24):
            row = [0.0]*121
            row[i] = -1.0
            row[i+24] = 1.0
            row[i+23] = -1.0
            self._A_eq.append(row)
        
        self._A_ub = []
        
        # Ramp constraints
        row_p = [0.0]*121
        row_p[48] = 1.0
        row_p[72] = -1.0
        row_p[96] = -1.0
        self._A_ub.append(row_p)
        
        row_n = [0.0]*121
        row_n[48] = -1.0
        row_n[72] = 1.0
        row_n[96] = -1.0
        self._A_ub.append(row_n)
        
        for i in range(1,24):
            row_p = [0.0]*121
            row_p[47+i] = 1.0
            row_p[48+i] = -1.0
            row_p[71+i] = -1.0
            row_p[72+i] = 1.0
            row_p[96+i] = -1.0
            self._A_ub.append(row_p)
            
            row_n = [0.0]*121
            row_n[47+i] = -1.0
            row_n[48+i] = 1.0
            row_n[71+i] = 1.0
            row_n[72+i] = -1.0
            row_n[96+i] = -1.0
            self._A_ub.append(row_n)
        
        # Load factor: N+i - L <= max
        for i in range(24):
            row = [0.0]*121
            row[48+i] = 1.0
            row[120] = -1.0
            self._A_ub.append(row)

    def forecast(self, price, carbon, consumption, generation, battery, 
                 net_consumption, error, max_net_consumption, load_change_weight):
       
        try:
            # Ensure inputs are numpy arrays
            price = np.asarray(price, dtype=np.float64).flatten()
            carbon = np.asarray(carbon, dtype=np.float64).flatten()
            consumption = np.asarray(consumption, dtype=np.float64).flatten()
            generation = np.asarray(generation, dtype=np.float64).flatten()
            
            # Validate shapes
            if len(price) != 24 or len(carbon) != 24 or len(consumption) != 24 or len(generation) != 24:
                print(f'[MPCFluid] Shape error: price={len(price)}, carbon={len(carbon)}, '
                      f'consumption={len(consumption)}, generation={len(generation)}')
                return 0.0
            
            # Check for invalid values
            if (not np.all(np.isfinite(price)) or not np.all(np.isfinite(carbon)) or 
                not np.all(np.isfinite(consumption)) or not np.all(np.isfinite(generation))):
                print(f'[MPCFluid] Invalid input values detected:')
                print(f'  price: min={np.min(price) if np.all(np.isfinite(price)) else "NaN/Inf"}, '
                      f'max={np.max(price) if np.all(np.isfinite(price)) else "NaN/Inf"}, '
                      f'has_nan={np.any(np.isnan(price))}, has_inf={np.any(np.isinf(price))}')
                print(f'  carbon: min={np.min(carbon) if np.all(np.isfinite(carbon)) else "NaN/Inf"}, '
                      f'max={np.max(carbon) if np.all(np.isfinite(carbon)) else "NaN/Inf"}, '
                      f'has_nan={np.any(np.isnan(carbon))}, has_inf={np.any(np.isinf(carbon))}')
                print(f'  consumption: min={np.min(consumption) if np.all(np.isfinite(consumption)) else "NaN/Inf"}, '
                      f'max={np.max(consumption) if np.all(np.isfinite(consumption)) else "NaN/Inf"}, '
                      f'has_nan={np.any(np.isnan(consumption))}, has_inf={np.any(np.isinf(consumption))}')
                print(f'  generation: min={np.min(generation) if np.all(np.isfinite(generation)) else "NaN/Inf"}, '
                      f'max={np.max(generation) if np.all(np.isfinite(generation)) else "NaN/Inf"}, '
                      f'has_nan={np.any(np.isnan(generation))}, has_inf={np.any(np.isinf(generation))}')
                return 0.0
            
            # Calculate net demand
            bi = consumption - generation
            
            # CRITICAL FIX: Robust ramp calculation
            ramp_no_battery = np.sum(np.abs(bi[0:23] - bi[1:24])) + np.abs(bi[0] - net_consumption)
            if not np.isfinite(ramp_no_battery) or ramp_no_battery < 1e-6:
                ramp_no_battery = 1.0  # Safe default
            
            # CRITICAL FIX: Robust total calculations with clipping
            bi_positive = np.clip(bi, 0, None)
            total_price = np.sum(price * bi_positive)
            total_carbon = np.sum(carbon * bi_positive)
            
            # CRITICAL FIX: Prevent division by zero
            eps = 1e-6
            if total_price < eps:
                total_price = eps
            if total_carbon < eps:
                total_carbon = eps
            
            # Calculate objective weights
            if total_carbon > eps and total_price > eps:
                obji = price / total_price + carbon / total_carbon
            elif total_price > eps:
                obji = price / total_price
            elif total_carbon > eps:
                obji = carbon / total_carbon
            else:
                obji = np.ones(24) / 24.0  # Uniform weights
            
            # CRITICAL FIX: Validate obji before using
            if not np.all(np.isfinite(obji)):
                print(f'[MPCFluid] Invalid obji detected, using uniform weights')
                obji = np.ones(24) / 24.0
            
            # CRITICAL FIX: Safe load_change_weight
            if not np.isfinite(load_change_weight) or load_change_weight < 0:
                load_change_weight = 0.1
            load_change_weight = np.clip(load_change_weight, 0.0, 10.0)
            
            # Construct objective vector with safety checks
            obj = ([0.0]*48 + 
                   [float(obji[i])*(46-i)/46 for i in range(24)] +
                   [load_change_weight*0.5/12*(46-i)/46 for i in range(24)] +
                   [0.5/float(ramp_no_battery)*(46-i)/46 for i in range(24)] +
                   [0.5*730/24*load_change_weight])
            
            # CRITICAL FIX: Final validation of objective
            obj = np.asarray(obj, dtype=np.float64)
            if not np.all(np.isfinite(obj)):
                print(f'[MPCFluid] Invalid objective function, using fallback')
                # Fallback: simple cost minimization
                obj = ([0.0]*48 + 
                       [1.0/24 for _ in range(24)] +  # Minimize positive net
                       [0.0]*24 +  # No penalty on negative net
                       [0.0]*24 +  # No ramping penalty
                       [0.0])  # No load factor penalty
            
            # CRITICAL FIX: Validate battery SOC
            battery = float(np.clip(battery, 0.0, 1.0))
            if not np.isfinite(battery):
                battery = 0.5
            
            # CRITICAL FIX: Validate net_consumption
            if not np.isfinite(net_consumption):
                net_consumption = 0.0
            
            # CRITICAL FIX: Validate max_net_consumption
            if not np.isfinite(max_net_consumption) or max_net_consumption < 0.1:
                max_net_consumption = 2.0
            
            # Construct constraint RHS
            b_eq = [float(bi[i]) for i in range(24)] + [float(battery)] + [0.0]*23
            b_ub = [net_consumption, -net_consumption] + [0.0]*46 + [max_net_consumption]*24
            
            # Final validation
            b_eq = np.asarray(b_eq, dtype=np.float64)
            b_ub = np.asarray(b_ub, dtype=np.float64)
            
            if not np.all(np.isfinite(b_eq)) or not np.all(np.isfinite(b_ub)):
                print(f'[MPCFluid] Invalid constraint values')
                return 0.0
            
            # Solve LP
            res = linprog(obj, A_ub=self._A_ub, b_ub=b_ub, A_eq=self._A_eq, 
                         b_eq=b_eq, bounds=self._bounds, method='highs-ds')
            
            if res.success and hasattr(res, 'x') and res.x is not None:
                action = float(np.clip(res.x[0], -1.0, 1.0))
                if not np.isfinite(action):
                    print(f'[MPCFluid] Action is not finite: {action}')
                    return 0.0
                return action
            else:
                if hasattr(res, 'message'):
                    print(f'[MPCFluid] Optimization failed: {res.message}')
                else:
                    print(f'[MPCFluid] Optimization failed (no message)')
                return 0.0
                
        except Exception as e:
            print(f'[MPCFluid] Exception: {e}')
            import traceback
            traceback.print_exc()
            return 0.0