#=================================================
#hems/algorithms/mpc_forecast/cufe_forecasters.py
#=================================================


"""
CUFE Forecasting Components
Implements power, solar, and carbon intensity forecasting for CUFE MPC algorithm.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
from typing import Optional, Tuple


class PowerBlender:
    """Linear regression forecaster for power consumption using historical patterns."""

    def __init__(self, data_file: str):
        self._first_index = 0
        self._X = np.load(data_file).astype(np.float64)
        self._Y = []
        self._model = None

    def reset(self):
        """Reset forecaster state for new episode."""
        self._Y = []
        self._model = None

    def get_first_index(self):
        return self._first_index

    def set_first_index(self, month: int, day: int, hour: int):
        """Find matching timestamp in training data."""
        self._first_index = 0
        max_search = min(8760, len(self._X))
        for idx in range(max_search):
            if (self._X[idx, 0] == month and
                self._X[idx, 1] == day and
                self._X[idx, 2] == hour):
                self._first_index = idx
                break
        else:
            self._first_index = hour % len(self._X)

    def add_observation(self, obs: float):
        """Add new observation to history."""
        self._Y.append(float(obs))

    def train(self):
        """Train linear model on recent 168 observations."""
        n = len(self._Y)
        if n < 2:
            return  # Prevent empty or single-sample fit

        try:
            begin = max(0, n - 168)
            if self._first_index == 0:
                # Fit using most recent 168 samples
                self._model = LinearRegression().fit(self._X[begin:n, 3:], self._Y[begin:n])
            else:
                # Wrap around yearly (8760-hour) dataset
                indexes = [(self._first_index + i) % len(self._X) for i in range(begin, n)]
                self._model = LinearRegression().fit(self._X[indexes, 3:], self._Y[begin:n])
        except Exception as e:
            print(f"[PowerBlender] Training failed: {e}")
            self._model = None


    def predict(self) -> np.ndarray:
        """Generate 24-hour forecast."""
        # Fallback: no model or insufficient history
        if self._model is None or len(self._Y) == 0:
            mean_val = np.mean(self._Y[-min(168, len(self._Y)):]) if len(self._Y) > 0 else 1.0
            return np.full((24, 1), mean_val, dtype=np.float64)

        try:
            indexes = [(len(self._Y) + self._first_index + i) % len(self._X) for i in range(24)]
            predictions = self._model.predict(self._X[indexes, 3:]).reshape((-1, 1))
            if not np.all(np.isfinite(predictions)):
                predictions = np.full((24, 1), 1.0, dtype=np.float64)
            # Power consumption must be positive
            predictions = np.maximum(predictions, 0.01)
            return predictions
        except Exception as e:
            print(f"[PowerBlender] Prediction failed: {e}")
            return np.full((24, 1), 1.0, dtype=np.float64)



class SolarBlender:
    """Linear regression forecaster for solar generation using historical patterns."""

    def __init__(self, data_file: str):
        self._first_index = 0
        self._X = np.load(data_file).astype(np.float64)
        self._Y = []
        self._model = None

    def reset(self):
        """Reset forecaster state for new episode."""
        self._Y = []
        self._model = None

    def get_first_index(self):
        return self._first_index

    def set_first_index(self, month: int, day: int, hour: int):
        """Find matching timestamp in training data."""
        self._first_index = 0
        max_search = min(8760, len(self._X))
        for idx in range(max_search):
            if (self._X[idx, 0] == month and
                self._X[idx, 1] == day and
                self._X[idx, 2] == hour):
                self._first_index = idx
                break
        else:
            self._first_index = hour % len(self._X)

    def add_observation(self, obs: float):
        """Add new observation to history."""
        self._Y.append(float(obs))

    def train(self):
        """Train linear model on recent 168 observations."""
        n = len(self._Y)
        if n < 2:
            return  # Prevent empty or single-sample fit

        try:
            begin = max(0, n - 168)
            if self._first_index == 0:
                # Fit using most recent 168 samples directly
                self._model = LinearRegression().fit(self._X[begin:n, 3:], self._Y[begin:n])
            else:
                # Wrap around in yearly (8760-hour) dataset
                indexes = [(self._first_index + i) % 8760 for i in range(begin, n)]
                self._model = LinearRegression().fit(self._X[indexes, 3:], self._Y[begin:n])
        except Exception as e:
            print(f"[SolarBlender] Training failed: {e}")
            self._model = None


    def predict(self) -> np.ndarray:
        """Generate 24-hour forecast."""
        # Fallback: no model or insufficient data
        if self._model is None or len(self._Y) == 0:
            # Return zeros for missing solar production
            return np.zeros((24, 1), dtype=np.float64)

        try:
            indexes = [(len(self._Y) + self._first_index + i) % len(self._X) for i in range(24)]
            predictions = self._model.predict(self._X[indexes, 3:]).reshape((-1, 1))
            # Replace invalid values with zeros
            if not np.all(np.isfinite(predictions)):
                predictions = np.zeros((24, 1), dtype=np.float64)
            # Ensure no negative solar generation
            predictions = np.maximum(predictions, 0.0)
            return predictions
        except Exception as e:
            print(f"[SolarBlender] Prediction failed: {e}")
            return np.zeros((24, 1), dtype=np.float64)


class CUFEPowerForecaster:
    """Power consumption forecaster combining LinearRegression blender with AR(168) model."""

    def __init__(self, lr_data_file: str, ar_beta_file: str):
        self._metaX = []
        self._metaY = []
        self._metaCoef = []

        self._historical_non_shiftable_loads = np.zeros((24, 1), dtype=np.float64)
        self._historical_non_shiftable_loads_large = np.zeros((7 * 24, 1), dtype=np.float64)
        self._historical_non_shiftable_loads_large_hours = np.full((7 * 24, 1), np.nan)

        self._blender = PowerBlender(lr_data_file)
        self.beta = np.load(ar_beta_file).astype(np.float64)

        self._correction = np.full((24, 1), 0.0)
        self._correctionCoeff = np.array([
            0.0246732260178073, 0.0148629156755033, 0.00598375520410082, 0.00288682545851409,
            0.00406717974523319, 0.00477531622729592, -0.000871411541765286, -0.00800443696796399,
            -0.00906473496149969, -0.0105722876839394, -0.00515319703336365, -0.00358145580057755,
            -0.00356015792289832, -0.00609082035130820, -0.00695579231942541, -0.00157742050484078,
            -0.000477118583228363, 0.00175314819317429, -0.00145796987476758, -0.00701183558871695,
            -0.00727965443504950, -0.00725708883062715, 0.00174657511614960, 0.00931902642694797
        ]).reshape((-1, 1))

        self._loss = [0.0, 0.0]
        self._last_forecast = [0.0, 0.0]
        self._first = True
        self._step = 0
        self._prev_estimation = np.zeros((24, 1), dtype=np.float64)
        self._eps = 1e-9

    def reset(self):
        """Reset forecaster state for new episode."""
        self._metaX = []
        self._metaY = []
        self._metaCoef = []
        self._historical_non_shiftable_loads = np.zeros((24, 1), dtype=np.float64)
        self._historical_non_shiftable_loads_large = np.zeros((7 * 24, 1), dtype=np.float64)
        self._historical_non_shiftable_loads_large_hours = np.full((7 * 24, 1), np.nan)
        self._correction = np.full((24, 1), 0.0)
        self._blender.reset()
        self._loss = [0.0, 0.0]
        self._last_forecast = [0.0, 0.0]
        self._first = True
        self._step = 0
        self._prev_estimation = np.zeros((24, 1), dtype=np.float64)

    def update_power(self, power: float, month: int, day: int, hour: int):
        """Update forecaster with new power observation."""
        if self._step == 0:
            self._blender.set_first_index(month, day, hour)

        power = float(np.clip(power, 0.0, 1e6))
        self._blender.add_observation(power)

        self._historical_non_shiftable_loads[:-1] = self._historical_non_shiftable_loads[1:]
        self._historical_non_shiftable_loads[-1] = power

        self._historical_non_shiftable_loads_large[:-1] = self._historical_non_shiftable_loads_large[1:]
        self._historical_non_shiftable_loads_large[-1] = power

        hour_idx = int(hour % 24)

        end_idx = min(121 + hour_idx, len(self._historical_non_shiftable_loads_large_hours))
        src_start = 24 + hour_idx
        src_end = min(145 + hour_idx, len(self._historical_non_shiftable_loads_large_hours))
        self._historical_non_shiftable_loads_large_hours[hour_idx:end_idx:24] = \
            self._historical_non_shiftable_loads_large_hours[src_start:src_end:24]

        self._historical_non_shiftable_loads_large_hours[144 + hour_idx] = power


        self._step += 1

        if not self._first:
            l0 = abs(power - self._last_forecast[0])
            l1 = abs(power - self._last_forecast[1])
            self._loss[0] += l0
            self._loss[1] += l1

            self._metaX.append([self._last_forecast[0], self._last_forecast[1]])
            self._metaY.append(power)

            self._correction[0:23] = self._correction[1:24]
            self._correction[23] = 0.0
        else:
            for i in range(7):
                start, end = i * 24, (i + 1) * 24
                self._historical_non_shiftable_loads_large[start:end] = self._historical_non_shiftable_loads
                self._historical_non_shiftable_loads_large_hours[start:end] = \
                    self._historical_non_shiftable_loads_large_hours[-24:]

    def forecast(self, weather=None, hour: int = 0) -> Tuple[np.ndarray, float]:
        """Generate 24-hour ahead power forecast."""
        if self._first:
            self._blender.train()

        if self._step % 168 == 0 and self._step > 0:
            self._blender.train()
            if len(self._metaX) >= 10:
                try:
                    model = LinearRegression(fit_intercept=False).fit(
                        np.array(self._metaX), np.array(self._metaY)
                    )
                    self._metaCoef = model.coef_
                except Exception as e:
                    print(f"[PowerForecaster] Meta-learning failed: {e}")

        if self._step % (4 * 168) == 0 and self._step > 0:
            if len(self._metaX) >= 10:
                try:
                    model = LinearRegression(fit_intercept=False).fit(
                        np.array(self._metaX), np.array(self._metaY)
                    )
                    self._metaCoef = model.coef_
                except Exception as e:
                    print(f"[PowerForecaster] Meta-learning failed: {e}")

        self._first = False

        if self._step >= 168 and len(self._metaCoef) >= 2:
            w0 = float(self._metaCoef[0])
            w1 = float(self._metaCoef[1])
        else:
            m = min(self._loss[0], self._loss[1])
            exp0 = np.exp(-self._loss[0] + m + self._eps)
            exp1 = np.exp(-self._loss[1] + m + self._eps)
            total = exp0 + exp1 + self._eps
            w0 = float(exp0 / total)
            w1 = float(exp1 / total)

        f0 = self._blender.predict()
        if not self._validate_forecast(f0):
            f0 = self._get_fallback_forecast()

        try:
            f1 = np.dot(self._historical_non_shiftable_loads_large.T, self.beta).T
            f1 = f1 + self._correction
        except Exception as e:
            print(f"[PowerForecaster] AR(168) failed: {e}")
            f1 = f0.copy()
        f1 = f1 + self._correction

        if not self._validate_forecast(f1):
            f1 = f0.copy()

        self._last_forecast[0] = float(f0[0, 0] if f0.ndim > 1 else f0[0])
        self._last_forecast[1] = float(f1[0, 0] if f1.ndim > 1 else f1[0])
        self._prev_estimation = f1.copy()

        if self._loss[0] < 0.0001 and w0 > 0.9999:
            error = self._loss[0] / max(1, self._step)
            return f0, error

        f = w0 * f0 + w1 * f1
        if not self._validate_forecast(f):
            f = f0.copy()

        error = (w0 * self._loss[0] + w1 * self._loss[1]) / max(1, self._step)
        return f, error

    def _validate_forecast(self, forecast: np.ndarray) -> bool:
        if forecast is None or not isinstance(forecast, np.ndarray):
            return False
        if forecast.size == 0 or not np.all(np.isfinite(forecast)):
            return False
        if forecast.shape[0] != 24:
            return False
        return True

    def _get_fallback_forecast(self) -> np.ndarray:
        if self._step > 24:
            recent_values = self._historical_non_shiftable_loads[-24:]
            finite_values = recent_values[np.isfinite(recent_values)]
            if finite_values.size > 0:
                recent_avg = np.mean(finite_values)
                if recent_avg > 0:
                    return np.full((24, 1), recent_avg, dtype=np.float64)
        return np.full((24, 1), 1.0, dtype=np.float64)



class CUFESolarForecaster:
    """Solar generation forecaster combining LinearRegression blender with AR(168) and rank envelope."""
    
    def __init__(self, lr_data_file: str, ar_beta_file: str, rank_file: Optional[str] = None):
        self._metaX = []
        self._metaY = []
        self._metaCoef = []
        
        self._historical_solar = np.zeros((24, 1), dtype=np.float64)
        self._historical_solar_large = np.zeros((7*24, 1), dtype=np.float64)
        
        self._blender = SolarBlender(lr_data_file)
        
        self._ranker = None
        if rank_file and Path(rank_file).exists():
            try:
                self._ranker = SolarRank(rank_file)
            except Exception as e:
                print(f"[SolarForecaster] SolarRank failed to load: {e}")
        
        self.beta = np.load(ar_beta_file).astype(np.float64)
        
        self._loss = [0.0, 0.0]
        self._last_forecast = [0.0, 0.0]
        self._first = True
        self._step = 0
        self._eps = 1e-9
    
    def reset(self):
        """Reset forecaster state for new episode."""
        self._metaX = []
        self._metaY = []
        self._metaCoef = []
        self._historical_solar = np.zeros((24, 1), dtype=np.float64)
        self._historical_solar_large = np.zeros((7*24, 1), dtype=np.float64)
        self._blender.reset()
        if self._ranker:
            self._ranker.reset()
        self._loss = [0.0, 0.0]
        self._last_forecast = [0.0, 0.0]
        self._first = True
        self._step = 0
    
    def update_solar(self, solar: float, month: int, day: int, hour: int):
        """Update forecaster with new solar observation."""
        if self._step == 0:
            self._blender.set_first_index(month, day, hour)
            if self._ranker:
                self._ranker.set_index(self._blender.get_first_index())
        
        solar = float(np.clip(solar, 0.0, 1e6))
        
        self._blender.add_observation(solar)
        if self._ranker:
            self._ranker.add_observation(solar)
        
        self._historical_solar[:-1] = self._historical_solar[1:]
        self._historical_solar[-1] = solar
        
        self._historical_solar_large[:-1] = self._historical_solar_large[1:]
        self._historical_solar_large[-1] = solar
        
        self._step += 1
        
        if not self._first:
            l0 = abs(solar - self._last_forecast[0])
            l1 = abs(solar - self._last_forecast[1])
            self._loss[0] += l0
            self._loss[1] += l1
            
            self._metaX.append([self._last_forecast[0], self._last_forecast[1]])
            self._metaY.append(solar)
        else:
            for i in range(7):
                start, end = i*24, (i+1)*24
                self._historical_solar_large[start:end] = self._historical_solar
    
    def forecast(self, weather=None) -> Tuple[np.ndarray, float]:
        """Generate 24-hour ahead solar forecast."""
        if self._first:
            self._blender.train()
        
        if self._step % 168 == 0 and self._step > 0:
            self._blender.train()
            if len(self._metaX) >= 10:
                try:
                    model = LinearRegression(fit_intercept=False).fit(
                        np.array(self._metaX), np.array(self._metaY)
                    )
                    self._metaCoef = model.coef_
                except Exception as e:
                    print(f"[SolarForecaster] Meta-learning failed: {e}")
        
        if self._step % (4*168) == 0 and self._step > 0:
            if len(self._metaX) >= 10:
                try:
                    model = LinearRegression(fit_intercept=False).fit(
                        np.array(self._metaX), np.array(self._metaY)
                    )
                    self._metaCoef = model.coef_
                except Exception as e:
                    print(f"[SolarForecaster] Meta-learning failed: {e}")
        
        self._first = False
        
        if self._step >= 168 and len(self._metaCoef) >= 2:
            w0 = float(self._metaCoef[0])
            w1 = float(self._metaCoef[1])
        else:
            m = min(self._loss[0], self._loss[1])
            exp0 = np.exp(-self._loss[0] + m + self._eps)
            exp1 = np.exp(-self._loss[1] + m + self._eps)
            total = exp0 + exp1 + self._eps
            w0 = float(exp0 / total)
            w1 = float(exp1 / total)
        
        solar_mask = None
        if weather is not None:
            try:
                solar_irr = weather.get_next_24_direct_solar_irradiance()
                if isinstance(solar_irr, np.ndarray) and len(solar_irr) == 24:
                    solar_mask = 1.0 * (solar_irr > 0)
            except Exception:
                pass

        if self._loss[0] < 0.0001 and w0 > 0.9999:
            f0 = self._blender.predict()
            f1 = np.dot(self._historical_solar_large.T, self.beta).T
            
            if solar_mask is not None:
                f0 = f0 * solar_mask
                f1 = f1 * solar_mask
            
            self._last_forecast[0] = float(f0[0])
            self._last_forecast[1] = float(f1[0])
            return f0, self._loss[0]

        f0 = self._blender.predict()
        f1 = np.dot(self._historical_solar_large.T, self.beta).T

        if solar_mask is not None:
            f0 = f0 * solar_mask
            f1 = f1 * solar_mask

        self._last_forecast[0] = float(f0[0])
        self._last_forecast[1] = float(f1[0])

        f = w0 * f0 + w1 * f1
        f = f * (1.0 * (f > 0))
        
        if not self._validate_forecast(f):
            f = np.zeros((24, 1), dtype=np.float64)
        
        # Apply historical min/max envelope
        if self._ranker:
            try:
                Mins, Maxs = self._ranker.get_ranges()
                for i in range(24):
                    f[i] = np.clip(f[i], Mins[i], Maxs[i])
            except Exception:
                pass
        
        error = (w0 * self._loss[0] + w1 * self._loss[1]) / max(1, self._step)
        return f, error
    
    def _validate_forecast(self, forecast: np.ndarray) -> bool:
        """Check if forecast is valid."""
        if forecast is None or not isinstance(forecast, np.ndarray):
            return False
        if forecast.size == 0 or not np.all(np.isfinite(forecast)):
            return False
        if forecast.shape[0] != 24:
            return False
        return True


class CUFECarbonForecaster:
    """Carbon intensity forecaster using neural network model."""
    
    def __init__(self, model_file: str, beta_file: Optional[str] = None):
        import joblib
        self._model = joblib.load(model_file)
        
        self.beta = np.array([[-0.22108, -0.61261, -0.74376, -0.76633, -0.76652, -0.73917, -0.69866, -0.64752, -0.58283,
                            -0.48703, -0.44199, -0.41432, -0.39143, -0.36283, -0.31182, -0.27815, -0.22186, -0.13391,
                            -0.068614, -0.034806, 0.0030711, 0.021417, 0.018681, -0.034829],
                            [0.25359, 0.54431, 0.32866, 0.22948, 0.2094, 0.17498, 0.15073, 0.12606, 0.096278,
                            0.035665, 0.072412, 0.081994, 0.080057, 0.065349, 0.026546, 0.034089, -0.0034307,
                            -0.06204, -0.059061, -0.0376, -0.054052, -0.041131, -0.020014, 0.043038],
                            [-0.01999, 0.087416, 0.32122, 0.092755, -0.010425, -0.022996, -0.044738, -0.052907,
                            -0.059814, -0.053852, -0.097368, -0.051532, -0.033899, -0.024781, -0.019371, -0.046632,
                            -0.022158, -0.027006, -0.061973, -0.047111, -0.010372, -0.018293, -0.0049859, 0.0034927],
                            [0.027747, 0.038782, 0.1554, 0.39028, 0.16115, 0.055355, 0.039192, 0.013043, -0.0014289,
                            -0.015736, -0.013271, -0.059203, -0.015427, 5.6771e-05, 0.0055699, 0.0082505, -0.02431,
                            -0.0068012, -0.016825, -0.054374, -0.042032, -0.0064, -0.013887, 0.004887],
                            [0.014121, 0.034186, 0.040316, 0.15524, 0.38896, 0.15957, 0.053878, 0.037868, 0.011102,
                            -0.0014382, -0.014767, -0.012103, -0.057728, -0.01328, 0.0035542, 0.0094449, 0.011995,
                            -0.01888, -0.00030989, -0.0099046, -0.046353, -0.033152, 0.0028664, -0.0035122],
                            [0.034963, 0.056062, 0.074578, 0.079063, 0.19208, 0.42397, 0.19276, 0.084803, 0.064464,
                            0.036198, 0.023039, 0.0086822, 0.010707, -0.035247, 0.0089487, 0.024748, 0.027499,
                            0.028433, -0.0039796, 0.013566, 0.0039425, -0.031992, -0.017953, 0.022631],
                            [-0.014081, 0.010441, 0.029134, 0.047592, 0.052555, 0.16665, 0.39993, 0.17051, 0.065046,
                            0.047309, 0.02021, 0.0079204, -0.0058476, -0.0030808, -0.047713, -0.0024494, 0.015325,
                            0.020452, 0.023285, -0.0080056, 0.010531, 0.0012106, -0.03501, -0.023338],
                            [0.090617, 0.096982, 0.11867, 0.13302, 0.14642, 0.14651, 0.25547, 0.48269, 0.24143,
                            0.13153, 0.11188, 0.081715, 0.067122, 0.052501, 0.05474, 0.0073728, 0.044018, 0.057087,
                            0.058216, 0.058981, 0.028029, 0.047942, 0.040518, 0.016445],
                            [-0.096822, -0.061252, -0.065588, -0.042305, -0.023574, -0.0034615, 0.0049545, 0.1237,
                            0.36745, 0.1389, 0.034954, 0.020724, -0.0047582, -0.016045, -0.025908, -0.018351,
                            -0.052704, -0.0036503, 0.018704, 0.024068, 0.02742, -0.0031076, 0.015141, -0.0077101],
                            [-0.010333, -0.068591, -0.015782, -0.015947, 0.0091521, 0.026558, 0.043721, 0.048714,
                            0.16398, 0.39836, 0.1652, 0.058908, 0.042138, 0.013829, -0.0025437, -0.015053, -0.011195,
                            -0.054113, -0.010731, 0.009474, 0.011225, 0.012107, -0.01914, 0.00076363],
                            [-0.0060503, -0.020045, -0.079113, -0.026288, -0.0262, -0.00066786, 0.01722, 0.034923,
                            0.041002, 0.15698, 0.39173, 0.15889, 0.052918, 0.036367, 0.0082796, -0.0077989,
                            -0.019388, -0.014726, -0.057045, -0.013393, 0.0069062, 0.0086821, 0.0094188, -0.022798],
                            [-0.0078318, -0.019565, -0.034951, -0.093997, -0.040847, -0.040157, -0.013845, 0.0049433,
                            0.024147, 0.031532, 0.14813, 0.38341, 0.15105, 0.045433, 0.029388, 0.0018071, -0.013057,
                            -0.023372, -0.017792, -0.059729, -0.0158, 0.0045927, 0.0062543, 0.0056468],
                            [0.023632, 0.030105, 0.021309, 0.0058002, -0.054189, -0.0027159, -0.0039834, 0.019935,
                            0.0347, 0.050734, 0.056605, 0.17196, 0.40622, 0.17294, 0.065977, 0.04858, 0.017724,
                            -0.00034251, -0.013182, -0.0089815, -0.051793, -0.0080764, 0.012863, 0.018286],
                            [-0.011212, 0.0063228, 0.011624, 0.0028942, -0.012149, -0.071372, -0.019081, -0.019386,
                            0.0064082, 0.022357, 0.039, 0.045389, 0.16123, 0.39587, 0.16299, 0.05656, 0.040739,
                            0.011225, -0.0057816, -0.018055, -0.013629, -0.056404, -0.012989, 0.0062213],
                            [-0.0050647, -0.019066, -0.0018426, 0.003707, -0.0047866, -0.019492, -0.078134, -0.025012,
                            -0.024623, 0.0022974, 0.018632, 0.035624, 0.042091, 0.15815, 0.39347, 0.16107, 0.054985,
                            0.039869, 0.010946, -0.0056676, -0.017386, -0.012857, -0.055623, -0.013013],
                            [0.12079, 0.14359, 0.12578, 0.13732, 0.13624, 0.12126, 0.099646, 0.032583, 0.070178,
                            0.064112, 0.088362, 0.10066, 0.11495, 0.1201, 0.23478, 0.46617, 0.2226, 0.11004,
                            0.089211, 0.056935, 0.040023, 0.029932, 0.03705, 0.010445],
                            [-0.086185, -0.02029, -0.009064, -0.026174, -0.011134, -0.0059759, -0.013156, -0.025161,
                            -0.077305, -0.026572, -0.026612, 0.0026438, 0.018857, 0.036729, 0.047662, 0.1678,
                            0.41096, 0.17982, 0.076873, 0.061081, 0.03263, 0.01669, 0.0049953, -0.0019087],
                            [-0.010514, -0.06921, 0.0091307, 0.023497, 0.0079348, 0.022179, 0.025283, 0.015517,
                            0.0015947, -0.057657, -0.010289, -0.011933, 0.01582, 0.029906, 0.04359, 0.05247, 0.17052,
                            0.40752, 0.17217, 0.06736, 0.048382, 0.017958, 0.001423, -0.0094979],
                            [-0.02025, -0.04064, -0.10078, -0.022109, -0.0068833, -0.021123, -0.0053859, -0.00037099,
                            -0.0070547, -0.018693, -0.076908, -0.028641, -0.029677, -0.0012612, 0.013847, 0.028592,
                            0.039887, 0.16017, 0.39906, 0.16487, 0.060802, 0.041917, 0.010996, -0.0085711],
                            [-0.006558, -0.016798, -0.03232, -0.090741, -0.01115, 0.0039091, -0.010661, 0.004501,
                            0.0092658, 0.00037882, -0.012397, -0.070867, -0.022824, -0.024759, 0.0020735, 0.016548,
                            0.030631, 0.039744, 0.15832, 0.39606, 0.16047, 0.055518, 0.036657, 0.0055683],
                            [-0.033294, -0.043717, -0.05141, -0.064817, -0.12117, -0.039962, -0.023173, -0.035763,
                            -0.0166, -0.01089, -0.019454, -0.031207, -0.088934, -0.04089, -0.04309, -0.015481,
                            0.001699, 0.016728, 0.026662, 0.14554, 0.38274, 0.1464, 0.040855, 0.017738],
                            [0.069977, 0.063673, 0.055419, 0.044944, 0.027852, -0.032767, 0.04312, 0.053522, 0.030587,
                            0.042742, 0.045479, 0.033553, 0.019294, -0.039897, 0.0057392, 0.00029391, 0.020451,
                            0.03137, 0.041593, 0.04933, 0.16703, 0.4046, 0.16927, 0.074026],
                            [-0.42099, -0.4665, -0.46699, -0.45675, -0.4446, -0.43822, -0.4729, -0.36545, -0.29878,
                            -0.29423, -0.27048, -0.25225, -0.25345, -0.26182, -0.3136, -0.25277, -0.21743, -0.17054,
                            -0.13698, -0.11414, -0.10356, 0.0095216, 0.23852, -0.054566],
                            [1.3344, 1.3455, 1.3088, 1.2578, 1.1993, 1.134, 1.0542, 0.91425, 0.83641, 0.80227,
                            0.76284, 0.73463, 0.71586, 0.69075, 0.6507, 0.54646, 0.46988, 0.40614, 0.37034, 0.35749,
                            0.36467, 0.38575, 0.52681, 0.94313]])
        
        self._historical_carbon_intensity = np.full((24, 1), 0.1565, dtype=np.float64)
        self._loss = [0.0, 0.0]
        self._last_forecast = [0.0, 0.0]
        self._first = True
        self._eps = 1e-9
    
    def reset(self):
        """Reset forecaster state for new episode."""
        self._historical_carbon_intensity = np.full((24, 1), 0.5, dtype=np.float64)
        self._loss = [0.0, 0.0]
        self._last_forecast = [0.0, 0.0]
        self._first = True
    
    def update_carbon_intensity(self, carbon: float):
        """Update forecaster with new carbon intensity observation."""
        carbon = float(np.clip(carbon, 0.0, 2.0))
        
        if not self._first:
            l0 = abs(carbon - self._last_forecast[0])
            l1 = abs(carbon - self._last_forecast[1])
            total = l0 + l1 + self._eps
            self._loss[0] += l0 / total
            self._loss[1] += l1 / total
        
        self._historical_carbon_intensity[:-1] = self._historical_carbon_intensity[1:]
        self._historical_carbon_intensity[-1] = carbon
        self._first = False
    
    def forecast(self) -> np.ndarray:
        """Generate 24-hour ahead carbon intensity forecast."""
        try:
            f0 = self._model.predict(self._historical_carbon_intensity[0:24].T).T
            
            if not np.all(np.isfinite(f0)) or len(f0) != 24:
                f0 = np.full((24, 1), 0.5, dtype=np.float64)
            
            if self.beta is not None:
                f1 = np.dot(self._historical_carbon_intensity[0:24].T, self.beta).T
                if not np.all(np.isfinite(f1)):
                    f1 = f0.copy()
            else:
                f1 = f0.copy()
            
            self._last_forecast[0] = float(f0[0] if f0.ndim == 1 else f0[0, 0])
            self._last_forecast[1] = float(f1[0] if f1.ndim == 1 else f1[0, 0])
            
            return f0
            
        except Exception as e:
            print(f"[CarbonForecaster] Forecast failed: {e}")
            return np.full((24, 1), 0.5, dtype=np.float64)


class SolarRank:
    """Tracks solar generation ranges for envelope constraints."""
    
    def __init__(self, rank_file: str):
        self._index = 0
        self._rank = np.load(rank_file).astype(np.int32)
        self._values = np.full(8760, -1.0, dtype=np.float64)
        self._initial_index = 0
    
    def reset(self):
        """Reset tracker state for new episode."""
        self._values = np.full(8760, -1.0, dtype=np.float64)
        self._index = self._initial_index
    
    def set_index(self, index: int):
        """Set starting index in annual cycle."""
        self._index = int(index) % 8760
        self._initial_index = self._index
    
    def add_observation(self, solar: float):
        """Record solar observation."""
        try:
            rank_idx = int(self._rank[self._index][0])
            self._values[rank_idx] = float(solar)
            self._index = (self._index + 1) % 8760
        except Exception as e:
            print(f"[SolarRank] Add observation failed: {e}")
    
    def get_ranges(self):
        """Get min/max envelope for next 24 hours."""
        limit = 3
        Mins, Maxs = [], []
        
        for i in range(24):
            minimum = 0.0
            maximum = 3.9
            
            try:
                cindex = int(self._rank[(self._index + i) % 8760][0])
                
                count = 0
                for j in range(cindex, max(0, cindex - 100), -1):
                    if self._values[j] >= 0 and self._values[j] < minimum:
                        minimum = self._values[j]
                        count += 1
                        if count == limit:
                            break
                
                count = 0
                for j in range(cindex, min(cindex + 100, 8760)):
                    if self._values[j] >= 0 and self._values[j] > maximum:
                        maximum = self._values[j]
                        count += 1
                        if count == limit:
                            break
            except Exception:
                pass
            
            Mins.append(minimum)
            Maxs.append(maximum)
        
        return Mins, Maxs


class Weather:
    """Manages weather forecast data."""
    
    def __init__(self):
        self._temperature = np.zeros((25, 1), dtype=np.float64)
        self._humidity = np.zeros((25, 1), dtype=np.float64)
        self._diffuse_solar_irradiance = np.zeros((25, 1), dtype=np.float64)
        self._direct_solar_irradiance = np.zeros((25, 1), dtype=np.float64)
    
    def update_temperature(self, temp, temp6h, temp12h, temp24h):
        self._temperature[:-1] = self._temperature[1:]
        self._temperature[0] = float(temp)
        
        if 6 < len(self._temperature):
            self._temperature[6] = float(temp6h)
        if 12 < len(self._temperature):
            self._temperature[12] = float(temp12h)
        if 24 < len(self._temperature):
            self._temperature[24] = float(temp24h)
    
    def update_humidity(self, humidity, humidity6h, humidity12h, humidity24h):
        self._humidity[:-1] = self._humidity[1:]
        self._humidity[0] = float(humidity)
        
        if 6 < len(self._humidity):
            self._humidity[6] = float(humidity6h)
        if 12 < len(self._humidity):
            self._humidity[12] = float(humidity12h)
        if 24 < len(self._humidity):
            self._humidity[24] = float(humidity24h)
    
    def update_diffuse_solar_irradiance(self, solar, solar6h, solar12h, solar24h):
        self._diffuse_solar_irradiance[:-1] = self._diffuse_solar_irradiance[1:]
        self._diffuse_solar_irradiance[0] = float(solar)
        
        if 6 < len(self._diffuse_solar_irradiance):
            self._diffuse_solar_irradiance[6] = float(solar6h)
        if 12 < len(self._diffuse_solar_irradiance):
            self._diffuse_solar_irradiance[12] = float(solar12h)
        if 24 < len(self._diffuse_solar_irradiance):
            self._diffuse_solar_irradiance[24] = float(solar24h)
    
    def update_direct_solar_irradiance(self, solar, solar6h, solar12h, solar24h):
        self._direct_solar_irradiance[:-1] = self._direct_solar_irradiance[1:]
        self._direct_solar_irradiance[0] = float(solar)
        
        if 6 < len(self._direct_solar_irradiance):
            self._direct_solar_irradiance[6] = float(solar6h)
        if 12 < len(self._direct_solar_irradiance):
            self._direct_solar_irradiance[12] = float(solar12h)
        if 24 < len(self._direct_solar_irradiance):
            self._direct_solar_irradiance[24] = float(solar24h)
    
    def get_next_24_direct_solar_irradiance(self) -> np.ndarray:
        """Get 24-hour ahead solar irradiance forecast."""
        return self._direct_solar_irradiance[1:25].copy()