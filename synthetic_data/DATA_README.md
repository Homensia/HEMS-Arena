# MATRIX Synthetic Dataset Generator & Evaluator

## Abstract

This module provides a **scientifically validated synthetic dataset generator** and a **comprehensive evaluator** for Home Energy Management Systems (HEMS) research.
It produces realistic time series for:

* Household electricity demand
* Photovoltaic (PV) generation
* Electric vehicle (EV) availability, state of charge (SoC), and charging
* Electricity prices (French tariffs, dynamic pricing, random models)

The companion evaluator runs **deep data quality checks** and produces metrics & plots to ensure realism.
The system is designed for **reinforcement learning benchmarking** and **energy system simulation**.

---

## Motivation & Context

HEMS integrate PV, EVs, batteries, and dynamic tariffs. Training and validating intelligent agents requires **realistic and reproducible datasets**.
Real-world data is scarce, fragmented, or confidential. Our generator solves this by combining **scientific models** with **configurable scenarios**, producing datasets suitable for research and education.

---

## Project Structure

```
matrix_data/
├── data_gen/         # Generators (weather, PV, load, EV, tariffs)
├── metricsviz/       # Evaluator & plots
├── main.py           # CLI entrypoint for dataset generation
└── README.md         # Documentation (this file)
```

---

## Data Generation

### Configurations

* **Default scenario**: 1 house, 1 EV, 365 days, 60-min step, FR HP/HC tariff.
* **Customizable parameters**:

  * Days & step size (1min, 15min, 1h, daily)
  * Number of houses and EVs per house
  * PV capacity, tilt, azimuth, performance ratio
  * Tariffs: FR Base, FR HP/HC, FR Tempo, Dynamic, Random
  * Random seed, output directory

### Models & Formulas

**Weather model**

* Temperature:

  $$
  T(d,h) = \mu + A_{season}\sin\Big(\tfrac{2\pi(d-172)}{365}\Big) + A_{daily}\sin\Big(\tfrac{2\pi(h-15)}{24}\Big) + \varepsilon
  $$
* Irradiance:

  $$
  GHI = 1000 \cdot f_{season}(d) \cdot \sin(\theta_{elevation})
  $$
* Clouds: 3-state Markov chain.

**PV model (PVWatts)** \[Dobos, 2014]:

$$
P_{ac} = P_{rated}\cdot \tfrac{G}{1000} \cdot (1 + \gamma(T_{cell}-25)) \cdot \eta_{inv} \cdot PR
$$

**Household load** \[Richardson et al., 2010]:

* Diurnal pattern with Gaussian morning/evening peaks.
* Noise: AR(1) process.
* Weekend scaling.

**EV model** \[IEA, 2023]:

* Daily distance → energy demand.
* Arrival/departure distributions.
* SoC evolution:

  $$
  SoC_t = SoC_{t-1} - \tfrac{drive}{Cap} + \tfrac{charge}{Cap}
  $$
* Charging policies: immediate or tariff-aware.

**Tariffs**

* **FR Base**: constant price.
* **FR HP/HC**: off-peak vs peak hours.
* **FR Tempo**: seasonal daily colors (blue/white/red).
* **Dynamic**: sinusoidal + stochastic spikes.
* **Random**: Gaussian prices.

### Outputs

Each scenario creates:

```
datasets/<scenario_name>_<N>/
├── csv/
│   ├── <scenario>_house_elec_<N>_<house>.csv
│   ├── <scenario>_pv_generation_<N>_<house>.csv
│   ├── <scenario>_ev_<N>_<house>_<ev>.csv
│   └── <scenario>_price_<N>.csv
├── metadata.json   # scenario config, validation, file hashes
├── README.md       # short description
└── reports/        # metrics + plots (after analysis)
```

---

## Evaluation & Visualization

The evaluator (`matrix_data/metricsviz/`) performs:

* **Data quality**: nulls, duplicates, monotonicity
* **Descriptive stats**: mean, std, quantiles, fractions
* **Energy KPIs**: total load/PV, imports/exports, self-consumption, load factor
* **Peaks & ramps**: max values, 95th percentile ramps
* **Autocorrelation**: up to 24h
* **Correlations**:

  * Load↔PV, Load↔Price, PV↔Price
  * House-to-house load & PV
  * Full correlation matrix across all variables
* **Cross-correlation**: ±24h lags (Load↔Price, Load↔PV, PV↔Price)
* **Seasonality**: month×hour profiles
* **Spectral (FFT)**: dominant periods (daily, weekly)
* **EV metrics**: availability, charging sessions, SoC stats, arrivals & departures
* **Tariff elasticity**: price vs net-import correlation

### Plots

Generated automatically under `reports/plots/`:

* Time series (full & first week)
* Duration curves
* Histograms
* Hourly & weekday/weekend profiles
* Heatmaps (day×hour, month×hour)
* Autocorrelation & cross-correlation plots
* FFT spectra
* Correlation heatmaps
* EV availability & charging plots

---

## Usage

### Generate dataset

Default scenario:

```bash
python -m matrix_data.main --use-default
```

Custom scenario:

```bash
python -m matrix_data.main --scenario-name demo --days 90 --step-minutes 15 --n-houses 2 --tariff FR_TEMPO
```

### Evaluate dataset

```bash
python -m matrix_data.metricsviz.analyze datasets/demo_1
```

---

## Example Datasets

We provide 3 sample scenarios:

* `basic_data_1`: 1 house, 1 EV, 365 days, 1h step, FR Base tariff.
* `basic_data_2`: 1 house, 1 EV, 365 days, 1h step, FR HP/HC tariff.
* `basic_data_3`: 1 house, 1 EV, 365 days, 1h step, FR HP/HC tariff with random seed variation.

Each includes CSVs, metadata, and reports after evaluation.

---

## References

1. Dobos, A. P. (2014). *PVWatts Version 5 Manual*. NREL.
2. Ineichen, P., & Perez, R. (2002). A new airmass independent formulation for the Linke turbidity coefficient. *Solar Energy*.
3. Richardson, I., Thomson, M., Infield, D., & Clifford, C. (2010). Domestic electricity use: A high-resolution energy demand model. *Energy and Buildings*.
4. International Energy Agency. (2023). *Global EV Outlook*. OECD/IEA.
5. EDF France. Official tariff documentation (Base, HP/HC, Tempo).
6. Conejo, A. J., Morales, J. M., & Baringo, L. (2010). Real-time demand response model. *IEEE Transactions on Smart Grid*.

---

## Future Roadmap

* Multi-household with diversity in PV/load
* Vehicle-to-home (V2H) / bidirectional EVs
* Seasonal PV/weather variation
* Tempo tariff daily calendar
* Higher resolution (15min, 5min)
* Integration with reinforcement learning benchmarks

---

## License

MIT License — freely reusable for research and education.
