# HEMS Benchmark Configurations

This folder contains YAML/JSON configuration files for running HEMS benchmarks.

## 📁 Structure

```
benchmark_configs/
├── README.md                  # This file
├── prototype.yaml            # Template to follow
├── example_sequential.yaml   # Sequential training example
├── example_parallel.yaml     # Parallel training example
└── example_both.yaml        # Compare sequential vs parallel
```

## 🚀 Quick Start

### 1. Using Example Configs

```bash
# Run sequential training
python -m hems.main --benchmark-config benchmark_configs/example_sequential.yaml

# Run parallel training
python -m hems.main --benchmark-config benchmark_configs/example_parallel.yaml

# Compare both approaches
python -m hems.main --benchmark-config benchmark_configs/example_both.yaml
```

### 2. Creating Your Own Config

1. Copy `prototype.yaml`
2. Modify parameters for your experiment
3. Run: `python -m hems.main --benchmark-config benchmark_configs/your_config.yaml`

## 📋 Configuration Guide

### Training Modes

#### **Sequential Training**
- Trains agent on Building_1, then Building_2, then Building_3
- Weights are transferred between buildings (transfer learning)
- Best model saved for each building + final averaged model
- Good for: Limited compute, transfer learning research

```yaml
training:
  mode: "sequential"
  episodes: 150  # Per building
  buildings:
    selection: "manual"
    ids: ["Building_1", "Building_2", "Building_3"]
```

#### **Parallel Training**
- Agent sees observations from all buildings simultaneously
- Single training process, multiple buildings per episode
- Buildings controlled separately (decentralized)
- Good for: Faster training, multi-building learning

```yaml
training:
  mode: "parallel"
  episodes: 450  # Total episodes (not per building)
  buildings:
    selection: "manual"
    ids: ["Building_1", "Building_2", "Building_3"]
```

#### **Both (Comparison)**
- Runs sequential AND parallel training
- Compares results between the two approaches
- Good for: Research, understanding trade-offs

```yaml
training:
  mode: "both"
  episodes: 150  # Per building for sequential
```

### Building Selection Strategies

#### **Manual Selection**
```yaml
buildings:
  selection: "manual"
  ids: ["Building_1", "Building_2", "Building_3"]
```

#### **Random Selection**
```yaml
buildings:
  selection: "random"
  count: 3  # Number of random buildings
```

#### **All Buildings**
```yaml
buildings:
  selection: "all"
```

#### **Different Buildings** (for validation/test)
```yaml
buildings:
  selection: "different"  # Automatically excludes training buildings
  count: 2
```

### Agent Configuration

Override any default configuration from `hems/core/config.py`:

```yaml
agents:
  - name: "dqn"
    enabled: true
    config:
      # Override defaults
      learning_rate: 0.0005
      batch_size: 256
      buffer_size: 500000
      epsilon_decay: 0.99
      
  - name: "rbc"
    enabled: true
    config:
      soc_threshold: 0.5
      
  - name: "baseline"
    enabled: false  # Skip this agent
```

### Tariff Types

```yaml
tariff:
  type: "hp_hc"  # Options: "default", "hp_hc", "tempo", "standard"
  price_hp: 0.22
  price_hc: 0.14
  hc_hours: [23, 0, 1, 2, 3, 4, 5, 6]
```

### Reward Functions

```yaml
reward:
  name: "custom_v5"  # Options: "custom_v5", "simple", "battery_health", "Chen_Bu_p2p", "mp_ppo"
  config:
    alpha_import_hp: 0.6
    alpha_peak: 0.01
    alpha_pv_base: 0.10
    alpha_pv_soc: 0.35
    alpha_soc: 0.05
```

### Testing Modes

```yaml
testing:
  enabled: true
  mode: "normal"  # Options: "normal", "general", "specific"
  episodes: 50
  buildings:
    selection: "manual"
    ids: ["Building_6", "Building_7"]
```

## 📊 Output Structure

After running a benchmark:

```
experiments/your_benchmark_name_TIMESTAMP/
├── benchmark_config.yaml          # Copy of your config
├── simulation_config.json         # Converted config
├── models/
│   ├── sequential/
│   │   ├── dqn_Building_1_best.pkl
│   │   ├── dqn_Building_2_best.pkl
│   │   ├── dqn_Building_3_best.pkl
│   │   └── dqn_final_sequential.pkl
│   └── parallel/
│       └── dqn_parallel_best.pkl
├── results/
│   └── benchmark_results.json
├── logs/
│   └── benchmark.log
└── visualizations/
    ├── training_curves.png
    └── performance_comparison.png
```

## 🎯 Best Practices

### 1. **Avoid Building Overlap**
```yaml
# ✅ GOOD - No overlap
training:
  buildings: ["Building_1", "Building_2", "Building_3"]
validation:
  buildings: ["Building_4", "Building_5"]
testing:
  buildings: ["Building_6", "Building_7"]

# ❌ BAD - Overlap between train and test
training:
  buildings: ["Building_1", "Building_2"]
testing:
  buildings: ["Building_2", "Building_3"]  # Building_2 appears in both!
```

### 2. **Choose Appropriate Episodes**
```yaml
# Sequential: episodes per building
training:
  mode: "sequential"
  episodes: 150  # Each building gets 150 episodes

# Parallel: total episodes
training:
  mode: "parallel"
  episodes: 450  # All buildings share 450 episodes (150 each effectively)
```

### 3. **Use Validation for Hyperparameter Tuning**
```yaml
validation:
  enabled: true
  frequency: 50  # Validate every 50 episodes
  buildings:
    selection: "different"  # Automatically excludes training buildings
```

### 4. **Reproducibility**
```yaml
seed: 42  # Always set a seed for reproducible results
```

## 🔧 Advanced Features

### Transfer Learning (Sequential Mode)
```yaml
advanced:
  sequential:
    transfer_learning: true  # Use weights from previous building
```

### Early Stopping
```yaml
training:
  early_stopping:
    enabled: true
    patience: 50
    min_delta: 0.01
```

### Model Checkpointing
```yaml
training:
  save_frequency: 50  # Save checkpoint every 50 episodes
  save_best: true     # Save best model
  save_final: true    # Save final model
```

## 📝 Example Workflows

### Research: Compare Algorithms
```yaml
agents:
  - name: "dqn"
    enabled: true
  - name: "rbc"
    enabled: true
  - name: "baseline"
    enabled: true
```

### Research: Transfer Learning Study
```yaml
training:
  mode: "sequential"
advanced:
  sequential:
    transfer_learning: true
```

### Production: Fast Training
```yaml
training:
  mode: "parallel"
  episodes: 200
use_gpu: true
```

## 🐛 Troubleshooting

### "Building overlap detected"
- Check that train/val/test buildings don't overlap
- Use `selection: "different"` for automatic separation

### "Invalid training mode"
- Must be: "sequential", "parallel", or "both"

### "Configuration validation failed"
- Check YAML syntax
- Ensure all required fields are present
- See `prototype.yaml` for complete template

## 📞 Support

For questions or issues:
- Email: akrem.dabbech@homensia.fr
- Check logs in `experiments/*/logs/benchmark.log`
- Review `prototype.yaml` for complete documentation

## 🔄 Version History

- **v1.0**: Initial release with sequential/parallel/both modes
- Support for multiple agents, validation, testing
- Automatic building overlap detection