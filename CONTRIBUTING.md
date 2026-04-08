# Contributing to HEMS-Arena

HEMS-Arena is designed to grow through community contributions. The registry/factory architecture makes it possible to add new components without modifying the benchmark core.

## How to Add a New Component

| I want to add... | Inherit from | Register in | Tests |
|---|---|---|---|
| Algorithm | `BaseAlgorithm` (`hems/algorithms/base.py`) | `ALGORITHM_REGISTRY` in `hems/algorithms/registry.py` | `tests/` |
| Reward function | `BaseRewardFunction` (`hems/rewards/base.py`) | `REWARD_REGISTRY` in `hems/rewards/registry.py` | `tests/` |
| Strategy | `BaseStrategy` (`hems/strategies/base.py`) | `STRATEGY_REGISTRY` in `hems/strategies/registry.py` | `tests/` |
| Environment adapter | `BaseHEMSEnvironment` (`hems/environments/base.py`) | `EnvironmentRegistry` in `hems/environments/factory.py` | `tests/` |

### Adding a New Algorithm (step-by-step)

1. Create `hems/algorithms/my_algorithm.py` with a class inheriting from `BaseAlgorithm`.
2. Implement the required methods: `act()`, `learn()`, `reset()`.
3. Register it in `hems/algorithms/registry.py`:
   ```python
   from .my_algorithm import MyAlgorithm
   ALGORITHM_REGISTRY['my_algorithm'] = MyAlgorithm
   ```
4. Add a legacy mapping in `hems/agents/factory.py` if you want it accessible via `--agents my_algorithm`:
   ```python
   'my_algorithm': ('my_algorithm', 'custom_v5', 'single_agent'),
   ```
5. Write tests in `tests/test_my_algorithm.py`.
6. Run tests: `pytest tests/test_my_algorithm.py -v`.

### Adding a New Reward Function

1. Create your reward class in `hems/rewards/` inheriting from `BaseRewardFunction`.
2. Implement the `calculate()` method.
3. Register it in `hems/rewards/registry.py`.

### Adding a New Environment Adapter

1. Create your adapter in `hems/environments/` inheriting from `BaseHEMSEnvironment`.
2. Implement the `reset()` and `step()` methods following the Gymnasium API.
3. Register it in `hems/environments/factory.py`.

## Git Workflow

- **Branches**: `dev` is the development branch, `main` is for stable releases only.
- **Feature branches**: `feature/issue-N-description`
- **Commit format**: Gitmoji required -- `<emoji> <type>: <description> (#issue-number)`
- Every PR must reference an existing GitHub issue and target `dev`.

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

- Google-style docstrings for all public classes and methods.
- Type hints encouraged.
