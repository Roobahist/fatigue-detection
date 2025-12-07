import itertools
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna


def generate_grid_configs(search_space: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate all combinations for grid search.

    search_space: {name: [values]}

    Returns:
        List of dicts {name: value, ...}
    """
    if not search_space:
        return [{}]

    keys = list(search_space.keys())
    value_lists = []
    for k in keys:
        v = search_space[k]
        if not isinstance(v, (list, tuple)):
            raise ValueError(
                f"Grid search requires list/tuple for key '{k}', got {type(v)}"
            )
        value_lists.append(list(v))

    configs: List[Dict[str, Any]] = []
    for values in itertools.product(*value_lists):
        cfg = {k: values[i] for i, k in enumerate(keys)}
        configs.append(cfg)
    return configs


def sample_random_config(
    search_space: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    """
    Sample a single random configuration from a mixed search_space.

    Rules:
        - If value is a list -> sample categorical uniformly.
        - If value is a tuple and first element is a string:
            ("log_uniform", low, high)
            ("uniform", low, high)
            ("int", low, high)
        - Else, treat as list/tuple of categorical values.
    """
    hparams: Dict[str, Any] = {}
    for name, spec in search_space.items():
        if isinstance(spec, tuple) and spec and isinstance(spec[0], str):
            kind = spec[0]
            if kind == "log_uniform":
                _, low, high = spec
                if low <= 0 or high <= 0:
                    raise ValueError(
                        f"log_uniform for '{name}' requires positive bounds."
                    )
                log_low = math.log(low)
                log_high = math.log(high)
                val = math.exp(rng.uniform(log_low, log_high))
                hparams[name] = val
            elif kind == "uniform":
                _, low, high = spec
                hparams[name] = rng.uniform(low, high)
            elif kind == "int":
                _, low, high = spec
                hparams[name] = rng.randint(int(low), int(high))
            else:
                raise ValueError(
                    f"Unknown random distribution kind '{kind}' for '{name}'."
                )
        else:
            if not isinstance(spec, (list, tuple)):
                raise ValueError(
                    f"Random search categorical spec for '{name}' must be list/tuple."
                )
            hparams[name] = rng.choice(list(spec))
    return hparams


def generate_random_configs(
    search_space: Dict[str, Any],
    n_trials: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate n_trials random configs from search_space.
    """
    if not search_space:
        return [{}]
    rng = random.Random(seed)
    return [sample_random_config(search_space, rng) for _ in range(n_trials)]


def suggest_hparams_from_trial(
    trial: optuna.trial.Trial,
    search_space: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Suggest hyperparameters from an Optuna trial based on search_space.

    Rules parallel to random search:
        - list / tuple (not starting with str) -> categorical
        - ("log_uniform", low, high)
        - ("uniform", low, high)
        - ("int", low, high)
    """
    hparams: Dict[str, Any] = {}
    for name, spec in search_space.items():
        if isinstance(spec, tuple) and spec and isinstance(spec[0], str):
            kind = spec[0]
            if kind == "log_uniform":
                _, low, high = spec
                hparams[name] = trial.suggest_float(name, low, high, log=True)
            elif kind == "uniform":
                _, low, high = spec
                hparams[name] = trial.suggest_float(name, low, high, log=False)
            elif kind == "int":
                _, low, high = spec
                hparams[name] = trial.suggest_int(name, int(low), int(high))
            else:
                raise ValueError(
                    f"Unknown bayesian distribution kind '{kind}' for '{name}'."
                )
        else:
            if not isinstance(spec, (list, tuple)):
                raise ValueError(
                    f"Bayesian categorical spec for '{name}' must be list/tuple."
                )
            hparams[name] = trial.suggest_categorical(name, list(spec))
    return hparams
