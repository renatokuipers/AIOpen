from __future__ import annotations

import random
from typing import Callable, List, Tuple

from .config import TrainingConfig


Modification = Callable[[TrainingConfig], Tuple[TrainingConfig, str]]


def modify_learning_rate(cfg: TrainingConfig) -> Tuple[TrainingConfig, str]:
    factor = random.choice([0.5, 2.0, 3.0])
    new_lr = max(min(cfg.learning_rate * factor, 10), 1e-6)
    new_cfg = cfg.copy()
    new_cfg.learning_rate = new_lr
    return new_cfg, f"learning_rate * {factor}"


def modify_batch_size(cfg: TrainingConfig) -> Tuple[TrainingConfig, str]:
    factor = random.choice([0.5, 2.0])
    new_bs = max(int(cfg.batch_size * factor), 1)
    new_cfg = cfg.copy()
    new_cfg.batch_size = new_bs
    return new_cfg, f"batch_size * {factor}"


def modify_optimizer(cfg: TrainingConfig) -> Tuple[TrainingConfig, str]:
    opts = ["adam", "sgd", "rmsprop"]
    opts.remove(cfg.optimizer_type)
    new_opt = random.choice(opts)
    new_cfg = cfg.copy()
    new_cfg.optimizer_type = new_opt
    return new_cfg, f"optimizer -> {new_opt}"


def modify_architecture_add_layer(cfg: TrainingConfig) -> Tuple[TrainingConfig, str]:
    new_cfg = cfg.copy()
    size = random.randint(16, 128)
    new_cfg.hidden_layer_sizes.append(size)
    new_cfg.activation_functions.append(random.choice(["relu", "tanh", "sigmoid"]))
    return new_cfg, f"added layer {size}"


def modify_architecture_remove_layer(cfg: TrainingConfig) -> Tuple[TrainingConfig, str]:
    if len(cfg.hidden_layer_sizes) <= 1:
        return cfg.copy(), "no layer removed"
    new_cfg = cfg.copy()
    removed = new_cfg.hidden_layer_sizes.pop(random.randrange(len(new_cfg.hidden_layer_sizes)))
    if len(new_cfg.activation_functions) >= len(new_cfg.hidden_layer_sizes) + 1:
        new_cfg.activation_functions.pop()
    return new_cfg, f"removed layer {removed}"


def modify_dropout(cfg: TrainingConfig) -> Tuple[TrainingConfig, str]:
    new_cfg = cfg.copy()
    new_cfg.dropout_rate = round(min(max(cfg.dropout_rate + random.choice([-0.1, 0.1, 0.2]), 0), 0.9), 2)
    return new_cfg, f"dropout -> {new_cfg.dropout_rate}"


def modify_weight_decay(cfg: TrainingConfig) -> Tuple[TrainingConfig, str]:
    new_cfg = cfg.copy()
    new_cfg.weight_decay = round(min(max(cfg.weight_decay + random.choice([-0.01, 0.01, 0.05]), 0), 0.5), 3)
    return new_cfg, f"weight_decay -> {new_cfg.weight_decay}"


def modify_epochs(cfg: TrainingConfig) -> Tuple[TrainingConfig, str]:
    new_cfg = cfg.copy()
    new_cfg.epochs = max(cfg.epochs + random.choice([-1, 1, 2]), 1)
    return new_cfg, f"epochs -> {new_cfg.epochs}"


MODIFICATIONS: List[Modification] = [
    modify_learning_rate,
    modify_batch_size,
    modify_optimizer,
    modify_architecture_add_layer,
    modify_architecture_remove_layer,
    modify_dropout,
    modify_weight_decay,
    modify_epochs,
]
