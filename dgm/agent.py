from __future__ import annotations

import ast
import inspect
import textwrap
import types
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from pydantic import BaseModel

from .config import TrainingConfig


@dataclass
class Agent:
    config: TrainingConfig
    parent_id: Optional[str] = None
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_code: str = ""
    train_code: str = ""
    score: float = 0.0
    lineage: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.parent_id:
            self.lineage.append(self.parent_id)
        self.generate_code()

    def generate_model_code(self) -> str:
        layers = []
        in_features = 2  # features for synthetic dataset
        for i, size in enumerate(self.config.hidden_layer_sizes):
            layers.append(
                f"nn.Linear({in_features}, {size})",
            )
            act = self.config.activation_functions[min(i, len(self.config.activation_functions) - 1)]
            act_cls = {"relu": "ReLU", "tanh": "Tanh", "sigmoid": "Sigmoid"}[act]
            layers.append(f"nn.{act_cls}()")
            if self.config.dropout_rate > 0:
                layers.append(f"nn.Dropout({self.config.dropout_rate})")
            in_features = size
        layers.append(f"nn.Linear({in_features}, 2)")
        body = ",\n            ".join(layers)
        model_code = textwrap.dedent(
            f"""
            import torch.nn as nn

            class GeneratedModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        {body}
                    )

                def forward(self, x):
                    return self.net(x)
            """
        )
        return model_code

    def generate_training_code(self) -> str:
        optim_line = {
            "adam": "torch.optim.Adam",
            "sgd": "torch.optim.SGD",
            "rmsprop": "torch.optim.RMSprop",
        }[self.config.optimizer_type]
        code = textwrap.dedent(
            f"""
            import torch

            def train(model, train_data):
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model.to(device)
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = {optim_line}(model.parameters(), lr={self.config.learning_rate}, weight_decay={self.config.weight_decay})
                model.train()
                for epoch in range({self.config.epochs}):
                    X, y = train_data
                    X = X.to(device)
                    y = y.to(device)
                    for i in range(0, len(X), {self.config.batch_size}):
                        x_batch = X[i:i+{self.config.batch_size}]
                        y_batch = y[i:i+{self.config.batch_size}]
                        optimizer.zero_grad()
                        outputs = model(x_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()
                return model
            """
        )
        return code

    def generate_code(self):
        self.model_code = self.generate_model_code()
        self.train_code = self.generate_training_code()

    def build(self) -> types.ModuleType:
        namespace: Dict[str, Any] = {}
        exec(self.model_code, namespace)
        exec(self.train_code, namespace)
        module = types.SimpleNamespace(**namespace)
        return module

    def evaluate(self, data: tuple[torch.Tensor, torch.Tensor]) -> float:
        module = self.build()
        model = module.GeneratedModel()
        train_func = module.train
        trained = train_func(model, data)
        trained.eval()
        with torch.no_grad():
            X, y = data
            logits = trained(X)
            preds = logits.argmax(dim=1)
            accuracy = (preds == y).float().mean().item()
        self.score = accuracy
        return accuracy
