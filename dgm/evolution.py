from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import torch

from .agent import Agent
from .config import TrainingConfig
from .modifications import MODIFICATIONS, Modification


class Archive:
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.agents: List[Agent] = []

    def add(self, agent: Agent):
        self.agents.append(agent)
        if len(self.agents) > self.max_size:
            self.agents.sort(key=lambda a: a.score, reverse=True)
            self.agents = self.agents[: self.max_size]

    def sample_parents(self, k: int = 2) -> List[Agent]:
        return random.sample(self.agents, k)


class DarwinGodelMachine:
    def __init__(self, initial_config: TrainingConfig, population_size: int = 4):
        self.population_size = population_size
        self.archive = Archive(max_size=10)
        base_agent = Agent(config=initial_config)
        self.archive.add(base_agent)
        self.history: List[Agent] = []
        self.dataset = self._make_dataset()

    def _make_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        np.random.seed(0)
        X = np.random.randn(200, 2)
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def evolve(self, generations: int = 3):
        for _ in range(generations):
            parents = self.archive.sample_parents(1)
            parent = parents[0]
            offspring = []
            for _ in range(self.population_size):
                cfg = parent.config.copy()
                mod = random.choice(MODIFICATIONS)
                cfg, desc = mod(cfg)
                child = Agent(config=cfg, parent_id=parent.agent_id)
                child.evaluate(self.dataset)
                offspring.append((child, desc))
            for child, desc in offspring:
                self.archive.add(child)
                self.history.append(child)
                print(f"Added offspring {child.agent_id[:8]} from parent {parent.agent_id[:8]}: {desc} score={child.score:.3f}")

