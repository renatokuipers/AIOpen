from dgm.config import TrainingConfig
from dgm.evolution import DarwinGodelMachine


if __name__ == "__main__":
    cfg = TrainingConfig()
    machine = DarwinGodelMachine(cfg, population_size=3)
    machine.evolve(generations=3)
