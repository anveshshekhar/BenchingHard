from abc import ABC, abstractmethod
import numpy as np

class BaseProblem(ABC):
    def __init__(self, num_vars: int):
        self.num_vars = num_vars

    @abstractmethod
    def get_ising(self) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def evaluate(self, spins: np.ndarray) -> float:
        pass