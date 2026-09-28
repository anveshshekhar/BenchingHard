import numpy as np
from .base import BaseProblem

class NumberPartitionProblem(BaseProblem):
    def __init__(self, numbers: np.ndarray):
        super().__init__(len(numbers))
        self.numbers = np.array(numbers, dtype=float)

    def get_ising(self) -> tuple[np.ndarray, np.ndarray]:
        N = self.num_vars
        J = np.zeros((N, N))
        h = np.zeros(N)
        for i in range(N):
            for j in range(i + 1, N):
                val = -self.numbers[i] * self.numbers[j]
                J[i, j] = val
                J[j, i] = val
        return J, h

    def evaluate(self, spins: np.ndarray) -> float:
        return float(np.abs(np.dot(self.numbers, spins)))