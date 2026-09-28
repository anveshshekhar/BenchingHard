import numpy as np

class ClassicalGreedySolver:
    def __init__(self, restarts: int = 30):
        self.restarts = restarts

    def solve(self, J: np.ndarray, h: np.ndarray) -> np.ndarray:
        N = J.shape[0]
        best_spins = None
        best_energy = float('inf')
        for _ in range(self.restarts):
            spins = np.random.choice([-1.0, 1.0], size=N)
            improved = True
            while improved:
                improved = False
                for i in range(N):
                    delta = 2 * spins[i] * (2 * np.dot(J[i], spins) + h[i])
                    if delta < 0:
                        spins[i] *= -1.0
                        improved = True
            energy = -np.dot(spins, J @ spins) - np.dot(h, spins)
            if energy < best_energy:
                best_energy = energy
                best_spins = spins.copy()
        return best_spins