import numpy as np

class ExactIsingSolver:
    @staticmethod
    def solve(J: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, float]:
        N = J.shape[0]
        best_spins = None
        best_energy = float('inf')
        
        for i in range(1 << N):
            spins = np.array([1.0 if (i >> j) & 1 else -1.0 for j in range(N)])
            energy = -np.dot(spins, J @ spins) - np.dot(h, spins)
            if energy < best_energy:
                best_energy = energy
                best_spins = spins
                
        return best_spins, best_energy