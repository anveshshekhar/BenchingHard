import numpy as np

class SimulatedAnnealingSolver:
    def __init__(self, steps: int = None, sweeps_per_var: int = 2000, t_init: float = 10.0, t_min: float = 0.001):
        self.steps = steps
        self.sweeps_per_var = sweeps_per_var
        self.t_init = t_init
        self.t_min = t_min

    def solve(self, J: np.ndarray, h: np.ndarray) -> np.ndarray:
        N = J.shape[0]
        total_steps = self.steps if self.steps is not None else self.sweeps_per_var * N
        
        spins = np.random.choice([-1.0, 1.0], size=N)
        current_energy = -np.dot(spins, J @ spins) - np.dot(h, spins)
        
        best_spins = spins.copy()
        best_energy = current_energy
        
        alpha = (self.t_min / self.t_init) ** (1.0 / total_steps)
        temp = self.t_init

        for _ in range(total_steps):
            i = np.random.randint(0, N)
            delta_e = 2 * spins[i] * (2 * np.dot(J[i], spins) + h[i])
            
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / temp):
                spins[i] *= -1.0
                current_energy += delta_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_spins = spins.copy()
                    
            temp *= alpha

        return best_spins