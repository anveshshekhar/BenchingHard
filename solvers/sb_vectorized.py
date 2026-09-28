import numpy as np

class VectorizedSBSolver:
    def __init__(self, steps: int = 2000, dt: float = 0.5, c0: float = 1.0, instances: int = 200):
        self.steps = steps
        self.dt = dt
        self.c0 = c0
        self.instances = instances

    def solve(self, J: np.ndarray, h: np.ndarray) -> np.ndarray:
        N = J.shape[0]
        
        scale = max(np.max(np.abs(J)), np.max(np.abs(h)), 1e-12)
        J_norm = J / scale
        h_norm = h / scale

        xi = 0.5 / np.sqrt(N)
        h_col = h_norm[:, None]

        x = np.random.uniform(-0.1, 0.1, (N, self.instances))
        y = np.zeros((N, self.instances))

        for step in range(self.steps):
            a_t = step / self.steps
            
            sgn_x = np.sign(x)
            sgn_x[sgn_x == 0] = 1.0
            
            force = -(1.0 - a_t) * x + xi * (2.0 * (J_norm @ sgn_x) + h_col)
            y += self.dt * force
            x += self.dt * y

            over_limit = np.abs(x) >= 1.0
            x = np.clip(x, -1.0, 1.0)
            y[over_limit] = 0.0

        spins = np.sign(x)
        spins[spins == 0] = 1.0
        
        energies = -np.sum(spins * (J @ spins), axis=0) - np.dot(h, spins)
        best_idx = np.argmin(energies)
        return spins[:, best_idx]