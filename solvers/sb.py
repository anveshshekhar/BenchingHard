import numpy as np

class SimulatedBifurcationSolver:
    def __init__(self, steps: int = 1000, dt: float = 0.1, c0: float = 1.0):
        self.steps = steps
        self.dt = dt
        self.c0 = c0

    def solve(self, J: np.ndarray, h: np.ndarray) -> np.ndarray:
        N = J.shape[0]
        x = np.random.uniform(-0.1, 0.1, N)
        y = np.random.uniform(-0.1, 0.1, N)
        j_max = np.max(np.abs(J)) if np.max(np.abs(J)) > 0 else 1.0
        xi = 0.5 / (np.sqrt(N) * j_max)

        for step in range(self.steps):
            a_t = step / self.steps
            x = x + self.dt * y
            force = -(1.0 - a_t) * x - self.c0 * (x**3) + xi * (2 * (J @ x) + h)
            y = y + self.dt * force
            x = np.clip(x, -1.0, 1.0)

        spins = np.sign(x)
        spins[spins == 0] = 1.0
        return spins