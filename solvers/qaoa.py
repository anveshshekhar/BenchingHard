import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

class QAOASolver:
    def __init__(self, p: int = 2, maxiter: int = 80, restarts: int = 4):
        self.p = p
        self.maxiter = maxiter
        self.restarts = restarts
        self.backend = AerSimulator()

    def solve(self, J: np.ndarray, h: np.ndarray) -> np.ndarray:
        N = J.shape[0]
        
        scale = max(np.max(np.abs(J)), np.max(np.abs(h)), 1e-8)
        J_norm = J / scale
        h_norm = h / scale

        def build_circuit(params):
            gamma = params[:self.p]
            beta = params[self.p:]
            qc = QuantumCircuit(N)
            qc.h(range(N))
            for l in range(self.p):
                for i in range(N):
                    for j in range(i + 1, N):
                        if J_norm[i, j] != 0:
                            qc.rzz(-4 * gamma[l] * J_norm[i, j], i, j)
                    if h_norm[i] != 0:
                        qc.rz(-2 * gamma[l] * h_norm[i], i)
                for i in range(N):
                    qc.rx(2 * beta[l], i)
            qc.measure_all()
            return qc

        def cost_fn(params):
            qc = build_circuit(params)
            counts = self.backend.run(qc, shots=1024).result().get_counts()
            total_energy = 0.0
            shots = sum(counts.values())
            for bitstr, count in counts.items():
                spins = np.array([1 if b == '0' else -1 for b in reversed(bitstr)])
                E = -np.dot(spins, J @ spins) - np.dot(h, spins)
                total_energy += E * count
            return total_energy / shots

        best_res = None
        best_cost = float('inf')

        for _ in range(self.restarts):
            init_params = np.random.uniform(0, np.pi, 2 * self.p)
            res = minimize(cost_fn, init_params, method='COBYLA', options={'maxiter': self.maxiter})
            if res.fun < best_cost:
                best_cost = res.fun
                best_res = res

        opt_qc = build_circuit(best_res.x)
        counts = self.backend.run(opt_qc, shots=2048).result().get_counts()
        best_bitstr = max(counts, key=counts.get)
        return np.array([1 if b == '0' else -1 for b in reversed(best_bitstr)])