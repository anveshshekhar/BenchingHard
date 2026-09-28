import numpy as np
import networkx as nx
from .base import BaseProblem

class MaxCutProblem(BaseProblem):
    def __init__(self, graph: nx.Graph):
        super().__init__(graph.number_of_nodes())
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.node_map = {n: i for i, n in enumerate(self.nodes)}

    def get_ising(self) -> tuple[np.ndarray, np.ndarray]:
        N = self.num_vars
        J = np.zeros((N, N))
        h = np.zeros(N)
        for u, v, d in self.graph.edges(data=True):
            i, j = self.node_map[u], self.node_map[v]
            w = d.get('weight', 1.0)
            J[i, j] = -w / 2.0
            J[j, i] = -w / 2.0
        return J, h

    def evaluate(self, spins: np.ndarray) -> float:
        weight = 0.0
        for u, v, d in self.graph.edges(data=True):
            i, j = self.node_map[u], self.node_map[v]
            if spins[i] != spins[j]:
                weight += d.get('weight', 1.0)
        return weight