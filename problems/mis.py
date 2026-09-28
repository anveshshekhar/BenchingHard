import numpy as np
import networkx as nx
from .base import BaseProblem

class MISProblem(BaseProblem):
    def __init__(self, graph: nx.Graph, penalty: float = 3.0):
        super().__init__(graph.number_of_nodes())
        self.graph = graph
        self.penalty = penalty
        self.nodes = list(graph.nodes())
        self.node_map = {n: i for i, n in enumerate(self.nodes)}

    def get_ising(self) -> tuple[np.ndarray, np.ndarray]:
        N = self.num_vars
        J = np.zeros((N, N))
        h = np.zeros(N)
        
        for i in range(N):
            deg = self.graph.degree(self.nodes[i])
            h[i] = (self.penalty / 4.0) * deg - 0.5
            
        for u, v in self.graph.edges():
            i, j = self.node_map[u], self.node_map[v]
            J[i, j] = -self.penalty / 8.0
            J[j, i] = -self.penalty / 8.0
            
        return J, h

    def evaluate(self, spins: np.ndarray) -> float:
        selected = (spins == -1).astype(int)
        for u, v in self.graph.edges():
            i, j = self.node_map[u], self.node_map[v]
            if selected[i] == 1 and selected[j] == 1:
                return 0.0
        return float(np.sum(selected))