import numpy as np
import networkx as nx

def generate_hard_maxcut(n: int, seed: int = 42) -> nx.Graph:
    np.random.seed(seed)
    g = nx.erdos_renyi_graph(n=n, p=0.6, seed=seed)
    for u, v in g.edges():
        g[u][v]['weight'] = float(np.random.randint(1, 100))
    return g

def generate_hard_mis(n: int, seed: int = 42) -> nx.Graph:
    return nx.erdos_renyi_graph(n=n, p=0.25, seed=seed)

def generate_hard_npp(n: int, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    return np.random.randint(10**12, 10**15, size=n, dtype=np.int64)