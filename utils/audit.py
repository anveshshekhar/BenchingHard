import numpy as np
import networkx as nx

class SolutionAuditor:
    @staticmethod
    def audit_maxcut(graph: nx.Graph, spins: np.ndarray) -> dict:
        nodes = list(graph.nodes())
        node_map = {n: i for i, n in enumerate(nodes)}
        
        set_a, set_b = [], []
        cut_weight = 0.0
        
        for n, idx in node_map.items():
            if spins[idx] == 1:
                set_a.append(n)
            else:
                set_b.append(n)
                
        for u, v, d in graph.edges(data=True):
            i, j = node_map[u], node_map[v]
            w = d.get('weight', 1.0)
            if spins[i] != spins[j]:
                cut_weight += w

        return {
            "valid": True,
            "metric": cut_weight,
            "details": f"Partition A: {len(set_a)} nodes, Partition B: {len(set_b)} nodes"
        }

    @staticmethod
    def audit_mis(graph: nx.Graph, spins: np.ndarray) -> dict:
        nodes = list(graph.nodes())
        node_map = {n: i for i, n in enumerate(nodes)}
        
        selected_nodes = [n for n, idx in node_map.items() if spins[idx] == -1]
        violations = []

        for u, v in graph.edges():
            i, j = node_map[u], node_map[v]
            if spins[i] == -1 and spins[j] == -1:
                violations.append((u, v))

        is_valid = len(violations) == 0
        set_size = len(selected_nodes) if is_valid else 0

        return {
            "valid": is_valid,
            "metric": set_size,
            "details": f"Selected: {len(selected_nodes)} nodes, Violations: {len(violations)} edges"
        }

    @staticmethod
    def audit_npp(numbers: np.ndarray, spins: np.ndarray) -> dict:
        set_1 = numbers[spins == 1]
        set_2 = numbers[spins == -1]
        
        sum1 = np.sum(set_1)
        sum2 = np.sum(set_2)
        diff = abs(sum1 - sum2)

        return {
            "valid": True,
            "metric": diff,
            "details": f"Sum1: {sum1}, Sum2: {sum2}, Diff: {diff}"
        }