import csv
import time
import numpy as np

from problems.maxcut import MaxCutProblem
from problems.mis import MISProblem
from problems.npp import NumberPartitionProblem
from problems.generators import generate_hard_maxcut, generate_hard_mis, generate_hard_npp

from solvers.greedy import ClassicalGreedySolver
from solvers.sa import SimulatedAnnealingSolver
from solvers.sb_vectorized import VectorizedSBSolver

from utils.audit import SolutionAuditor

def run():
    scales = [
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
        120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500
    ]

    solvers = {
        "Greedy Local Search": ClassicalGreedySolver(restarts=50),
        "Simulated Annealing": SimulatedAnnealingSolver(sweeps_per_var=1500),
        "Vectorized SB (200 parallel)": VectorizedSBSolver(steps=2000, instances=200)
    }

    csv_filename = "benchmark_results.csv"
    csv_headers = ["Problem", "Scale_N", "Solver", "Metric", "Time_ms", "Audit_Status"]

    print("=== BenchingHard: Thorough Fine-Grained Benchmark Suite ===")
    print(f"Scales ({len(scales)} grid points): {scales}")
    print(f"Streaming live results to '{csv_filename}'...\n")

    with open(csv_filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_headers)

        for idx, N in enumerate(scales, 1):
            print(f"\n==================================================")
            print(f"  PROGRESS: [{idx}/{len(scales)}] Running Benchmark for N = {N}")
            print(f"==================================================")

            # 1. Max-Cut Benchmark
            g_mc = generate_hard_maxcut(N)
            maxcut = MaxCutProblem(g_mc)
            J_mc, h_mc = maxcut.get_ising()

            print(f"\n  [Max-Cut (N={N})]")
            for name, solver in solvers.items():
                t0 = time.time()
                spins = solver.solve(J_mc, h_mc)
                t1 = (time.time() - t0) * 1000
                audit = SolutionAuditor.audit_maxcut(g_mc, spins)
                status = "PASS" if audit["valid"] else "FAIL"

                print(f"    {name:<30} | Cut: {audit['metric']:10.2f} | Time: {t1:8.2f} ms | [{status}]")
                writer.writerow(["MaxCut", N, name, audit["metric"], round(t1, 2), status])

            # 2. Number Partitioning Benchmark
            numbers = generate_hard_npp(N)
            npp = NumberPartitionProblem(numbers)
            J_npp, h_npp = npp.get_ising()

            print(f"\n  [Number Partitioning (N={N})]")
            for name, solver in solvers.items():
                t0 = time.time()
                spins = solver.solve(J_npp, h_npp)
                t1 = (time.time() - t0) * 1000
                audit = SolutionAuditor.audit_npp(numbers, spins)
                status = "PASS" if audit["valid"] else "FAIL"

                print(f"    {name:<30} | Diff: {audit['metric']:15.0f} | Time: {t1:8.2f} ms | [{status}]")
                writer.writerow(["NPP", N, name, audit["metric"], round(t1, 2), status])

            # 3. Maximum Independent Set Benchmark
            g_mis = generate_hard_mis(N)
            mis = MISProblem(g_mis)
            J_mis, h_mis = mis.get_ising()

            print(f"\n  [Maximum Independent Set (N={N})]")
            for name, solver in solvers.items():
                t0 = time.time()
                spins = solver.solve(J_mis, h_mis)
                t1 = (time.time() - t0) * 1000
                audit = SolutionAuditor.audit_mis(g_mis, spins)
                status = "PASS" if audit["valid"] else "FAIL"

                print(f"    {name:<30} | Size: {audit['metric']:4.0f} | Time: {t1:8.2f} ms | [{status}]")
                writer.writerow(["MIS", N, name, audit["metric"], round(t1, 2), status])

            csv_file.flush()

    print("\n\n=== Benchmark Complete ===")
    print(f"Full dataset successfully saved to {csv_filename}")

if __name__ == "__main__":
    run()