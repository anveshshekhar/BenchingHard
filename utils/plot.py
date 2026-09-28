import csv
import matplotlib.pyplot as plt

def plot_benchmark_results(csv_filename: str = "benchmark_results.csv"):
    data = {"MaxCut": {}, "MIS": {}, "NPP": {}}

    with open(csv_filename, mode="r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            prob = row["Problem"]
            N = int(row["Scale_N"])
            solver = row["Solver"]
            metric = float(row["Metric"])
            time_ms = float(row["Time_ms"])

            if solver not in data[prob]:
                data[prob][solver] = {"N": [], "Metric": [], "Time": []}

            data[prob][solver]["N"].append(N)
            data[prob][solver]["Metric"].append(metric)
            data[prob][solver]["Time"].append(time_ms)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # 1. Max-Cut Plots
    for solver, vals in data["MaxCut"].items():
        axes[0, 0].plot(vals["N"], vals["Metric"], marker="o", label=solver)
        axes[0, 1].plot(vals["N"], vals["Time"], marker="s", label=solver)
    axes[0, 0].set_title("Max-Cut: Cut Weight vs N (Higher is Better)")
    axes[0, 0].set_ylabel("Cut Weight")
    axes[0, 1].set_title("Max-Cut: Execution Time vs N")
    axes[0, 1].set_ylabel("Time (ms)")

    # 2. MIS Plots
    for solver, vals in data["MIS"].items():
        axes[1, 0].plot(vals["N"], vals["Metric"], marker="o", label=solver)
        axes[1, 1].plot(vals["N"], vals["Time"], marker="s", label=solver)
    axes[1, 0].set_title("MIS: Valid Set Size vs N (Higher is Better)")
    axes[1, 0].set_ylabel("Set Size")
    axes[1, 1].set_title("MIS: Execution Time vs N")
    axes[1, 1].set_ylabel("Time (ms)")

    # 3. NPP Plots
    for solver, vals in data["NPP"].items():
        axes[2, 0].plot(vals["N"], vals["Metric"], marker="o", label=solver)
        axes[2, 1].plot(vals["N"], vals["Time"], marker="s", label=solver)
    axes[2, 0].set_title("Number Partitioning: Difference vs N (Log Scale, Lower is Better)")
    axes[2, 0].set_yscale("log")
    axes[2, 0].set_ylabel("Partition Difference")
    axes[2, 1].set_title("Number Partitioning: Execution Time vs N")
    axes[2, 1].set_ylabel("Time (ms)")

    for ax in axes.flat:
        ax.set_xlabel("Problem Size (N)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.savefig("benchmark_scaling_plots.png", dpi=300)
    print("Plots generated and saved as 'benchmark_scaling_plots.png'.")

if __name__ == "__main__":
    plot_benchmark_results()