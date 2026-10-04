#!/usr/bin/env python3

import re
import matplotlib.pyplot as plt

# ============================================================
# Input files
# ============================================================

files = {
    "MPI-SYNC": "result--2048-32768-1node-4GPUs-MMM.txt",
    "MPI-CHUNKED-ASYNC": "result--2048-32768-1node-4GPUs-YYY.txt",
    "CUDA-Aware-SYNC": "result--2048-32768-1node-4GPUs-CCC.txt",
    "CUDA-Aware-ASYNC": "result--2048-32768-1node-4GPUs-XXX.txt",
    "NCCL-ASYNC": "result--2048-32768-1node-4GPUs-NNN.txt",
    "NVSHMEM-SYNC": "result--2048-32768-1node-4GPUs-WWW.txt",
    "NVSHMEM-ASYNC": "result--2048-32768-1node-4GPUs-SSS.txt",
}

# ============================================================
# Read benchmark results
# ============================================================

def read_results(filename):
    results = {}

    with open(filename, "r") as f:
        text = f.read()

    # Flexible parser for benchmark lines such as:
    # RESULT: Matrix size: 2048    Time (seconds): 0.004
    #
    # It tolerates arbitrary spaces/tabs and does not depend on "RESULT:".
    pattern = (
        r"Matrix\s+size\s*:\s*(\d+)"
        r"[^\n]*?"
        r"Time\s*\(seconds\)\s*:\s*([0-9.eE+-]+)"
    )

    for n, t in re.findall(pattern, text):
        results[int(n)] = float(t)

    return results

data = {library: read_results(filename) for library, filename in files.items()}

print("\nDEBUG - Results detected:\n")
for library, results in data.items():
    print(f"{library:25s}: {sorted(results.keys())}")

# ============================================================
# Select matrix sizes available for all libraries
# ============================================================

common_sizes = set(data["MPI-SYNC"].keys())

for library in data:
    common_sizes &= set(data[library].keys())

matrix_sizes = sorted(common_sizes)

if not matrix_sizes:
    details = "\n".join(
        f"  {library}: {sorted(results.keys())}"
        for library, results in data.items()
    )
    raise RuntimeError(
        "No common matrix sizes were found in all result files.\n"
        "Detected matrix sizes by library:\n" + details
    )

# ============================================================
# Generic centered table
# ============================================================

def print_centered_table(headers, rows, widths):
    print("".join(f"{str(h):^{w}}" for h, w in zip(headers, widths)))
    for row in rows:
        print("".join(f"{str(v):^{w}}" for v, w in zip(row, widths)))


libraries = list(files.keys())
headers = ["N"] + libraries
widths = [10] + [max(16, len(name) + 3) for name in libraries]

# ============================================================
# Display extracted execution times
# ============================================================

print("\nExecution Time (seconds)\n")

rows = []
for size in matrix_sizes:
    rows.append(
        [size] + [f"{data[library][size]:.3f}" for library in libraries]
    )

print_centered_table(headers, rows, widths)

# ============================================================
# Plot 1 - Execution Time
# ============================================================

plt.figure(figsize=(11, 6))

for library in libraries:
    times = [data[library][size] for size in matrix_sizes]
    plt.plot(
        matrix_sizes,
        times,
        marker="o",
        linewidth=2,
        label=library
    )

plt.xlabel("Matrix Size (N × N)")
plt.ylabel("Execution Time (s)")
plt.title(
    "Matrix Multiplication — Execution Time\n"
    "1 Node / 4 GPUs — 1D Row-Block Decomposition"
)
plt.xticks(matrix_sizes)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0))
plt.tight_layout()

plt.savefig(
    "execution_time.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# Calculate Speedup relative to MPI-SYNC
# ============================================================

speedup = {}

for library in libraries:
    speedup[library] = [
        data["MPI-SYNC"][size] / data[library][size]
        for size in matrix_sizes
    ]

# ============================================================
# Display Speedup
# ============================================================

print("\nSpeedup relative to MPI-SYNC (MMM)\n")

rows = []
for i, size in enumerate(matrix_sizes):
    rows.append(
        [size] + [f"{speedup[library][i]:.3f}" for library in libraries]
    )

print_centered_table(headers, rows, widths)

# ============================================================
# Plot 2 - Speedup
# ============================================================

plt.figure(figsize=(11, 6))

for library in libraries:
    plt.plot(
        matrix_sizes,
        speedup[library],
        marker="o",
        linewidth=2,
        label=library
    )

plt.axhline(y=1.0, linestyle="--", linewidth=1)
plt.xlabel("Matrix Size (N × N)")
plt.ylabel("Speedup")
plt.title(
    "Matrix Multiplication — Speedup Relative to MPI-SYNC\n"
    "1 Node / 4 GPUs — 1D Row-Block Decomposition"
)
plt.xticks(matrix_sizes)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0))
plt.tight_layout()

plt.savefig(
    "speedup.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# Pairwise SYNC x ASYNC comparisons
# ============================================================

def print_comparison(title, sync_label, async_label):
    print(f"\n{title}\n")

    comparison_headers = [
        "N", sync_label, async_label, "Speedup", "Improvement"
    ]
    comparison_widths = [
        10,
        max(18, len(sync_label) + 3),
        max(18, len(async_label) + 3),
        12,
        15
    ]

    rows = []

    for size in matrix_sizes:
        sync_time = data[sync_label][size]
        async_time = data[async_label][size]
        async_speedup = sync_time / async_time
        improvement = ((sync_time - async_time) / sync_time) * 100.0

        rows.append([
            size,
            f"{sync_time:.3f}",
            f"{async_time:.3f}",
            f"{async_speedup:.3f}",
            f"{improvement:.2f}%"
        ])

    print_centered_table(
        comparison_headers,
        rows,
        comparison_widths
    )


print_comparison(
    "MPI-CHUNKED-ASYNC improvement over MPI-SYNC",
    "MPI-SYNC",
    "MPI-CHUNKED-ASYNC"
)

print_comparison(
    "CUDA-Aware-ASYNC improvement over CUDA-Aware-SYNC",
    "CUDA-Aware-SYNC",
    "CUDA-Aware-ASYNC"
)

# ============================================================
# NVSHMEM-SYNC comparison relative to MPI-SYNC
# ============================================================

print_comparison(
    "NVSHMEM-SYNC comparison with MPI-SYNC",
    "MPI-SYNC",
    "NVSHMEM-SYNC"
)

print_comparison(
    "NVSHMEM-ASYNC improvement over NVSHMEM-SYNC",
    "NVSHMEM-SYNC",
    "NVSHMEM-ASYNC"
)

# ============================================================
# Final information
# ============================================================

print("\nGenerated files:")
print("  execution_time.png")
print("  speedup.png")
