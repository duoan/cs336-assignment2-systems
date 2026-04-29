"""Benchmark all-reduce latency across data sizes and world sizes.

Problem (distributed_communication_single_node):
  data sizes: 1 MB, 10 MB, 100 MB, 1 GB  (float32 tensors)
  world sizes: 2, 4, 6

Usage (local, gloo/CPU):
  uv run python cs336_systems/all_reduce_benchmark.py

Usage (Modal, nccl/multi-GPU):
  modal run cs336_systems/all_reduce_benchmark.py
"""

import json
import os
import time

import modal
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

app = modal.App("dist_comm_bench")

BACKEND = "gloo" if modal.is_local() else "nccl"

WORLD_SIZES = [2, 4, 6]
INPUT_SIZES = {
    "1MB": 1_024 * 1_024 // 4,
    "10MB": 10 * 1_024 * 1_024 // 4,
    "100MB": 100 * 1_024 * 1_024 // 4,
    "1GB": 1_024 * 1_024 * 1_024 // 4,
}
NUM_WARMUP = 5
NUM_ITERS = 20

OUTPUT_DIR = "benchmark_results"

# ---------------------------------------------------------------------------
# Modal image (only used when running on Modal)
# ---------------------------------------------------------------------------
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "numpy")


# ---------------------------------------------------------------------------
# Worker (spawned by mp.spawn — re-imports module, so BACKEND is set correctly)
# ---------------------------------------------------------------------------
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(BACKEND, rank=rank, world_size=world_size)
    if BACKEND == "nccl":
        torch.cuda.set_device(rank)


def worker(rank, world_size, n_elements, result_queue):
    setup(rank, world_size)
    use_cuda = BACKEND == "nccl"
    device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")

    data = torch.randn(n_elements, dtype=torch.float32, device=device)

    for _ in range(NUM_WARMUP):
        dist.all_reduce(data, async_op=False)
        if use_cuda:
            torch.cuda.synchronize()

    dist.barrier()
    times = []
    for _ in range(NUM_ITERS):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(data, async_op=False)
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    gathered = [None] * world_size
    dist.all_gather_object(gathered, times)

    if rank == 0:
        all_times = np.array(gathered)
        avg_across_ranks = all_times.mean(axis=0)
        result_queue.put(
            {
                "median_ms": float(np.median(avg_across_ranks)),
                "mean_ms": float(np.mean(avg_across_ranks)),
                "std_ms": float(np.std(avg_across_ranks)),
            }
        )

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Core benchmark loop (shared by local and remote)
# ---------------------------------------------------------------------------
def _run_all():
    """Run every (world_size, data_size) config and return results dict."""
    print(f"Backend: {BACKEND}")
    if BACKEND == "nccl":
        print(f"GPUs available: {torch.cuda.device_count()}")

    results = {}
    for ws in WORLD_SIZES:
        for label, n_elem in INPUT_SIZES.items():
            print(f"  world_size={ws}, data_size={label} ...", end=" ", flush=True)
            ctx = mp.get_context("spawn")
            q = ctx.Queue()
            mp.spawn(worker, args=(ws, n_elem, q), nprocs=ws, join=True)
            r = q.get()
            results[f"{ws}_{label}"] = r
            print(f"median={r['median_ms']:.3f} ms")

    return results


# ---------------------------------------------------------------------------
# Post-processing: table + plot  (always runs locally)
# ---------------------------------------------------------------------------
def _parse_results(raw: dict):
    """Convert '2_1MB' keys back to (int, str) tuples."""
    out = {}
    for key, val in raw.items():
        ws_str, label = key.split("_", 1)
        out[(int(ws_str), label)] = val
    return out


def make_table(results, output_dir):
    size_labels = list(INPUT_SIZES.keys())
    header = "World size | " + " | ".join(f"{s:>10s}" for s in size_labels)
    sep = "-" * len(header)
    lines = [header, sep]
    for ws in WORLD_SIZES:
        vals = [f"{results[(ws, sl)]['median_ms']:10.3f}" for sl in size_labels]
        lines.append(f"{ws:>10d} | " + " | ".join(vals))
    table_str = "\n".join(lines)
    print("\n" + table_str + "\n")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "all_reduce_table.txt"), "w") as f:
        f.write(table_str + "\n")


def make_plot(results, output_dir, backend_label=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size_labels = list(INPUT_SIZES.keys())
    x = np.arange(len(size_labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, ws in enumerate(WORLD_SIZES):
        medians = [results[(ws, sl)]["median_ms"] for sl in size_labels]
        stds = [results[(ws, sl)]["std_ms"] for sl in size_labels]
        ax.bar(x + i * width, medians, width, yerr=stds, label=f"{ws} GPUs", capsize=3)

    title = "All-Reduce Benchmark"
    if backend_label:
        title += f" ({backend_label})"
    ax.set_xlabel("Data Size")
    ax.set_ylabel("All-Reduce Latency (ms)")
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels(size_labels)
    ax.legend()
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "all_reduce_benchmark.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}")


def _save_and_plot(raw, backend_label=""):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "all_reduce_results.json"), "w") as f:
        json.dump(raw, f, indent=2)

    results = _parse_results(raw)
    make_table(results, OUTPUT_DIR)
    make_plot(results, OUTPUT_DIR, backend_label=backend_label)


# ---------------------------------------------------------------------------
# Modal remote entry  (modal run → NCCL on multi-GPU)
# ---------------------------------------------------------------------------
@app.function(image=image, gpu="H100:6", timeout=600)
def run_remote():
    return _run_all()


@app.local_entrypoint()
def modal_main():
    print("Dispatching to Modal (NCCL, multi-GPU) ...")
    raw = run_remote.remote()
    _save_and_plot(raw, backend_label="NCCL, single node")


# ---------------------------------------------------------------------------
# Plain python entry  (python script.py → gloo/CPU locally)
# ---------------------------------------------------------------------------
def main():
    raw = _run_all()
    _save_and_plot(raw, backend_label="Gloo, CPU")


if __name__ == "__main__":
    main()
