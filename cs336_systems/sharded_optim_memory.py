"""Profile peak GPU memory and training speed with vs without optimizer state sharding.

Measures:
  Memory at three points: after init, before step, after step.
  Timing: total step, forward+backward, gradient sync, optimizer step.

Usage (Modal):
  modal run cs336_systems/sharded_optim_memory.py
"""

import json
import os
import time

import humanfriendly
import modal

app = modal.App("sharded_optim_memory")

VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 512
BATCH_SIZE = 4
WARMUP_STEPS = 5
BENCH_STEPS = 10
WORLD_SIZE = 2
MODEL_CONFIG = dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32)


def _fmt(nbytes: int) -> str:
    return humanfriendly.format_size(nbytes, binary=True)


def _worker(rank, world_size, result_queue, use_sharded: bool):
    import numpy as np
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    from cs336_basics.model import BasicsTransformerLM
    from cs336_basics.nn_utils import cross_entropy
    from cs336_systems.my_ddp_impl import OverlapDDP
    from cs336_systems.optimizers import ShardedOptimizer

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # ---- Model init ----
    torch.manual_seed(0)
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE, context_length=CONTEXT_LENGTH, **MODEL_CONFIG
    ).to(device)
    ddp_model = OverlapDDP(model)

    if use_sharded:
        optimizer = ShardedOptimizer(
            ddp_model.parameters(), torch.optim.AdamW, lr=1e-4
        )
        label = "sharded"
    else:
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)
        label = "standard"

    torch.cuda.synchronize()
    mem_after_init = torch.cuda.max_memory_allocated(device)

    # ---- Data ----
    local_bs = BATCH_SIZE // world_size
    torch.manual_seed(42)
    all_data = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH + 1), device=device)
    offset = rank * local_bs
    inputs = all_data[offset : offset + local_bs, :-1]
    targets = all_data[offset : offset + local_bs, 1:]

    def timed_step():
        """Returns (total_s, fwd_bwd_s, sync_s, optim_s)."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad()
        logits = ddp_model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        ddp_model.finish_gradient_synchronization()

        torch.cuda.synchronize()
        t2 = time.perf_counter()

        optimizer.step()

        torch.cuda.synchronize()
        t3 = time.perf_counter()

        return t3 - t0, t1 - t0, t2 - t1, t3 - t2

    # ---- Warmup ----
    if rank == 0:
        print(f"[{label}] warmup {WARMUP_STEPS} steps ...")
    for i in range(WARMUP_STEPS):
        timed_step()
        if rank == 0:
            print(f"  warmup {i + 1}/{WARMUP_STEPS}")

    # ---- Memory measurement (single step) ----
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    mem_baseline = torch.cuda.memory_allocated(device)

    optimizer.zero_grad()
    logits = ddp_model(inputs)
    loss = cross_entropy(logits, targets)
    loss.backward()
    ddp_model.finish_gradient_synchronization()

    torch.cuda.synchronize()
    mem_before_step = torch.cuda.max_memory_allocated(device)

    optimizer.step()

    torch.cuda.synchronize()
    mem_after_step = torch.cuda.max_memory_allocated(device)

    # ---- Timing benchmark ----
    dist.barrier()
    if rank == 0:
        print(f"[{label}] benchmarking {BENCH_STEPS} steps ...")

    totals, fwd_bwds, syncs, optims = [], [], [], []
    for _ in range(BENCH_STEPS):
        t_total, t_fb, t_sync, t_optim = timed_step()
        totals.append(t_total)
        fwd_bwds.append(t_fb)
        syncs.append(t_sync)
        optims.append(t_optim)

    # Gather from all ranks and average
    local_times = {"total": totals, "fwd_bwd": fwd_bwds, "sync": syncs, "optim": optims}
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_times)

    num_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    if rank == 0:
        avg = {}
        for key in ("total", "fwd_bwd", "sync", "optim"):
            arr = np.array([g[key] for g in gathered]).mean(axis=0)
            avg[key] = arr

        result_queue.put({
            "label": label,
            "num_params": num_params,
            "param_bytes": param_bytes,
            "mem_after_init": mem_after_init,
            "mem_before_step": mem_before_step,
            "mem_after_step": mem_after_step,
            "mem_baseline": mem_baseline,
            "total_median_ms": float(np.median(avg["total"]) * 1000),
            "total_mean_ms": float(np.mean(avg["total"]) * 1000),
            "total_std_ms": float(np.std(avg["total"]) * 1000),
            "fwd_bwd_median_ms": float(np.median(avg["fwd_bwd"]) * 1000),
            "sync_median_ms": float(np.median(avg["sync"]) * 1000),
            "optim_median_ms": float(np.median(avg["optim"]) * 1000),
        })

    dist.barrier()
    dist.destroy_process_group()


def _run(use_sharded: bool):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.spawn(_worker, args=(WORLD_SIZE, q, use_sharded), nprocs=WORLD_SIZE, join=True)
    return q.get()


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "einx", "einops", "jaxtyping", "humanfriendly")
    .run_commands(
        "mkdir -p /usr/local/lib/python3.12/site-packages/cs336_systems-0.0.0.dist-info"
        " && echo 'Metadata-Version: 2.1\\nName: cs336-systems\\nVersion: 0.0.0'"
        " > /usr/local/lib/python3.12/site-packages/cs336_systems-0.0.0.dist-info/METADATA"
    )
    .add_local_python_source("cs336_basics")
    .add_local_python_source("cs336_systems")
)


@app.function(image=image, gpu="A100-80GB:2", timeout=900)
def run_remote():
    r_std = _run(use_sharded=False)
    r_shard = _run(use_sharded=True)
    return {"standard": r_std, "sharded": r_shard}


def _print_results(results: dict):
    for key in ("standard", "sharded"):
        r = results[key]
        print(f"\n{'=' * 64}")
        print(f"  Optimizer: {r['label']}  |  XL model, {WORLD_SIZE} GPUs")
        print(f"  Model: {r['num_params']:,} params ({_fmt(r['param_bytes'])} fp32)")
        print(f"{'=' * 64}")
        print(f"  --- Memory ---")
        print(f"  After init (model+optimizer): {_fmt(r['mem_after_init']):>12s}")
        print(f"  Before optimizer.step:        {_fmt(r['mem_before_step']):>12s}")
        print(f"  After optimizer.step:         {_fmt(r['mem_after_step']):>12s}")
        print(f"  --- Timing (median over {BENCH_STEPS} steps) ---")
        print(f"  Total step:          {r['total_median_ms']:>8.1f} ms")
        print(f"    Forward + backward:{r['fwd_bwd_median_ms']:>8.1f} ms")
        print(f"    Gradient sync:     {r['sync_median_ms']:>8.1f} ms")
        print(f"    Optimizer step:    {r['optim_median_ms']:>8.1f} ms")
        print(f"{'=' * 64}")

    std = results["standard"]
    shard = results["sharded"]
    pb = std["param_bytes"]

    print(f"\n--- Theoretical Memory Breakdown (fp32, {_fmt(pb)} params) ---")
    print(f"  Parameters:                {_fmt(pb):>12s}")
    print(f"  Gradients:                 {_fmt(pb):>12s}")
    print(f"  AdamW states (m + v):      {_fmt(2 * pb):>12s}  (standard)")
    print(f"  AdamW states sharded (÷{WORLD_SIZE}): {_fmt(pb):>12s}  (per rank)")
    print(f"  Expected saving:           {_fmt(pb):>12s}")
    print(f"  Measured saving (after step): {_fmt(std['mem_after_step'] - shard['mem_after_step']):>12s}")

    print(f"\n--- Speed Comparison ---")
    print(f"  Standard total step:   {std['total_median_ms']:>8.1f} ms")
    print(f"  Sharded  total step:   {shard['total_median_ms']:>8.1f} ms")
    overhead = shard['total_median_ms'] - std['total_median_ms']
    pct = overhead / std['total_median_ms'] * 100
    print(f"  Overhead:              {overhead:>+8.1f} ms ({pct:+.1f}%)")
    print(f"  (from broadcast in sharded optimizer step: "
          f"{shard['optim_median_ms']:.1f} ms vs {std['optim_median_ms']:.1f} ms)")


@app.local_entrypoint()
def modal_main():
    print(f"Profiling memory + speed (XL model, {WORLD_SIZE} GPUs) ...")
    results = run_remote.remote()
    _print_results(results)

    os.makedirs("benchmark_results", exist_ok=True)
    path = "benchmark_results/sharded_optim_memory.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")
