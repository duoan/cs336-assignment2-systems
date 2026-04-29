"""Benchmark DDP implementations with the XL language model (1 node x 2 GPUs).

Measures:
  - Total time per training step
  - Time spent communicating gradients (all-reduce)
  - Communication fraction

Usage (Modal, nccl/2×GPU):
  modal run cs336_systems/naive_ddp_benchmark.py --impl naive
  modal run cs336_systems/naive_ddp_benchmark.py --impl flat

Local:
  python cs336_systems/naive_ddp_benchmark.py --impl naive
"""

import argparse
import json
import os
import time

import modal
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

app = modal.App("ddp_benchmark")

BACKEND = "nccl" if torch.cuda.is_available() else "gloo"

MODEL_CONFIGS = {
    "small":  dict(d_model=768,   d_ff=3072,   num_layers=12, num_heads=12),
    "medium": dict(d_model=1024,  d_ff=4096,   num_layers=24, num_heads=16),
    "large":  dict(d_model=1280,  d_ff=5120,   num_layers=36, num_heads=20),
    "xl":     dict(d_model=2560,  d_ff=10240,  num_layers=32, num_heads=32),
}

VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 512
BATCH_SIZE = 4
WARMUP_STEPS = 5
BENCH_STEPS = 10
WORLD_SIZE = 2


def _worker(rank, world_size, result_queue, impl="naive"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    if backend == "nccl":
        torch.cuda.set_device(rank)
    if rank == 0:
        print(f"[init] Backend: {backend}, device: {device}, GPUs available: {torch.cuda.device_count()}")
        print(f"[init] Config: model=xl, batch_size={BATCH_SIZE}, context_length={CONTEXT_LENGTH}, world_size={world_size}")

    from cs336_basics.model import BasicsTransformerLM
    from cs336_basics.nn_utils import cross_entropy

    cfg = MODEL_CONFIGS["xl"]
    if rank == 0:
        print(f"[init] Creating XL model: {cfg}")
    torch.manual_seed(0)
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        **cfg,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"[init] Model created: {num_params:,} params ({num_params/1e9:.2f}B), on {device}")

    from cs336_systems.my_ddp_impl import NaiveDDP, FlatDDP, OverlapDDP
    DDP_CLS = {
        "naive": NaiveDDP, 
        "flat": FlatDDP,
        "overlap": OverlapDDP,
    }
    if impl not in DDP_CLS:
        raise ValueError(f"Unknown impl '{impl}', choose from {list(DDP_CLS)}")
    ddp_model = DDP_CLS[impl](model)

    if rank == 0:
        print(f"[init] DDP wrapper: {impl} ({DDP_CLS[impl].__name__}), broadcasting params from rank 0")

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)

    global_bs = BATCH_SIZE
    assert global_bs % world_size == 0
    local_bs = global_bs // world_size

    torch.manual_seed(42)
    all_data = torch.randint(0, VOCAB_SIZE, (global_bs, CONTEXT_LENGTH + 1), device=device)
    offset = rank * local_bs
    local_data = all_data[offset : offset + local_bs]
    inputs = local_data[:, :-1]
    targets = local_data[:, 1:]

    if rank == 0:
        print(f"[init] Data ready: global_bs={global_bs}, local_bs={local_bs}, inputs={tuple(inputs.shape)}, device={inputs.device}")

    use_cuda = backend == "nccl"

    def train_step():
        """Returns (total_s, comm_s)."""
        if use_cuda:
            torch.cuda.synchronize()
        t_total_start = time.perf_counter()

        optimizer.zero_grad()
        logits = ddp_model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()

        if use_cuda:
            torch.cuda.synchronize()
        t_comm_start = time.perf_counter()

        ddp_model.finish_gradient_synchronization()

        if use_cuda:
            torch.cuda.synchronize()
        t_comm_end = time.perf_counter()

        optimizer.step()

        if use_cuda:
            torch.cuda.synchronize()
        t_total_end = time.perf_counter()

        return t_total_end - t_total_start, t_comm_end - t_comm_start

    if rank == 0:
        print(f"[warmup] Running {WARMUP_STEPS} warmup steps ...")
    for i in range(WARMUP_STEPS):
        train_step()
        if rank == 0:
            print(f"  warmup step {i+1}/{WARMUP_STEPS} done")

    dist.barrier()
    if rank == 0:
        print(f"[bench] Running {BENCH_STEPS} benchmark steps ...")

    total_times = []
    comm_times = []
    for _ in range(BENCH_STEPS):
        t_total, t_comm = train_step()
        total_times.append(t_total)
        comm_times.append(t_comm)

    gathered = [None] * world_size
    dist.all_gather_object(gathered, {"total": total_times, "comm": comm_times})

    if rank == 0:
        all_total = np.array([g["total"] for g in gathered])
        all_comm = np.array([g["comm"] for g in gathered])
        avg_total = all_total.mean(axis=0)
        avg_comm = all_comm.mean(axis=0)

        result_queue.put({
            "total_median_ms": float(np.median(avg_total) * 1000),
            "total_mean_ms": float(np.mean(avg_total) * 1000),
            "total_std_ms": float(np.std(avg_total) * 1000),
            "comm_median_ms": float(np.median(avg_comm) * 1000),
            "comm_mean_ms": float(np.mean(avg_comm) * 1000),
            "comm_std_ms": float(np.std(avg_comm) * 1000),
            "comm_fraction": float(np.median(avg_comm) / np.median(avg_total)),
        })

    dist.barrier()
    dist.destroy_process_group()


def _run(impl="naive"):
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    mp.spawn(_worker, args=(WORLD_SIZE, result_queue, impl), nprocs=WORLD_SIZE, join=True)
    result = result_queue.get()
    result["impl"] = impl
    return result


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "einx", "einops", "jaxtyping")
    .run_commands(
        "mkdir -p /usr/local/lib/python3.12/site-packages/cs336_systems-0.0.0.dist-info"
        " && echo 'Metadata-Version: 2.1\\nName: cs336-systems\\nVersion: 0.0.0'"
        " > /usr/local/lib/python3.12/site-packages/cs336_systems-0.0.0.dist-info/METADATA"
    )
    .add_local_python_source("cs336_basics")
    .add_local_python_source("cs336_systems")
)


@app.function(image=image, gpu="A100-80GB:2", timeout=600)
def run_remote(impl: str = "naive"):
    return _run(impl)


@app.local_entrypoint()
def modal_main(impl: str = "naive"):
    print(f"Dispatching to Modal (nccl, {WORLD_SIZE} GPUs, impl={impl}) ...")
    result = run_remote.remote(impl)
    _print_result(result)


def _print_result(r):
    impl = r.get("impl", "naive")
    print("\n" + "=" * 60)
    print(f"DDP Benchmark [{impl}]  (XL model, {WORLD_SIZE} GPUs)")
    print(f"  batch_size={BATCH_SIZE}, context_length={CONTEXT_LENGTH}")
    print(f"  warmup={WARMUP_STEPS}, bench_steps={BENCH_STEPS}")
    print("=" * 60)
    print(f"  Total step (median):       {r['total_median_ms']:8.1f} ms")
    print(f"  Total step (mean ± std):   {r['total_mean_ms']:8.1f} ± {r['total_std_ms']:.1f} ms")
    print(f"  Gradient comm (median):    {r['comm_median_ms']:8.1f} ms")
    print(f"  Gradient comm (mean ± std):{r['comm_mean_ms']:8.1f} ± {r['comm_std_ms']:.1f} ms")
    print(f"  Communication fraction:    {r['comm_fraction']*100:8.1f}%")
    print("=" * 60)

    os.makedirs("benchmark_results", exist_ok=True)
    path = f"benchmark_results/{impl}_ddp_benchmark.json"
    with open(path, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["naive", "flat", "overlap"], default="naive")
    args = parser.parse_args()
    result = _run(args.impl)
    _print_result(result)


if __name__ == "__main__":
    main()
