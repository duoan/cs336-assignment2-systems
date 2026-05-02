"""Profile FSDP all-gather overlap with torch.profiler.

Generates Chrome traces viewable in https://ui.perfetto.dev
and prints timing analysis of all-gather vs compute overlap.

Usage (Modal):
  modal run cs336_systems/fsdp_profile.py
"""

import base64
import os

import modal

app = modal.App("fsdp_profile")

VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 512
BATCH_SIZE = 4
WARMUP_STEPS = 3
WORLD_SIZE = 2

MODEL_CONFIG = dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32)

TRACE_DIR = "/tmp/fsdp_traces"


def _worker(rank, world_size):
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=device)
    torch.cuda.set_device(rank)

    from cs336_basics.model import BasicsTransformerLM
    from cs336_basics.nn_utils import cross_entropy
    from cs336_systems.fsdp import FullyShardedDataParallel

    torch.manual_seed(0)
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE, context_length=CONTEXT_LENGTH, **MODEL_CONFIG
    ).to(device)

    fsdp_model = FullyShardedDataParallel(model, compute_dtype=None)
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)

    local_bs = BATCH_SIZE // world_size
    torch.manual_seed(42)
    all_data = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH + 1), device=device)
    offset = rank * local_bs
    inputs = all_data[offset : offset + local_bs, :-1]
    targets = all_data[offset : offset + local_bs, 1:]

    def train_step():
        optimizer.zero_grad()
        logits = fsdp_model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        fsdp_model.finish_gradient_synchronization()
        optimizer.step()

    if rank == 0:
        print(f"[fsdp_profile] warmup={WARMUP_STEPS}, profile=1 step")

    for i in range(WARMUP_STEPS):
        train_step()
        if rank == 0:
            print(f"  warmup {i + 1}/{WARMUP_STEPS}")

    dist.barrier()

    os.makedirs(TRACE_DIR, exist_ok=True)
    trace_path = f"{TRACE_DIR}/fsdp_rank{rank}.json"

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        train_step()

    prof.export_chrome_trace(trace_path)

    if rank == 0:
        print(f"[fsdp_profile] rank {rank} trace: {os.path.getsize(trace_path) / 1e6:.1f} MB")

        table = prof.key_averages()

        nccl_total_us = 0
        nccl_items = []
        compute_total_us = 0
        compute_items = []

        for evt in table:
            name = evt.key
            cuda_us = getattr(evt, "device_time_total", 0) or getattr(evt, "cuda_time_total", 0)
            if cuda_us <= 0:
                continue
            if "nccl" in name.lower():
                nccl_total_us += cuda_us
                nccl_items.append((name, cuda_us, evt.count))
            elif any(k in name.lower() for k in ["gemm", "mm", "matmul", "addmm", "bmm"]):
                compute_total_us += cuda_us
                compute_items.append((name, cuda_us, evt.count))

        summary_lines = []
        summary_lines.append("=" * 70)
        summary_lines.append("  FSDP All-Gather Timing Analysis (rank 0)")
        summary_lines.append("=" * 70)

        summary_lines.append(f"\nTotal NCCL time: {nccl_total_us / 1000:.1f} ms")
        for name, us, count in sorted(nccl_items, key=lambda x: -x[1]):
            summary_lines.append(f"  {name:<50s} {us/1000:>8.1f} ms  (x{count})")

        summary_lines.append(f"\nTotal compute (matmul) time: {compute_total_us / 1000:.1f} ms")
        for name, us, count in sorted(compute_items, key=lambda x: -x[1])[:10]:
            summary_lines.append(f"  {name:<50s} {us/1000:>8.1f} ms  (x{count})")

        overlap_ratio = nccl_total_us / compute_total_us if compute_total_us > 0 else 0
        summary_lines.append(f"\nNCCL / Compute ratio: {overlap_ratio:.2f}")
        if overlap_ratio < 1.0:
            summary_lines.append("=> Communication < Compute: all-gathers can be fully hidden behind compute.")
        else:
            summary_lines.append("=> Communication > Compute: all-gathers are on the critical path.")

        summary_text = "\n".join(summary_lines)
        print(summary_text)

        with open(f"{TRACE_DIR}/summary.txt", "w") as f:
            f.write(summary_text)

    dist.barrier()
    dist.destroy_process_group()


def _run():
    import torch.multiprocessing as mp
    mp.spawn(_worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)

    trace_path = f"{TRACE_DIR}/fsdp_rank0.json"
    with open(trace_path, "rb") as f:
        trace_bytes = f.read()
    with open(f"{TRACE_DIR}/summary.txt") as f:
        summary = f.read()
    return trace_bytes, summary


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "einx", "einops", "jaxtyping")
    .run_commands(
        "mkdir -p /usr/local/lib/python3.12/site-packages/cs336_systems-0.0.0.dist-info"
        " && echo 'Metadata-Version: 2.1\\nName: cs336-systems\\nVersion: 0.0.0'"
        " > /usr/local/lib/python3.12/site-packages/cs336_systems-0.0.0.dist-info/METADATA"
    )
    .add_local_dir("cs336-basics/cs336_basics", remote_path="/root/cs336_basics")
    .add_local_dir("cs336_systems", remote_path="/root/cs336_systems")
)


@app.function(image=image, gpu="A100-80GB:2", timeout=900)
def run_remote():
    trace_bytes, summary = _run()
    return {
        "trace_b64": base64.b64encode(trace_bytes).decode("ascii"),
        "summary": summary,
    }


@app.local_entrypoint()
def modal_main():
    print(f"FSDP Profile: XL model, {WORLD_SIZE} GPUs ...")
    result = run_remote.remote()

    trace_bytes = base64.b64decode(result["trace_b64"])
    summary = result["summary"]

    os.makedirs("benchmark_results", exist_ok=True)
    path = "benchmark_results/fsdp_trace.json"
    with open(path, "wb") as f:
        f.write(trace_bytes)
    print(f"\nTrace saved to {path} ({len(trace_bytes) / 1e6:.1f} MB)")
    print("Open in https://ui.perfetto.dev to view\n")
    print(summary)
