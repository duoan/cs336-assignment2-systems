"""Profile DDP implementations with torch.profiler to visualize compute/comm overlap.

Generates Chrome traces viewable in chrome://tracing or https://ui.perfetto.dev

Usage (Modal):
  modal run cs336_systems/ddp_profile.py --impl naive
  modal run cs336_systems/ddp_profile.py --impl overlap
"""

import base64
import os

import modal
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

app = modal.App("ddp_profile")

VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 512
BATCH_SIZE = 4
WARMUP_STEPS = 3
WORLD_SIZE = 2

MODEL_CONFIG = dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32)

TRACE_DIR = "/tmp/ddp_traces"


def _worker(rank, world_size, impl="naive"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    from cs336_basics.model import BasicsTransformerLM
    from cs336_basics.nn_utils import cross_entropy

    torch.manual_seed(0)
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE, context_length=CONTEXT_LENGTH, **MODEL_CONFIG
    ).to(device)

    from cs336_systems.my_ddp_impl import FlatDDP, NaiveDDP, OverlapDDP

    DDP_CLS = {"naive": NaiveDDP, "flat": FlatDDP, "overlap": OverlapDDP}
    ddp_model = DDP_CLS[impl](model)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)

    local_bs = BATCH_SIZE // world_size
    torch.manual_seed(42)
    all_data = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH + 1), device=device)
    offset = rank * local_bs
    inputs = all_data[offset : offset + local_bs, :-1]
    targets = all_data[offset : offset + local_bs, 1:]

    def train_step():
        optimizer.zero_grad()
        logits = ddp_model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()

    if rank == 0:
        print(f"[profile] impl={impl}, warmup={WARMUP_STEPS}, profile=1 step")

    for i in range(WARMUP_STEPS):
        train_step()
        if rank == 0:
            print(f"  warmup {i + 1}/{WARMUP_STEPS}")

    dist.barrier()

    os.makedirs(TRACE_DIR, exist_ok=True)
    trace_path = f"{TRACE_DIR}/{impl}_rank{rank}.json"

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
        print(f"[profile] rank {rank} trace: {os.path.getsize(trace_path) / 1e6:.1f} MB")

    dist.barrier()
    dist.destroy_process_group()


def _run(impl="naive"):
    mp.spawn(_worker, args=(WORLD_SIZE, impl), nprocs=WORLD_SIZE, join=True)
    trace_path = f"{TRACE_DIR}/{impl}_rank0.json"
    with open(trace_path, "rb") as f:
        return f.read()


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


@app.function(image=image, gpu="A100-80GB:2", timeout=900)
def run_remote(impl: str = "naive"):
    trace_bytes = _run(impl)
    return base64.b64encode(trace_bytes).decode("ascii")


@app.local_entrypoint()
def modal_main(impl: str = "naive"):
    print(f"Dispatching profile to Modal (impl={impl}, {WORLD_SIZE} GPUs) ...")
    encoded = run_remote.remote(impl)
    trace_bytes = base64.b64decode(encoded)

    os.makedirs("benchmark_results", exist_ok=True)
    path = f"benchmark_results/{impl}_ddp_trace.json"
    with open(path, "wb") as f:
        f.write(trace_bytes)
    print(f"Trace saved to {path} ({len(trace_bytes) / 1e6:.1f} MB)")
    print(f"Open in https://ui.perfetto.dev to view")
