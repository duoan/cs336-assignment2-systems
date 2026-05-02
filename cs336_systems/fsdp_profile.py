"""Profile FSDP all-gather overlap with Nsight Systems.

The Modal entrypoint launches an inner worker process under ``nsys profile``.
That lets Nsight Systems observe the full multiprocessing/NCCL/CUDA process
tree and collect the real timeline rather than only torch.profiler aggregates.

Usage (Modal):
  modal run cs336_systems/fsdp_profile.py

The local output is written to:
  benchmark_results/fsdp_profile.nsys-rep
  benchmark_results/fsdp_profile_nsys.log
  benchmark_results/fsdp_profile_nsys_stats.txt
"""

import argparse
import base64
import os
import shutil
import subprocess
import sys

import modal

app = modal.App("fsdp_profile")

VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 512
BATCH_SIZE = 4
WARMUP_STEPS = 3
WORLD_SIZE = 2

MODEL_CONFIG = dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32)

PROFILE_DIR = "/tmp/fsdp_nsys"
NSYS_OUTPUT_BASE = f"{PROFILE_DIR}/fsdp_profile"
NSYS_REPORT_PATH = f"{NSYS_OUTPUT_BASE}.nsys-rep"


def _worker(rank, world_size):
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    import torch.cuda.nvtx as nvtx
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
        nvtx.range_push("zero_grad")
        optimizer.zero_grad()
        nvtx.range_pop()

        nvtx.range_push("forward")
        logits = fsdp_model(inputs)
        nvtx.range_pop()

        nvtx.range_push("loss")
        loss = cross_entropy(logits, targets)
        nvtx.range_pop()

        nvtx.range_push("backward")
        loss.backward()
        nvtx.range_pop()

        nvtx.range_push("finish_gradient_synchronization")
        fsdp_model.finish_gradient_synchronization()
        nvtx.range_pop()

        nvtx.range_push("optimizer_step")
        optimizer.step()
        nvtx.range_pop()

    if rank == 0:
        print(f"[fsdp_profile] warmup={WARMUP_STEPS}, nsys capture=1 step")

    for i in range(WARMUP_STEPS):
        train_step()
        if rank == 0:
            print(f"  warmup {i + 1}/{WARMUP_STEPS}")

    dist.barrier()
    torch.cuda.synchronize()

    if rank == 0:
        print("[fsdp_profile] starting CUDA profiler capture")

    # ``nsys profile --capture-range=cudaProfilerApi`` records only the region
    # bracketed by cudaProfilerStart/Stop. This keeps warmup out of the trace
    # while still allowing nsys to launch and observe the whole process tree.
    fsdp_model.clear_wait_profile()
    torch.cuda.cudart().cudaProfilerStart()
    nvtx.range_push(f"rank_{rank}_profiled_train_step")
    train_step()
    nvtx.range_pop()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.cudart().cudaProfilerStop()

    if rank == 0:
        print("[fsdp_profile] stopped CUDA profiler capture")
        print(fsdp_model.wait_profile_summary())

    dist.barrier()
    dist.destroy_process_group()


def _run_worker_profile():
    import torch.multiprocessing as mp

    mp.spawn(_worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)


def _run_nsys_profile():
    os.makedirs(PROFILE_DIR, exist_ok=True)
    nsys = shutil.which("nsys")
    if nsys is None:
        raise RuntimeError(
            "Nsight Systems CLI (`nsys`) was not found in the Modal image. "
            "Install Nsight Systems or use a CUDA image that includes nsys."
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root" + os.pathsep + env.get("PYTHONPATH", "")
    env["FSDP_PROFILE_WAITS"] = "1"

    cmd = [
        nsys,
        "profile",
        "--force-overwrite=true",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--cuda-memory-usage=true",
        "--sample=none",
        "--trace=cuda,nvtx,osrt,cublas,cudnn",
        f"--output={NSYS_OUTPUT_BASE}",
        sys.executable,
        "-m",
        "cs336_systems.fsdp_profile",
        "--worker-profile",
    ]

    completed = subprocess.run(
        cmd,
        cwd="/root",
        env=env,
        text=True,
        capture_output=True,
    )
    profile_log = (
        "$ " + " ".join(cmd) + "\n\n"
        "=== STDOUT ===\n" + completed.stdout + "\n"
        "=== STDERR ===\n" + completed.stderr
    )
    if completed.returncode != 0:
        raise RuntimeError(profile_log)

    stats_completed = subprocess.run(
        [nsys, "stats", NSYS_REPORT_PATH],
        cwd="/root",
        env=env,
        text=True,
        capture_output=True,
    )
    stats_log = (
        "$ nsys stats " + NSYS_REPORT_PATH + "\n\n"
        "=== STDOUT ===\n" + stats_completed.stdout + "\n"
        "=== STDERR ===\n" + stats_completed.stderr
    )

    with open(NSYS_REPORT_PATH, "rb") as f:
        report_bytes = f.read()

    return report_bytes, profile_log, stats_log


image = (
    # The slim Modal image has CUDA at runtime but does not include Nsight
    # Systems. NVIDIA's PyTorch image includes the CUDA toolkit, PyTorch, NCCL,
    # and the ``nsys`` CLI needed by the subprocess profiler wrapper above.
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.12-py3")
    .pip_install("einx", "einops", "jaxtyping")
    .run_commands(
        "mkdir -p"
        " /usr/local/lib/python3.12/site-packages/cs336_systems-0.0.0.dist-info"
        " /usr/local/lib/python3.12/dist-packages/cs336_systems-0.0.0.dist-info"
        " && printf 'Metadata-Version: 2.1\\nName: cs336-systems\\nVersion: 0.0.0\\n'"
        " | tee"
        " /usr/local/lib/python3.12/site-packages/cs336_systems-0.0.0.dist-info/METADATA"
        " /usr/local/lib/python3.12/dist-packages/cs336_systems-0.0.0.dist-info/METADATA"
        " >/dev/null"
    )
    .add_local_dir("cs336-basics/cs336_basics", remote_path="/root/cs336_basics")
    .add_local_dir("cs336_systems", remote_path="/root/cs336_systems")
)


@app.function(image=image, gpu="A100-80GB:2", timeout=900)
def run_remote():
    output = subprocess.check_output(["nvidia-smi"], text=True)
    print(output)

    report_bytes, profile_log, stats_log = _run_nsys_profile()
    return {
        "report_b64": base64.b64encode(report_bytes).decode("ascii"),
        "profile_log": profile_log,
        "stats_log": stats_log,
    }


@app.local_entrypoint()
def modal_main():
    print(f"FSDP Profile: XL model, {WORLD_SIZE} GPUs ...")
    result = run_remote.remote()

    report_bytes = base64.b64decode(result["report_b64"])

    os.makedirs("benchmark_results", exist_ok=True)
    report_path = "benchmark_results/fsdp_profile.nsys-rep"
    log_path = "benchmark_results/fsdp_profile_nsys.log"
    stats_path = "benchmark_results/fsdp_profile_nsys_stats.txt"

    with open(report_path, "wb") as f:
        f.write(report_bytes)
    with open(log_path, "w") as f:
        f.write(result["profile_log"])
    with open(stats_path, "w") as f:
        f.write(result["stats_log"])

    print(f"\nNsight report saved to {report_path} ({len(report_bytes) / 1e6:.1f} MB)")
    print(f"Nsight profile log saved to {log_path}")
    print(f"Nsight stats saved to {stats_path}\n")
    print(result["stats_log"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker-profile",
        action="store_true",
        help="Run the inner multiprocessing training job. Intended to be launched by nsys.",
    )
    args = parser.parse_args()
    if args.worker_profile:
        _run_worker_profile()
