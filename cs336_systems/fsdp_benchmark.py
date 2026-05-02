"""Benchmark FSDP vs DDP: peak memory and training speed.

Compares three modes:
  1. DDP (OverlapDDP) + standard AdamW
  2. FSDP fp32 + AdamW
  3. FSDP fp16 + AdamW  (mixed-precision)

Also supports a prefetch-depth experiment:
  DDP vs FSDP fp32 with prefetch depth 0, 1, 2, and 4.

Standard configuration: 1 node, 2 GPUs, XL model.

Usage (Modal):
  modal run cs336_systems/fsdp_benchmark.py                         # run main 3
  modal run cs336_systems/fsdp_benchmark.py --mode ddp              # run single
  modal run cs336_systems/fsdp_benchmark.py --experiment prefetch   # run prefetch sweep
"""

import json
import os
import time

import humanfriendly
import modal

app = modal.App("fsdp-benchmark")

VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 512
BATCH_SIZE = 4
WARMUP_STEPS = 3
BENCH_STEPS = 8
WORLD_SIZE = 2
MODEL_CONFIG = dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32)


def _fmt(nbytes: int) -> str:
    return humanfriendly.format_size(nbytes, binary=True)


def _worker(rank, world_size, result_queue, mode: str):
    """mode: 'ddp' | 'fsdp_fp32' | 'fsdp_fp16' | 'fsdp_prefetch_<depth>'"""
    import numpy as np
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=device)
    torch.cuda.set_device(rank)

    from cs336_basics.model import BasicsTransformerLM
    from cs336_basics.nn_utils import cross_entropy

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # ---- Build model + wrapper ----
    torch.manual_seed(0)
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE, context_length=CONTEXT_LENGTH, **MODEL_CONFIG
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    param_bytes = num_params * 4  # fp32

    use_autocast = mode == "fsdp_fp16"

    if mode == "ddp":
        from cs336_systems.my_ddp_impl import OverlapDDP
        wrapped = OverlapDDP(model)
        label = "DDP + AdamW"
    elif mode in ("fsdp_fp32", "fsdp_fp16") or mode.startswith("fsdp_prefetch_"):
        from cs336_systems.fsdp import FullyShardedDataParallel
        prefetch_depth = 1
        if mode.startswith("fsdp_prefetch_"):
            prefetch_depth = int(mode.rsplit("_", 1)[1])
        wrapped = FullyShardedDataParallel(
            model,
            compute_dtype=torch.float16 if mode == "fsdp_fp16" else None,
            prefetch_depth=prefetch_depth,
        )
        if mode == "fsdp_fp16":
            label = "FSDP fp16 + AdamW"
        elif mode.startswith("fsdp_prefetch_"):
            label = f"FSDP fp32 prefetch={prefetch_depth}"
        else:
            label = "FSDP fp32 + AdamW"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    local_param_count = sum(p.numel() for p in wrapped.parameters())
    local_param_bytes = sum(p.numel() * p.element_size() for p in wrapped.parameters())
    local_param_debug = {
        "rank": rank,
        "param_count": local_param_count,
        "param_bytes": local_param_bytes,
    }
    gathered_param_debug = [None] * world_size
    dist.all_gather_object(gathered_param_debug, local_param_debug)
    if rank == 0:
        print(f"[{label}] parameter residency after wrapping:")
        print(f"  logical full model: {num_params:,} params ({_fmt(param_bytes)})")
        for item in gathered_param_debug:
            pct = item["param_count"] / num_params * 100
            print(
                f"  rank {item['rank']}: "
                f"{item['param_count']:,} params ({pct:.1f}% of full), "
                f"{_fmt(item['param_bytes'])}"
            )

    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_autocast)

    torch.cuda.synchronize()
    mem_after_wrap_current = torch.cuda.memory_allocated(device)
    mem_after_wrap_peak = torch.cuda.max_memory_allocated(device)

    # ---- Data ----
    local_bs = BATCH_SIZE // world_size
    torch.manual_seed(42)
    all_data = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH + 1), device=device)
    offset = rank * local_bs
    inputs = all_data[offset : offset + local_bs, :-1]
    targets = all_data[offset : offset + local_bs, 1:]

    def timed_step():
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_autocast):
            logits = wrapped(inputs)
            loss = cross_entropy(logits, targets)
        scaler.scale(loss).backward()

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        wrapped.finish_gradient_synchronization()

        torch.cuda.synchronize()
        t2 = time.perf_counter()

        scaler.step(optimizer)
        scaler.update()

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

    torch.cuda.synchronize()
    mem_after_warmup_current = torch.cuda.memory_allocated(device)
    mem_after_warmup_peak = torch.cuda.max_memory_allocated(device)

    # ---- Memory measurement ----
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    mem_measure_start_current = torch.cuda.memory_allocated(device)

    optimizer.zero_grad()
    torch.cuda.synchronize()
    mem_after_zero_grad_current = torch.cuda.memory_allocated(device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_autocast):
        logits = wrapped(inputs)
        loss = cross_entropy(logits, targets)
    loss.backward()
    wrapped.finish_gradient_synchronization()

    torch.cuda.synchronize()
    mem_after_backward_current = torch.cuda.memory_allocated(device)
    mem_after_backward_peak = torch.cuda.max_memory_allocated(device)

    optimizer.step()

    torch.cuda.synchronize()
    mem_after_step_current = torch.cuda.memory_allocated(device)
    mem_after_step_peak = torch.cuda.max_memory_allocated(device)

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

    local_times = {"total": totals, "fwd_bwd": fwd_bwds, "sync": syncs, "optim": optims}
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_times)

    if rank == 0:
        avg = {}
        for key in ("total", "fwd_bwd", "sync", "optim"):
            arr = np.array([g[key] for g in gathered]).mean(axis=0)
            avg[key] = arr

        result_queue.put({
            "label": label,
            "mode": mode,
            "num_params": num_params,
            "param_bytes": param_bytes,
            "mem_after_wrap_current": mem_after_wrap_current,
            "mem_after_wrap_peak": mem_after_wrap_peak,
            "mem_after_warmup_current": mem_after_warmup_current,
            "mem_after_warmup_peak": mem_after_warmup_peak,
            "mem_measure_start_current": mem_measure_start_current,
            "mem_after_zero_grad_current": mem_after_zero_grad_current,
            "mem_after_backward_current": mem_after_backward_current,
            "mem_after_backward_peak": mem_after_backward_peak,
            "mem_after_step_current": mem_after_step_current,
            "mem_after_step_peak": mem_after_step_peak,
            # Backward-compatible aliases used by older result printers.
            "mem_after_init": mem_after_wrap_peak,
            "mem_before_step": mem_after_backward_peak,
            "mem_after_step": mem_after_step_peak,
            "total_median_ms": float(np.median(avg["total"]) * 1000),
            "total_mean_ms": float(np.mean(avg["total"]) * 1000),
            "total_std_ms": float(np.std(avg["total"]) * 1000),
            "fwd_bwd_median_ms": float(np.median(avg["fwd_bwd"]) * 1000),
            "sync_median_ms": float(np.median(avg["sync"]) * 1000),
            "optim_median_ms": float(np.median(avg["optim"]) * 1000),
        })

    dist.barrier()
    dist.destroy_process_group()


def _run(mode: str):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.spawn(_worker, args=(WORLD_SIZE, q, mode), nprocs=WORLD_SIZE, join=True)
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


MAIN_MODES = ["ddp", "fsdp_fp32", "fsdp_fp16"]
PREFETCH_MODES = ["ddp", "fsdp_prefetch_0", "fsdp_prefetch_1", "fsdp_prefetch_2", "fsdp_prefetch_4"]
ALL_MODES = MAIN_MODES

SHORT_LABELS = {
    "ddp": "DDP",
    "fsdp_fp32": "FSDP fp32",
    "fsdp_fp16": "FSDP fp16",
    "fsdp_prefetch_0": "FSDP p0",
    "fsdp_prefetch_1": "FSDP p1",
    "fsdp_prefetch_2": "FSDP p2",
    "fsdp_prefetch_4": "FSDP p4",
}


def _modes_for_experiment(experiment: str) -> list[str]:
    if experiment == "main":
        return MAIN_MODES
    if experiment == "prefetch":
        return PREFETCH_MODES
    raise ValueError(f"Unknown experiment: {experiment}")


@app.function(image=image, gpu="A100-80GB:2", timeout=1800)
def run_remote(experiment: str = "main"):
    modes = _modes_for_experiment(experiment)
    results = {}
    for m in modes:
        results[m] = _run(mode=m)
    return results


@app.function(image=image, gpu="A100-80GB:2", timeout=1200)
def run_single(mode: str):
    return _run(mode=mode)


def _print_results(results: dict, experiment: str = "main"):
    modes = [m for m in _modes_for_experiment(experiment) if m in results]
    col_w = 12

    print("\n" + "=" * 90)
    print("  FSDP Benchmark  |  XL model, {} GPUs".format(WORLD_SIZE))
    print("=" * 90)

    # Header
    header = f"{'Phase':<25s}" + "".join(f"  {SHORT_LABELS[m]:>{col_w}s}" for m in modes)
    sep = "-" * len(header)

    # Current allocated memory table
    print("\n--- Current Allocated Memory (per rank) ---")
    print(header)
    print(sep)
    for phase, key in [
        ("After wrap", "mem_after_wrap_current"),
        ("After warmup", "mem_after_warmup_current"),
        ("Measure start", "mem_measure_start_current"),
        ("After zero_grad", "mem_after_zero_grad_current"),
        ("After backward+sync", "mem_after_backward_current"),
        ("After optim.step", "mem_after_step_current"),
    ]:
        vals = "".join(f"  {_fmt(results[m][key]):>{col_w}s}" for m in modes)
        print(f"{phase:<25s}{vals}")

    # Peak memory table
    print("\n--- Peak Memory (per rank) ---")
    print(header)
    print(sep)
    for phase, key in [
        ("Build/wrap peak", "mem_after_wrap_peak"),
        ("Warmup peak", "mem_after_warmup_peak"),
        ("Measured fwd+bwd peak", "mem_after_backward_peak"),
        ("Measured step peak", "mem_after_step_peak"),
    ]:
        vals = "".join(f"  {_fmt(results[m][key]):>{col_w}s}" for m in modes)
        print(f"{phase:<25s}{vals}")

    # Savings
    print("\n--- Memory Savings (measured step peak, vs DDP) ---")
    ddp_peak = results["ddp"]["mem_after_step_peak"]
    for m in modes:
        if m == "ddp":
            continue
        peak = results[m]["mem_after_step_peak"]
        saving = ddp_peak - peak
        pct = saving / ddp_peak * 100
        print(f"  {SHORT_LABELS[m]:<16s}: {_fmt(saving):>10s} saved ({pct:+.1f}%)")

    # Timing table
    print("\n--- Timing (median, ms) ---")
    theader = f"{'Component':<25s}" + "".join(f"  {SHORT_LABELS[m]:>{col_w}s}" for m in modes)
    print(theader)
    print("-" * len(theader))
    for comp, key in [
        ("Total step", "total_median_ms"),
        ("  Forward + backward", "fwd_bwd_median_ms"),
        ("  Gradient sync", "sync_median_ms"),
        ("  Optimizer step", "optim_median_ms"),
    ]:
        vals = "".join(f"  {results[m][key]:>{col_w}.1f}" for m in modes)
        print(f"{comp:<25s}{vals}")

    # Speed comparison
    print("\n--- Speed Overhead vs DDP ---")
    ddp_t = results["ddp"]["total_median_ms"]
    for m in modes:
        if m == "ddp":
            continue
        t = results[m]["total_median_ms"]
        overhead = t - ddp_t
        pct = overhead / ddp_t * 100
        print(f"  {SHORT_LABELS[m]:<16s}: {overhead:+.1f} ms ({pct:+.1f}%)")

    # Parameter accounting
    r = results["ddp"]
    pb = r["param_bytes"]
    print(f"\n--- Theoretical Memory ({_fmt(pb)} params, N={WORLD_SIZE}) ---")
    print(f"  DDP:  4P = {_fmt(4 * pb)}")
    print(f"  FSDP: 4P/N = {_fmt(4 * pb // WORLD_SIZE)} + prefetch-dependent layer buffers")
    print(f"  Expected saving: {_fmt(4 * pb - 4 * pb // WORLD_SIZE)}")


@app.local_entrypoint()
def modal_main(mode: str = "all", experiment: str = "main"):
    """Run FSDP benchmark.

    Args:
        mode: "all" runs every mode for the selected experiment.
              Or specify one mode directly, e.g. ddp, fsdp_fp32, fsdp_prefetch_2.
        experiment: "main" runs DDP/FSDP fp32/FSDP fp16.
                    "prefetch" runs DDP/FSDP prefetch depth 0/1/2/4.
    """
    os.makedirs("benchmark_results", exist_ok=True)

    if mode == "all":
        modes = _modes_for_experiment(experiment)
        print(f"FSDP Benchmark ({experiment}): XL model, {WORLD_SIZE} GPUs ...")
        print(f"Modes: {', '.join(modes)}")
        results = run_remote.remote(experiment)
        _print_results(results, experiment=experiment)
        path = (
            "benchmark_results/fsdp_benchmark.json"
            if experiment == "main"
            else f"benchmark_results/fsdp_{experiment}_benchmark.json"
        )
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {path}")
    else:
        print(f"FSDP Benchmark (single mode={mode}): XL model, {WORLD_SIZE} GPUs ...")
        r = run_single.remote(mode)
        print(f"\n{'='*60}")
        print(f"  Mode: {r['label']}")
        print(f"{'='*60}")
        print(f"  Current Memory:")
        print(f"    After wrap:           {_fmt(r['mem_after_wrap_current'])}")
        print(f"    After warmup:         {_fmt(r['mem_after_warmup_current'])}")
        print(f"    Measure start:        {_fmt(r['mem_measure_start_current'])}")
        print(f"    After backward+sync:  {_fmt(r['mem_after_backward_current'])}")
        print(f"    After optimizer.step: {_fmt(r['mem_after_step_current'])}")
        print(f"  Peak Memory:")
        print(f"    Build/wrap peak:      {_fmt(r['mem_after_wrap_peak'])}")
        print(f"    Warmup peak:          {_fmt(r['mem_after_warmup_peak'])}")
        print(f"    Fwd+bwd peak:         {_fmt(r['mem_after_backward_peak'])}")
        print(f"    Step peak:            {_fmt(r['mem_after_step_peak'])}")
        print(f"  Timing (median):")
        print(f"    Total step:          {r['total_median_ms']:.1f} ms")
        print(f"    Forward + backward:  {r['fwd_bwd_median_ms']:.1f} ms")
        print(f"    Gradient sync:       {r['sync_median_ms']:.1f} ms")
        print(f"    Optimizer step:      {r['optim_median_ms']:.1f} ms")
        path = f"benchmark_results/fsdp_{mode}.json"
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"\nResults saved to {path}")
