import argparse
import contextlib
import timeit

import torch
import torch.cuda.nvtx as nvtx

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy, clip_gradient

MODEL_CONFIGS = {
    "small":  dict(d_model=768,   d_ff=3072,   num_layers=12, num_heads=12),   # ~125M
    "medium": dict(d_model=1024,  d_ff=4096,   num_layers=24, num_heads=16),   # ~350M
    "large":  dict(d_model=1280,  d_ff=5120,   num_layers=36, num_heads=20),   # ~760M
    "xl":     dict(d_model=2560,  d_ff=10240,  num_layers=32, num_heads=32),
    "10B":    dict(d_model=4608,  d_ff=12288,  num_layers=50, num_heads=36),
}

def run_benchmark(
    batch_size: int=4,
    vocab_size: int=10_000,
    context_length: int=512,
    d_model: int=768,
    d_ff: int=3072,
    num_layers: int=12,
    num_heads: int=12,
    warmup_steps: int=5,
    repetion_steps: int=10,
    dtype: torch.dtype=torch.float32,
    enable_profile: bool=False,
    inference_only: bool=False,
    memory_snapshot_path: str=None,
):
    """Run a benchmark for given parameters
    
    Args:
        vocab_size: int
            The number of unique items in the output vocabulary to be predicted.
        context_length: int,
            The maximum number of tokens to process at once.
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        num_layers: int
            The number of Transformer layers to use.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
    """
    device = torch.accelerator.current_accelerator()
    
    torch.manual_seed(0)
    data = torch.randint(0, vocab_size, (batch_size, context_length + 1), dtype=torch.int64, device=device)
    inputs = data[:,:-1]
    targets = data[:,1:] # shift 1
    
    
    model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M / {num_params / 1e9:.2f}B)")
    if inference_only:
        model.eval()
    optimizer = AdamW(model.parameters())
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        if inference_only:
            with torch.no_grad():
                logits = model(inputs)
        else:
            logits = model(inputs)
            losses = cross_entropy(logits, targets)
            losses.backward()
            optimizer.step()
    torch.cuda.synchronize()
    
    print(f"Warmup done, will run benchmark with {repetion_steps} repetion steps (inference_only={inference_only})")
    
    events = [dict() for _ in range(repetion_steps)]
    autocast_ctx = torch.autocast(device_type="cuda", dtype=dtype) if dtype != torch.float32 else contextlib.nullcontext()

    record_memory = enable_profile or memory_snapshot_path is not None
    if record_memory:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    def trace_handler(profiler):
        print(f"\n[Step {profiler.step_num}] Profiler Report:")
        print(profiler.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
        print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    profile_ctx = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=trace_handler,
        with_flops=True,
        with_stack=True,
        profile_memory=True,
        record_shapes=True,
        with_modules=True) if enable_profile else contextlib.nullcontext()

    no_grad_ctx = torch.no_grad() if inference_only else contextlib.nullcontext()

    with profile_ctx as profiler, no_grad_ctx:
        for i in range(repetion_steps):
            torch.cuda.synchronize()
            start_fw_time = timeit.default_timer()

            with autocast_ctx:
                with torch.cuda.nvtx.range("forward_pass"):
                    logits = model(inputs)
                if not inference_only:
                    with torch.cuda.nvtx.range("loss_calculation"):
                        losses = cross_entropy(logits, targets)
            torch.cuda.synchronize()
            events[i]['fw'] = timeit.default_timer() - start_fw_time

            if not inference_only:
                start_bw_time = timeit.default_timer()
                with torch.cuda.nvtx.range("backward_pass"):
                    losses.backward()
                torch.cuda.synchronize()
                events[i]['bw'] = timeit.default_timer() - start_bw_time

                start_op_time = timeit.default_timer()
                with torch.cuda.nvtx.range("optimizer.step"):
                    optimizer.step()
                    optimizer.zero_grad()
                torch.cuda.synchronize()
                events[i]['op'] = timeit.default_timer() - start_op_time

                events[i]['total'] = events[i]['fw'] + events[i]['bw'] + events[i]['op']
            else:
                events[i]['total'] = events[i]['fw']

            if profiler is not None:
                profiler.step()

    snapshot_path = memory_snapshot_path or ("memory_snapshot.pickle" if enable_profile else None)
    if record_memory and snapshot_path:
        torch.cuda.memory._dump_snapshot(snapshot_path)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"Memory snapshot saved to {snapshot_path}")
    
    from pandas import DataFrame
    df = DataFrame.from_records(events)
    print(df.to_string())
    print(df.to_latex())
    print("="*100)
    print(df.describe())
    print(df.describe().to_latex())
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("benchmark")
    parser.add_argument("--warmup-steps", "-w", type=int, default=5)
    parser.add_argument("--size", "-s", type=str, default="small",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model size config to benchmark")
    parser.add_argument("--batch-size", "-b", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--repeat", "-r", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "bfloat16", "float16"],
                        help="Data type for mixed precision")
    parser.add_argument("--profile", action="store_true", help="Enable torch profiler and memory snapshot")
    parser.add_argument("--inference-only", action="store_true", help="Run forward pass only (no backward/optimizer)")
    parser.add_argument("--memory-snapshot", type=str, default=None, help="Path to save memory snapshot pickle")
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    config = MODEL_CONFIGS[args.size]
    print(f"Benchmarking model size: {args.size} | config: {config} | dtype: {args.dtype}")
    run_benchmark(
        batch_size=args.batch_size,
        context_length=args.context_length,
        warmup_steps=args.warmup_steps,
        repetion_steps=args.repeat,
        enable_profile=args.profile,
        inference_only=args.inference_only,
        memory_snapshot_path=args.memory_snapshot,
        dtype=dtype,
        **config,
    )