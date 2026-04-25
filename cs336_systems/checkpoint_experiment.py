"""Single-level gradient checkpointing experiment.

Measures peak memory for different checkpointing group sizes G
(number of consecutive TransformerBlocks per checkpoint segment).

Usage:
  .venv/bin/python -m cs336_systems.checkpoint_experiment --group-size 1
"""
import argparse
import contextlib
import torch
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

MODEL_CONFIGS = {
    "small":  dict(d_model=768,   d_ff=3072,   num_layers=12, num_heads=12),
    "medium": dict(d_model=1024,  d_ff=4096,   num_layers=24, num_heads=16),
    "large":  dict(d_model=1280,  d_ff=5120,   num_layers=36, num_heads=20),
    "xl":     dict(d_model=2560,  d_ff=10240,  num_layers=32, num_heads=32),
}


def run_block_group(blocks, x):
    for block in blocks:
        x = block(x)
    return x


def forward_with_checkpointing(model, input_ids, group_size):
    """Forward pass with single-level checkpointing."""
    _, seq_len = input_ids.size()
    x = model.token_embeddings(input_ids)

    layers = list(model.layers)
    N = len(layers)

    if group_size <= 0 or group_size >= N:
        for layer in layers:
            x = layer(x)
    else:
        for start in range(0, N, group_size):
            end = min(start + group_size, N)
            group = layers[start:end]
            x = checkpoint(run_block_group, group, x, use_reentrant=False)

    x = model.ln_final(x)
    logits = model.lm_head(x)
    return logits


def run_experiment(size, context_length, batch_size, group_size, dtype, warmup_steps=3):
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda")
    cfg = MODEL_CONFIGS[size]
    vocab_size = 10_000

    torch.manual_seed(0)
    data = torch.randint(0, vocab_size, (batch_size, context_length + 1),
                         dtype=torch.int64, device=device)
    inputs, targets = data[:, :-1], data[:, 1:]

    model = BasicsTransformerLM(
        vocab_size, context_length, cfg["d_model"],
        cfg["num_layers"], cfg["num_heads"], cfg["d_ff"]
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {size} | params: {num_params:,} | ctx={context_length} "
          f"batch={batch_size} group_size={group_size} dtype={dtype}")

    optimizer = AdamW(model.parameters())
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=dtype)
                    if dtype != torch.float32 else contextlib.nullcontext())

    for _ in range(warmup_steps):
        optimizer.zero_grad()
        with autocast_ctx:
            logits = forward_with_checkpointing(model, inputs, group_size)
            loss = cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    optimizer.zero_grad()
    with autocast_ctx:
        logits = forward_with_checkpointing(model, inputs, group_size)
        loss = cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    peak_mem_gib = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"  Peak GPU memory: {peak_mem_gib:.2f} GiB")
    return peak_mem_gib


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", "-s", default="xl", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--batch-size", "-b", type=int, default=4)
    parser.add_argument("--group-size", "-g", type=int, default=1,
                        help="Blocks per checkpoint segment (0=no checkpointing)")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "bfloat16"])
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep group sizes: 1,2,4,6,8,16,32 and no-ckpt")
    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    if args.sweep:
        results = {}
        for g in [1, 2, 4, 8, 16, 32]:
            try:
                peak = run_experiment(args.size, args.context_length,
                                      args.batch_size, g, dtype)
                results[g] = peak
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM with group_size={g}")
                results[g] = "OOM"
                torch.cuda.empty_cache()
        print("\n" + "=" * 60)
        print("SWEEP RESULTS")
        print("=" * 60)
        print(f"{'Group Size':>12} {'Peak Memory':>14}")
        print("-" * 30)
        for g, mem in sorted(results.items()):
            if mem == "OOM":
                print(f"{g:>12} {'OOM':>14}")
            else:
                print(f"{g:>12} {mem:>13.2f} GiB")
    else:
        run_experiment(args.size, args.context_length,
                       args.batch_size, args.group_size, dtype)
