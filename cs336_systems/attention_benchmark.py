"""PyTorch Attention Benchmarking (pytorch_attention problem).

Benchmarks scaled_dot_product_attention at different scales:
  batch=8, single-head (no head dim)
  d_model in [16, 32, 64, 128]
  seq_len in [256, 1024, 4096, 8192, 16384]

Usage:
  .venv/bin/python -m cs336_systems.attention_benchmark
"""
import argparse
import itertools
import timeit
import torch

from cs336_basics.model import scaled_dot_product_attention

BATCH = 8
D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]
N_ITERS = 100
N_WARMUP = 5


def benchmark_config(d_model, seq_len, compile=False):
    device = torch.device("cuda")

    Q = torch.randn(BATCH, seq_len, d_model, device=device, requires_grad=True)
    K = torch.randn(BATCH, seq_len, d_model, device=device, requires_grad=True)
    V = torch.randn(BATCH, seq_len, d_model, device=device, requires_grad=True)

    iota = torch.arange(seq_len, device=device)
    causal_mask = iota.unsqueeze(0) >= iota.unsqueeze(1)
    
    
    if compile:
        attention_layer = torch.compile(scaled_dot_product_attention, fullgraph=True)
    else:
        attention_layer = scaled_dot_product_attention

    for _ in range(N_WARMUP):
        out = attention_layer(Q, K, V, mask=causal_mask)
        loss = out.sum()
        loss.backward()
        Q.grad = K.grad = V.grad = None
    torch.cuda.synchronize()

    # --- Forward timing ---
    torch.cuda.synchronize()
    fw_start = timeit.default_timer()
    for _ in range(N_ITERS):
        out = attention_layer(Q, K, V, mask=causal_mask)
        torch.cuda.synchronize()
    fw_elapsed = timeit.default_timer() - fw_start

    # --- Memory before backward (from last forward's graph) ---
    torch.cuda.reset_peak_memory_stats()
    mem_before_bw = torch.cuda.memory_allocated()

    # --- Backward timing (separate from forward) ---
    bw_elapsed = 0.0
    for _ in range(N_ITERS):
        out = attention_layer(Q, K, V, mask=causal_mask)
        loss = out.sum()
        torch.cuda.synchronize()
        t0 = timeit.default_timer()
        loss.backward()
        torch.cuda.synchronize()
        bw_elapsed += timeit.default_timer() - t0
        Q.grad = K.grad = V.grad = None

    peak_mem = torch.cuda.max_memory_allocated()

    return {
        "fw_ms": fw_elapsed / N_ITERS * 1000,
        "bw_ms": bw_elapsed / N_ITERS * 1000,
        "mem_before_bw_mib": mem_before_bw / (1024**2),
        "peak_mem_mib": peak_mem / (1024**2),
    }


def main():
    torch.set_float32_matmul_precision("high")
    results = []
    print(f"{'d_model':>8} {'seq_len':>8} {'FW (ms)':>10} {'BW (ms)':>10} "
          f"{'Mem@BW (MiB)':>14} {'Peak (MiB)':>12} {'Status':>8}")
    print("-" * 80)

    parser = argparse.ArgumentParser("attention_benchmark")
    parser.add_argument("--compile", action="store_true", help="Enable torch compile")
    args = parser.parse_args()

    for d_model, seq_len in itertools.product(D_MODELS, SEQ_LENS):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            r = benchmark_config(d_model, seq_len, args.compile)
            status = "OK"
            print(f"{d_model:>8} {seq_len:>8} {r['fw_ms']:>10.3f} {r['bw_ms']:>10.3f} "
                  f"{r['mem_before_bw_mib']:>14.1f} {r['peak_mem_mib']:>12.1f} {status:>8}")
            results.append((d_model, seq_len, r, status))
        except torch.cuda.OutOfMemoryError:
            status = "OOM"
            print(f"{d_model:>8} {seq_len:>8} {'---':>10} {'---':>10} "
                  f"{'---':>14} {'---':>12} {status:>8}")
            results.append((d_model, seq_len, None, status))
            torch.cuda.empty_cache()

    # Memory analysis
    print("\n\nMemory scaling analysis (d_model=16, varying seq_len):")
    print(f"{'seq_len':>8} {'Mem@BW (MiB)':>14} {'seq^2 ratio':>14}")
    base = None
    for d, s, r, st in results:
        if d == 16 and st == "OK":
            if base is None:
                base = (s, r['mem_before_bw_mib'])
            ratio = r['mem_before_bw_mib'] / base[1]
            seq_ratio = (s / base[0]) ** 2
            print(f"{s:>8} {r['mem_before_bw_mib']:>14.1f} {ratio:>14.1f} (expected {seq_ratio:.1f})")

    print("\n\nLaTeX table:")
    print(r"\begin{tabular}{r r r r r r}")
    print(r"\toprule")
    print(r"$d$ & Seq len & FW (ms) & BW (ms) & Mem before BW (MiB) & Peak (MiB) \\")
    print(r"\midrule")
    for d, s, r, st in results:
        if st == "OOM":
            print(f"{d} & {s} & \\multicolumn{{4}}{{c}}{{OOM}} \\\\")
        else:
            print(f"{d} & {s} & {r['fw_ms']:.3f} & {r['bw_ms']:.3f} "
                  f"& {r['mem_before_bw_mib']:.1f} & {r['peak_mem_mib']:.1f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
