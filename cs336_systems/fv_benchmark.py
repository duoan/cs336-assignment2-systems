"""
FlashAttention-2 benchmark on B200.

Compares 4 implementations side-by-side in a single sweep:
  1. pt_eager  - naive softmax(QK^T / sqrt(d)) V (NO FlashAttention)
  2. pt_compile- same math, wrapped in torch.compile (kernel fusion only)
  3. sdpa      - torch.nn.functional.scaled_dot_product_attention (industry FA backend)
  4. triton    - your tiled FlashAttention-2 implementation

Usage:
    python -m cs336_systems.flash_benchmark
    python -m cs336_systems.flash_benchmark --csv results.csv
    python -m cs336_systems.flash_benchmark --impls triton sdpa --seq 4096 16384
"""

import argparse
import gc
import itertools
import math
import sys
from dataclasses import dataclass
from collections.abc import Callable

import pandas as pd
import torch
import triton
from torch.nn.attention import SDPBackend, sdpa_kernel
from triton.testing import do_bench
import torch._functorch.config


from cs336_systems.triton_flash_attention import TritonFlashAttentionAutograd

torch._functorch.config.donated_buffer = False
torch._dynamo.config.cache_size_limit = 64
torch.set_float32_matmul_precision("high")


# ============================================================================
# Implementation 1: naive PyTorch (NO FlashAttention)
# ============================================================================
def pt_eager_attention(Q, K, V, is_causal: bool):
    """Materializes the full (L_q, L_k) score matrix - the baseline the spec asks for."""
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    S = torch.einsum("... q d, ... k d -> ... q k", Q, K) * scale
    if is_causal:
        L_q, L_k = Q.shape[-2], K.shape[-2]
        mask = torch.ones(L_q, L_k, dtype=torch.bool, device=Q.device).tril()
        S = S.masked_fill(~mask, float("-inf"))
    P = torch.softmax(S, dim=-1)
    return torch.einsum("... q k, ... k d -> ... q d", P, V)


# ============================================================================
# Implementation 2: torch.compile-d naive PyTorch (kernel fusion only)
#   dynamic=True so we don't recompile per shape
# ============================================================================
pt_compile_attention = torch.compile(pt_eager_attention, fullgraph=False, dynamic=True)


# ============================================================================
# Implementation 3: SDPA, forced to the FlashAttention backend when possible
# ============================================================================
def sdpa_attention(Q, K, V, is_causal: bool):
    """SDPA forced to the FA backend; falls back to EFFICIENT for fp32 / unsupported dims."""
    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    with sdpa_kernel(backends):
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)


# ============================================================================
# Implementation 4: your Triton FA2
# ============================================================================
def triton_attention(Q, K, V, is_causal: bool, block_q: int, block_k: int):
    return TritonFlashAttentionAutograd.apply(Q, K, V, is_causal, block_q, block_k)


# ============================================================================
# Tile-size heuristic for the Triton implementation
# ============================================================================
def pick_tile_size(seq_len: int, d: int, dtype: torch.dtype):
    """
    Pick (BLOCK_Q, BLOCK_K) so the per-tile SRAM footprint fits within the SM budget.

    Approximate SRAM cost (with num_stages=2 pipelined K/V loads):
        Q tile :  BQ * d  * bytes
        K tile :  BK * d  * bytes * num_stages
        V tile :  BK * d  * bytes * num_stages
        S tile :  BQ * BK * 4              (fp32 accumulator, always)

    RTX PRO 6000 Blackwell only has ~99KB SRAM/SM (vs 228KB on H100/B200),
    so we have to be conservative — especially for fp32 where bytes_per_elem doubles.
    """
    bytes_qkv = 2 if dtype == torch.bfloat16 else 4
    sram_budget = 90 * 1024  # leave ~10KB headroom under the 99KB hard limit
    num_stages = 2  # we use num_stages=2 in the kernel for safety

    def fits(bq, bk):
        q = bq * d * bytes_qkv
        kv = 2 * bk * d * bytes_qkv * num_stages
        s = bq * bk * 4  # fp32 accumulator
        return (q + kv + s) <= sram_budget

    # candidate tile sizes, sorted from large (fast) to small (safe)
    candidates = [(128, 128), (128, 64), (64, 64), (64, 32), (32, 32), (16, 16)]
    for bq, bk in candidates:
        if bq > seq_len or bk > seq_len:
            continue
        if fits(bq, bk):
            return bq, bk
    return 16, 16  # last resort


# ============================================================================
# Implementation registry: name -> (display_name, fn_factory, materializes_LL_matrix)
#   - fn_factory(Q,K,V,is_causal,seq_len,d,dtype) -> a zero-arg callable for do_bench
#   - materializes_LL_matrix=True means we should skip if (L,L) won't fit in memory
# ============================================================================
@dataclass
class Impl:
    name: str
    display: str
    builds_LL: bool  # whether it materializes a full (L_q, L_k) matrix
    factory: Callable  # (Q,K,V,is_causal,seq_len,d,dtype) -> Callable[[], Tensor]


IMPLS = {
    "pt_eager": Impl(
        name="pt_eager",
        display="PT eager",
        builds_LL=True,
        factory=lambda Q, K, V, is_causal, *_: lambda: pt_eager_attention(Q, K, V, is_causal),
    ),
    "pt_compile": Impl(
        name="pt_compile",
        display="PT compile",
        builds_LL=True,
        factory=lambda Q, K, V, is_causal, *_: lambda: pt_compile_attention(Q, K, V, is_causal),
    ),
    "sdpa": Impl(
        name="sdpa",
        display="SDPA",
        builds_LL=False,
        factory=lambda Q, K, V, is_causal, *_: lambda: sdpa_attention(Q, K, V, is_causal),
    ),
    "triton": Impl(
        name="triton",
        display="Triton FA",
        builds_LL=False,
        factory=lambda Q, K, V, is_causal, seq_len, d, dtype: lambda bq_bk=pick_tile_size(seq_len, d, dtype): triton_attention(Q, K, V, is_causal, bq_bk[0], bq_bk[1]),
    ),
}


def baseline_will_oom(seq_len: int, d: int, dtype: torch.dtype) -> bool:
    """Conservative check for impls that materialize (L, L) score / dscore matrices."""
    bytes_per_elem = 2 if dtype == torch.bfloat16 else 4
    mem_LL = 4 * (seq_len**2) * bytes_per_elem  # S, P, dS, dP
    mem_LD = 7 * seq_len * d * bytes_per_elem
    return mem_LL + mem_LD > 60 * 1024**3


# ============================================================================
# Bench a single (impl, shape) and return (fwd_ms, bwd_ms, e2e_ms)
# ============================================================================
def bench_one(impl: Impl, seq_len: int, d: int, dtype: torch.dtype, device="cuda"):
    B, H = 1, 1
    is_causal = True

    if impl.builds_LL and baseline_will_oom(seq_len, d, dtype):
        return None

    Q = K = V = dO = out = None
    try:
        torch.cuda.empty_cache()
        gc.collect()

        Q = torch.randn(B, H, seq_len, d, device=device, dtype=dtype, requires_grad=True)
        K = torch.randn(B, H, seq_len, d, device=device, dtype=dtype, requires_grad=True)
        V = torch.randn(B, H, seq_len, d, device=device, dtype=dtype, requires_grad=True)
        dO = torch.randn(B, H, seq_len, d, device=device, dtype=dtype)

        fwd_fn = impl.factory(Q, K, V, is_causal, seq_len, d, dtype)

        # ---- forward ----
        fwd_ms = do_bench(fwd_fn, warmup=10, rep=50)

        # ---- backward (forward done once, not timed) ----
        out = fwd_fn()

        def bwd_fn():
            Q.grad = K.grad = V.grad = None
            out.backward(dO, retain_graph=True)

        bwd_ms = do_bench(bwd_fn, warmup=5, rep=20)

        # ---- end-to-end (fwd + bwd) ----
        def e2e_fn():
            Q.grad = K.grad = V.grad = None
            o = fwd_fn()
            o.backward(dO)

        e2e_ms = do_bench(e2e_fn, warmup=5, rep=20)

        return (fwd_ms, bwd_ms, e2e_ms)

    except torch.cuda.OutOfMemoryError:
        return None
    except Exception as e:
        print(f"  ! {impl.display} failed @ L={seq_len}, d={d}, {dtype}: {type(e).__name__}: {e}", file=sys.stderr)
        return None
    finally:
        for v in (Q, K, V, dO, out):
            del v
        torch.cuda.empty_cache()
        gc.collect()


# ============================================================================
# Driver: run all impls for each (dtype, L, d) and print one merged row
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Optional: write results to CSV")
    parser.add_argument("--seq", nargs="+", type=int, default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--d", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--dtype", nargs="+", default=["bf16", "fp32"])
    parser.add_argument("--impls", nargs="+", default=list(IMPLS.keys()), choices=list(IMPLS.keys()), help="Which implementations to benchmark")
    args = parser.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp32": torch.float32}
    impls = [IMPLS[n] for n in args.impls]

    print(f"# Device: {torch.cuda.get_device_name(0)}")
    print(f"# torch={torch.__version__}, triton={triton.__version__}")
    print(f"# Implementations: {[i.display for i in impls]}")

    # Build header dynamically based on selected impls
    header_cells = ["dtype", "L", "d"]
    for imp in impls:
        header_cells += [f"{imp.display} fwd", f"{imp.display} bwd", f"{imp.display} e2e"]
    print(" | ".join(f"{c:>14}" for c in header_cells))
    print("-" * (17 * len(header_cells)))

    rows = []
    for dtype_name, seq_len, d in itertools.product(args.dtype, args.seq, args.d):
        dtype = dtype_map[dtype_name]

        # Run all selected impls back-to-back on this shape
        results = {imp.name: bench_one(imp, seq_len, d, dtype) for imp in impls}

        # ---- print one row ----
        def fmt(x):
            return f"{x:>14.3f}" if x is not None else f"{'OOM/—':>14}"

        cells = [f"{dtype_name:>14}", f"{seq_len:>14}", f"{d:>14}"]
        for imp in impls:
            r = results[imp.name]
            if r is None:
                cells += [fmt(None)] * 3
            else:
                cells += [fmt(r[0]), fmt(r[1]), fmt(r[2])]
        print(" | ".join(cells), flush=True)

        # ---- collect for CSV ----
        row = {"dtype": dtype_name, "L": seq_len, "d": d}
        for imp in impls:
            r = results[imp.name]
            row[f"{imp.name}_fwd_ms"] = r[0] if r else None
            row[f"{imp.name}_bwd_ms"] = r[1] if r else None
            row[f"{imp.name}_e2e_ms"] = r[2] if r else None
        rows.append(row)

    df = pd.DataFrame(rows)

    df = df.rename(
        columns={
            "pt_eager_fwd_ms": "PT fwd",
            "pt_eager_bwd_ms": "PT bwd",
            "pt_eager_e2e_ms": "PT e2e",
            "pt_compile_fwd_ms": "PTc fwd",
            "pt_compile_bwd_ms": "PTc bwd",
            "pt_compile_e2e_ms": "PTc e2e",
            "sdpa_fwd_ms": "SDPA fwd",
            "sdpa_bwd_ms": "SDPA bwd",
            "sdpa_e2e_ms": "SDPA e2e",
            "triton_fwd_ms": "TR fwd",
            "triton_bwd_ms": "TR bwd",
            "triton_e2e_ms": "TR e2e",
        }
    )

    df["TR vs PT"] = df["PT e2e"] / df["TR e2e"]
    df["TR vs SDPA"] = df["SDPA e2e"] / df["TR e2e"]

    print(df.to_string())

    print("=" * 100)

    def emit_phase_table(df, dtype, phase):
        """phase ∈ {'fwd', 'bwd', 'e2e'}"""
        sub = df[df["dtype"] == dtype].copy()
        cols = ["L", "d", f"PT {phase}", f"PTc {phase}", f"SDPA {phase}", f"TR {phase}"]
        sub = sub[cols].rename(
            columns={
                f"PT {phase}": "PT eager",
                f"PTc {phase}": "PT compile",
                f"SDPA {phase}": "SDPA",
                f"TR {phase}": "Triton (ours)",
            }
        )
        return sub.to_latex(
            index=False,
            float_format="%.3f",
            na_rep="OOM",
            caption=f"{phase.upper()} latency (ms), dtype={dtype}, batch=1, causal=True",
            label=f"tab:fa2-{dtype}-{phase}",
            column_format="rr" + "r" * 4,
        )

    for dtype in ["bf16", "fp32"]:
        for phase in ["fwd", "bwd", "e2e"]:
            latex = emit_phase_table(df, dtype, phase)
            print(latex)
            print("\n\n")

    if args.csv:
        df.to_csv("results.csv", index=False)
        print(f"\n# wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    torch.manual_seed(0)
    main()
