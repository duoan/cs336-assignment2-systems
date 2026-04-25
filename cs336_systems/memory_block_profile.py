"""Per-TransformerBlock memory profiling with NVTX labels.

Run standalone or under nsys:
  nsys profile --cuda-memory-usage=true -t cuda,nvtx \
    -o output .venv/bin/python -m cs336_systems.memory_block_profile
"""
import argparse
from torch.nn.modules.module import Module
import torch
import torch.cuda.nvtx as nvtx
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

MODEL_CONFIGS = {
    "small":  dict(d_model=768,   d_ff=3072,   num_layers=12, num_heads=12),
    "medium": dict(d_model=1024,  d_ff=4096,   num_layers=24, num_heads=16),
    "large":  dict(d_model=1280,  d_ff=5120,   num_layers=36, num_heads=20),
    "xl":     dict(d_model=2560,  d_ff=10240,  num_layers=32, num_heads=32),
}


def profile_block_memory(size="xl", context_length=128, batch_size=4):
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda")
    cfg = MODEL_CONFIGS[size]
    d_model, d_ff = cfg["d_model"], cfg["d_ff"]
    num_layers, num_heads = cfg["num_layers"], cfg["num_heads"]
    vocab_size = 10_000

    torch.manual_seed(0)
    data = torch.randint(0, vocab_size, (batch_size, context_length + 1),
                         dtype=torch.int64, device=device)
    inputs, targets = data[:, :-1], data[:, 1:]

    model = BasicsTransformerLM(
        vocab_size, context_length, d_model, num_layers, num_heads, d_ff
    ).to(device)
    optimizer = AdamW(model.parameters())

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {size} | params: {num_params:,} | ctx={context_length} batch={batch_size}")

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # --- Instrument per-block memory via hooks ---
    fw_mem_before = {}
    fw_mem_after = {}
    bw_mem_before = {}
    bw_mem_after = {}

    def make_fw_pre_hook(idx):
        def hook(module, inp):
            torch.cuda.synchronize()
            fw_mem_before[idx] = torch.cuda.memory_allocated()
            nvtx.range_push(f"TransformerBlock_{idx}_forward")
        return hook

    def make_fw_hook(idx):
        def hook(module, inp, out):
            torch.cuda.synchronize()
            fw_mem_after[idx] = torch.cuda.memory_allocated()
            nvtx.range_pop()
        return hook

    def make_bw_pre_hook(idx):
        def hook(module, grad_output):
            torch.cuda.synchronize()
            bw_mem_before[idx] = torch.cuda.memory_allocated()
            nvtx.range_push(f"TransformerBlock_{idx}_backward")
        return hook

    def make_bw_hook(idx):
        def hook(module, grad_input, grad_output):
            torch.cuda.synchronize()
            bw_mem_after[idx] = torch.cuda.memory_allocated()
            nvtx.range_pop()
        return hook

    handles = []
    for i, layer in enumerate[Module](model.layers):
        handles.append(layer.register_forward_pre_hook(make_fw_pre_hook(i)))
        handles.append(layer.register_forward_hook(make_fw_hook(i)))
        handles.append(layer.register_full_backward_pre_hook(make_bw_pre_hook(i)))
        handles.append(layer.register_full_backward_hook(make_bw_hook(i)))

    # --- Profile one training step ---
    optimizer.zero_grad()
    torch.cuda.synchronize()

    mem_before_fw = torch.cuda.memory_allocated()
    nvtx.range_push("forward_pass")
    logits = model(inputs)
    nvtx.range_pop()
    torch.cuda.synchronize()
    mem_after_fw = torch.cuda.memory_allocated()

    nvtx.range_push("loss_calculation")
    loss = cross_entropy(logits, targets)
    nvtx.range_pop()
    torch.cuda.synchronize()
    mem_after_loss = torch.cuda.memory_allocated()

    nvtx.range_push("backward_pass")
    loss.backward()
    nvtx.range_pop()
    torch.cuda.synchronize()
    mem_after_bw = torch.cuda.memory_allocated()

    nvtx.range_push("optimizer_step")
    optimizer.step()
    optimizer.zero_grad()
    nvtx.range_pop()
    torch.cuda.synchronize()

    for h in handles:
        h.remove()

    # --- Print per-block forward memory (saved for backward) ---
    print("\n" + "=" * 80)
    print("PER-BLOCK FORWARD PASS: Memory saved for backward")
    print("=" * 80)
    print(f"{'Block':>6}  {'Before (GiB)':>14}  {'After (GiB)':>14}  {'Delta (MiB)':>14}")
    print("-" * 60)
    total_fw_delta = 0
    fw_deltas = []
    for i in range(num_layers):
        before = fw_mem_before.get(i, 0)
        after = fw_mem_after.get(i, 0)
        delta_mib = (after - before) / 1024**2
        fw_deltas.append(delta_mib)
        total_fw_delta += delta_mib
        print(f"{i:>6}  {before/1024**3:>14.4f}  {after/1024**3:>14.4f}  {delta_mib:>14.2f}")
    print(f"\nTotal forward delta across all blocks: {total_fw_delta:.2f} MiB")
    print(f"Average per block: {total_fw_delta/num_layers:.2f} MiB")

    print(f"\nOverall: before_fw={mem_before_fw/1024**3:.4f} GiB, "
          f"after_fw={mem_after_fw/1024**3:.4f} GiB, "
          f"after_loss={mem_after_loss/1024**3:.4f} GiB, "
          f"after_bw={mem_after_bw/1024**3:.4f} GiB")

    # --- Print per-block backward memory (gradient creation) ---
    print("\n" + "=" * 80)
    print("PER-BLOCK BACKWARD PASS: Memory change (freed saved tensors, created gradients)")
    print("=" * 80)
    print(f"{'Block':>6}  {'Before (GiB)':>14}  {'After (GiB)':>14}  {'Delta (MiB)':>14}")
    print("-" * 60)
    total_bw_delta = 0
    bw_deltas = []
    for i in range(num_layers):
        before = bw_mem_before.get(i, 0)
        after = bw_mem_after.get(i, 0)
        delta_mib = (after - before) / 1024**2
        bw_deltas.append(delta_mib)
        total_bw_delta += delta_mib
        print(f"{i:>6}  {before/1024**3:>14.4f}  {after/1024**3:>14.4f}  {delta_mib:>14.2f}")
    print(f"\nTotal backward delta across all blocks: {total_bw_delta:.2f} MiB")
    print(f"Average per block: {total_bw_delta/num_layers:.2f} MiB")

    # --- Analysis ---
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    avg_fw = sum(fw_deltas) / len(fw_deltas)
    avg_bw = sum(bw_deltas) / len(bw_deltas)
    print(f"Avg memory saved for backward per block (forward delta): {avg_fw:.2f} MiB")
    print(f"Avg net memory change per block during backward:         {avg_bw:.2f} MiB")
    print(f"  -> saved_for_bw freed:  -{avg_fw:.2f} MiB")
    print(f"  -> gradients created:   {avg_fw + avg_bw:.2f} MiB")
    print(f"     (net change = grads - freed = {avg_bw:.2f} MiB)")

    # Weight parameter count per block
    block_params = sum(p.numel() for p in model.layers[0].parameters())
    grad_mem_expected = block_params * 4 / 1024**2
    print(f"\nExpected gradient memory per block: {block_params:,} params * 4 bytes = {grad_mem_expected:.2f} MiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", "-s", default="xl", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", "-b", type=int, default=4)
    args = parser.parse_args()
    profile_block_memory(args.size, args.context_length, args.batch_size)
