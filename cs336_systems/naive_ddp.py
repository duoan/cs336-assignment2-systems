import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, dim, dtype=torch.float32):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim, bias=False, dtype=dtype)
        self.linear2 = nn.Linear(dim, dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.linear1(x))


def ddp_worker(rank, world_size, all_x, ref_params):
    """
    Each rank gets a shard of all_x, does forward/backward/all-reduce/step,
    then puts its updated params into ref_params so the caller can compare.
    """
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(0)
    model = ToyModel(dim=all_x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    dist.barrier()

    B = all_x.shape[0]
    assert B % world_size == 0
    bsz = B // world_size

    offset = rank * bsz
    x = all_x[offset : offset + bsz]

    y = model(x)
    loss = y.sum()
    loss.backward()

    for p in model.parameters():
        dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)

    optimizer.step()
    optimizer.zero_grad()

    if rank == 0:
        for name, param in model.named_parameters():
            ref_params[name] = param.data.clone()

    dist.destroy_process_group()


def single_gpu_baseline(all_x):
    """Run the same model + optimizer on the full batch (single process)."""
    torch.manual_seed(0)
    model = ToyModel(dim=all_x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    y = model(all_x)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return {name: p.data.clone() for name, p in model.named_parameters()}


def main():
    B, D = 16, 1024
    world_size = 2

    torch.manual_seed(42)
    all_x = torch.randn(B, D)

    baseline_params = single_gpu_baseline(all_x)

    manager = mp.Manager()
    ddp_params = manager.dict()

    mp.spawn(ddp_worker, args=(world_size, all_x, ddp_params), nprocs=world_size, join=True)

    print(f"{'Parameter':<25s} {'Max diff':>12s}  Match?")
    print("-" * 50)
    all_match = True
    for name in baseline_params:
        diff = (baseline_params[name] - ddp_params[name]).abs().max().item()
        match = diff < 1e-5
        all_match = all_match and match
        print(f"{name:<25s} {diff:12.2e}  {'✓' if match else '✗'}")

    print("-" * 50)
    if all_match:
        print("PASSED: DDP matches single-GPU baseline.")
    else:
        print("FAILED: DDP does NOT match single-GPU baseline.")


if __name__ == "__main__":
    main()
