import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        seen = set()
        for param in self.module.parameters():
            if param.grad is None or param.data_ptr() in seen:
                continue
            seen.add(param.data_ptr())
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data.div_(self.world_size)


# --------------- correctness verification ---------------

class _ToyModel(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear2(self.linear1(x))


def _ddp_worker(rank, world_size, all_x, all_y, ref_params):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(0)
    model = _ToyModel()
    ddp_model = NaiveDDP(model)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    B = all_x.shape[0]
    bsz = B // world_size
    offset = rank * bsz

    x = all_x[offset : offset + bsz]
    y = all_y[offset : offset + bsz]

    optimizer.zero_grad()
    out = ddp_model(x)
    loss = loss_fn(out, y)
    loss.backward()
    ddp_model.finish_gradient_synchronization()
    optimizer.step()

    if rank == 0:
        for name, p in model.named_parameters():
            ref_params[name] = p.data.clone()

    dist.barrier()
    dist.destroy_process_group()


def _single_baseline(all_x, all_y):
    torch.manual_seed(0)
    model = _ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    optimizer.zero_grad()
    out = model(all_x)
    loss = loss_fn(out, all_y)
    loss.backward()
    optimizer.step()

    return {name: p.data.clone() for name, p in model.named_parameters()}


def main():
    B, D = 16, 1024
    world_size = 2

    torch.manual_seed(42)
    all_x = torch.randn(B, D)
    all_y = torch.randn(B, D)

    baseline = _single_baseline(all_x, all_y)

    manager = mp.Manager()
    ddp_params = manager.dict()
    mp.spawn(_ddp_worker, args=(world_size, all_x, all_y, ddp_params), nprocs=world_size, join=True)

    print(f"{'Parameter':<25s} {'Max diff':>12s}  Match?")
    print("-" * 50)
    all_match = True
    for name in baseline:
        diff = (baseline[name] - ddp_params[name]).abs().max().item()
        ok = diff < 1e-5
        all_match = all_match and ok
        print(f"{name:<25s} {diff:12.2e}  {'✓' if ok else '✗'}")
    print("-" * 50)
    print("PASSED" if all_match else "FAILED")


if __name__ == "__main__":
    main()
