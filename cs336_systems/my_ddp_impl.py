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


class FlatDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        seen = set()
        self._param_list = []
        total_numel = 0
        for p in self.module.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                self._param_list.append(p)
                total_numel += p.numel()

        self._flat_grad = torch.zeros(
            total_numel, dtype=self._param_list[0].dtype,
            device=self._param_list[0].device,
        )

        self._views = []
        offset = 0
        for p in self._param_list:
            numel = p.numel()
            view = self._flat_grad[offset:offset + numel].view(p.shape)
            self._views.append(view)
            p.grad = view
            offset += numel

    def forward(self, *args, **kwargs):
        for p, view in zip(self._param_list, self._views):
            if p.grad is not view:
                view.zero_()
                p.grad = view
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        dist.all_reduce(self._flat_grad, op=dist.ReduceOp.SUM)
        self._flat_grad.div_(self.world_size)


class OverlapDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        seen = set()
        self._grad_sync_handlers: list[tuple[torch.nn.Parameter, dist.Work]] = []

        def _sync_grad(param: torch.nn.Parameter):
            handle = dist.all_reduce(param.grad.data, async_op=True)
            self._grad_sync_handlers.append((param, handle))

        for p in self.module.parameters():
            if p.data_ptr() not in seen and p.requires_grad:
                seen.add(p.data_ptr())
                p.register_post_accumulate_grad_hook(_sync_grad)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for param, handle in self._grad_sync_handlers:
            handle.wait()
            param.grad.data.div_(self.world_size)
        self._grad_sync_handlers.clear()


# --------------- correctness verification ---------------

class _ToyModel(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear2(self.linear1(x))


def _ddp_worker(rank, world_size, model_fn, all_x, all_y, ref_params):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(0)
    model = model_fn()
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


def _single_baseline(model_fn, all_x, all_y):
    torch.manual_seed(0)
    model = model_fn()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    optimizer.zero_grad()
    out = model(all_x)
    loss = loss_fn(out, all_y)
    loss.backward()
    optimizer.step()

    return {name: p.data.clone() for name, p in model.named_parameters()}


def verify_ddp(model_fn, input_dim, output_dim=None, batch_size=16, world_size=2):
    """Verify NaiveDDP matches single-process baseline for any model factory."""
    if output_dim is None:
        output_dim = input_dim

    torch.manual_seed(42)
    all_x = torch.randn(batch_size, input_dim)
    all_y = torch.randn(batch_size, output_dim)

    baseline = _single_baseline(model_fn, all_x, all_y)

    manager = mp.Manager()
    ddp_params = manager.dict()
    mp.spawn(
        _ddp_worker,
        args=(world_size, model_fn, all_x, all_y, ddp_params),
        nprocs=world_size,
        join=True,
    )

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
    return all_match


if __name__ == "__main__":
    verify_ddp(model_fn=_ToyModel, input_dim=1024)
