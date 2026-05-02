import os
import time
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn

from cs336_basics.model import Embedding, Linear

_TARGET_MODULES = (Linear, Embedding, nn.Linear, nn.Embedding)


def _storage_nbytes(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().nbytes()


def _alloc_storage_(tensor: torch.Tensor, numel: int) -> None:
    """Allocate tensor storage for numel elements without changing tensor metadata."""
    expected_nbytes = numel * tensor.element_size()
    if _storage_nbytes(tensor) != expected_nbytes:
        tensor.untyped_storage().resize_(expected_nbytes)


def _free_storage_(tensor: torch.Tensor) -> None:
    """Free tensor storage while keeping tensor metadata and storage identity."""
    if _storage_nbytes(tensor) == 0:
        return
    assert tensor.storage_offset() == 0
    tensor.untyped_storage().resize_(0)


class FullyShardedDataParallel(nn.Module):
    """Fully Sharded Data Parallel (FSDP) wrapper with prefetched all-gather.

    Combines ZeRO-1/2/3: each rank only stores 1/N of parameters, gradients,
    and optimizer states.  Weights are all-gathered on-the-fly before each
    layer's forward/backward, then immediately discarded.

    Prefetching: while layer i computes, we issue an async all-gather for
    layer i+1 (forward) or i-1 (backward) so communication overlaps with
    computation.

    Parameter sharding (init):
        Original W (e.g. [4096, 4096] = 64MB, world_size=4):
        ┌──────────┬──────────┬──────────┬──────────┐
        │ Shard 0  │ Shard 1  │ Shard 2  │ Shard 3  │  flatten & split
        │  16MB    │  16MB    │  16MB    │  16MB    │
        └──────────┴──────────┴──────────┴──────────┘
        Rank 0: param.data = Shard 0 (16MB only)
        Rank 1: param.data = Shard 1 (16MB only)  ...

    Forward pass (per layer, with prefetching):
        Layer 0                 Layer 1                 Layer 2
        ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
        │ pre-hook:      │     │ pre-hook:      │     │ pre-hook:      │
        │  sync gather W0│     │  wait(W1) done!│     │  wait(W2) done!│
        │  async W1 start│     │  async W2 start│     │  async W3 start│
        │ compute: X@W0  │     │ compute: X@W1  │     │ compute: X@W2  │
        │ post-hook:     │     │ post-hook:     │     │ post-hook:     │
        │  free W0→shard │     │  free W1→shard │     │  free W2→shard │
        └────────────────┘     └────────────────┘     └────────────────┘

    Communication / Computation overlap timeline:
                       time ──────────────────────────────────►
        compute:  [  L0 compute  ][  L1 compute  ][  L2 compute  ]
        comms:    [gather W0][gather W1][gather W2][gather W3]
                             ↑ overlaps  ↑ overlaps  ↑ overlaps
        Without prefetch: [gather][compute][gather][compute] (serial)
        With prefetch:    [gather][compute+gather][compute+gather] (pipelined)

    Backward pass (reverse order, same prefetch pattern):
        Layer N → Layer N-1 → ... → Layer 0
        - backward-pre-hook: ensure_gathered(Wi), async_gather(W_{i-1})
        - backward compute:  ∂L/∂Wi
        - grad-hook:         reduce_scatter(∂L/∂Wi) → each rank gets 1/N grad
                             free Wi → shard

    Memory comparison (N ranks, P total params):
        DDP:   4P per rank  (params + grads + Adam m + Adam v)
        FSDP:  P/N per rank (sharded params + grads + m + v)
               + 1 layer full weight peak during compute

    Parameter lifecycle (per sharded layer):
        idle:       param.data = local shard   (1/N of full weight)
        pre-hook:   all-gather → param.data = full weight
        compute:    layer uses full weight      (next layer's gather runs async)
        post-hook:  param.data = local shard   (full weight freed)
        grad-hook:  reduce-scatter → param.grad = gradient shard (1/N)

    Full-parameter storage reuse:
        Each sharded weight owns a reusable full-parameter buffer. We all-gather
        into that buffer before forward/backward, then resize its storage to
        zero after use. If autograd keeps a view to the full weight, the view
        points to the same storage, which is rematerialized before backward.
    """

    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
        prefetch_depth: int = 1,
    ):
        super().__init__()
        warnings.filterwarnings("ignore", message=".*no inputs require gradients.*")
        if prefetch_depth < 0:
            raise ValueError("prefetch_depth must be non-negative")
        self.module = module
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.compute_dtype = compute_dtype
        self.prefetch_depth = prefetch_depth

        # param → {shape, numel, chunk_size, dtype} for every sharded parameter
        self._shard_info: dict[nn.Parameter, dict] = {}
        # param → (Work handle, output views, gather input, direction, layer_idx)
        # for in-flight async gathers
        self._pending_work: dict[
            nn.Parameter, tuple[dist.Work, list[torch.Tensor], torch.Tensor, str, int]
        ] = {}
        self._profile_waits = os.environ.get("FSDP_PROFILE_WAITS") == "1"
        self._wait_profile_records: list[dict[str, float | int | str | bool]] = []
        self._record_forward_order = False
        self._observed_forward_order: list[nn.Module] = []
        self._has_learned_forward_order = False

        # ── Step 1: shard weights & register hooks in module execution order ──
        self._ordered_layers: list[nn.Module] = []
        for layer in module.modules():
            if not isinstance(layer, _TARGET_MODULES):
                continue
            self._shard_parameter(layer.weight)
            self._ordered_layers.append(layer)

            layer.register_forward_pre_hook(self._forward_pre_hook)
            layer.register_forward_hook(self._forward_hook)
            layer.register_full_backward_pre_hook(self._full_backward_pre_hook)

        # O(1) lookup: layer id → position in execution order
        self._refresh_layer_indices()

        # ── Step 2: register gradient hooks on sharded params ──
        for param in self.parameters():
            if param not in self._shard_info:
                continue
            param.register_post_accumulate_grad_hook(self._post_accumulate_grad_hook)

    def forward(self, *inputs, **kwargs):
        can_prefetch_boundaries = self._has_learned_forward_order and self.prefetch_depth > 0
        if can_prefetch_boundaries and self._ordered_layers:
            self._async_gather_weight(self._ordered_layers[0].weight, "fwd", 0)

        self._observed_forward_order = []
        self._record_forward_order = True
        try:
            output = self.module(*inputs, **kwargs)
        finally:
            self._record_forward_order = False

        self._maybe_update_forward_order()
        if can_prefetch_boundaries and self._output_requires_grad(output):
            last_idx = len(self._ordered_layers) - 1
            self._async_gather_weight(self._ordered_layers[last_idx].weight, "bwd", last_idx)
        return output

    def _output_requires_grad(self, output) -> bool:
        if torch.is_tensor(output):
            return output.requires_grad
        if isinstance(output, dict):
            return any(self._output_requires_grad(value) for value in output.values())
        if isinstance(output, (tuple, list)):
            return any(self._output_requires_grad(value) for value in output)
        return False

    def _refresh_layer_indices(self) -> None:
        self._layer_index = {id(layer): i for i, layer in enumerate(self._ordered_layers)}
        self._param_index = {layer.weight: i for i, layer in enumerate(self._ordered_layers)}

    def _maybe_update_forward_order(self) -> None:
        """Replace static module order with the actual forward hook order.

        ``module.modules()`` gives registration order, but Python execution can
        differ, e.g. ``w2(silu(w1(x)) * w3(x))`` executes as w1, w3, w2.
        Prefetch must follow execution order, not registration order.
        """
        observed = self._observed_forward_order
        if len(observed) != len(self._ordered_layers):
            return
        if {id(layer) for layer in observed} != {id(layer) for layer in self._ordered_layers}:
            return

        if any(current is not actual for current, actual in zip(self._ordered_layers, observed, strict=True)):
            self._ordered_layers = list(observed)
            self._refresh_layer_indices()
        self._has_learned_forward_order = True

    # ── public API ─────────────────────────────────────────────────────

    def finish_gradient_synchronization(self):
        """All-reduce gradients for replicated (non-sharded) parameters.

        Sharded params already have their gradients reduce-scattered in
        _post_accumulate_grad_hook.  This handles the remaining replicated
        params (e.g. LayerNorm / RMSNorm weights) that are identical across
        ranks and need a simple mean-reduce.
        """
        for param in self.module.parameters():
            if param in self._shard_info:
                continue
            if param.grad is not None:
                dist.all_reduce(param.grad.data)
                param.grad.data.div_(self.world_size)

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """Reconstruct full parameters on every rank (for correctness checks).

        Returns a dict mapping 'module.layer.weight' → full-size tensor.
        Sharded params are all-gathered; replicated params are cloned as-is.
        """
        result = {}
        for name, param in self.module.named_parameters():
            if param in self._shard_info:
                info = self._shard_info[param]
                local_shard = param._local_shard if hasattr(param, "_local_shard") else param.data
                shards = [torch.empty_like(local_shard) for _ in range(self.world_size)]
                dist.all_gather(shards, local_shard)
                full = torch.cat(shards)[: info["numel"]].view(info["shape"])
                result[name] = full.clone()
            else:
                result[name] = param.data.clone()
        return result

    # ── parameter sharding ─────────────────────────────────────────────

    def _shard_parameter(self, param: nn.Parameter):
        """Replace param.data with this rank's 1/N shard.

        Steps:
          1. Broadcast from rank 0 so all ranks start with identical weights
          2. Flatten → pad to be divisible by world_size → take local chunk
          3. Store metadata in _shard_info for later gather/scatter
        """
        if param in self._shard_info:
            return

        dist.broadcast(param.data, src=0)

        param_shape = param.data.shape
        param_numel = param.data.numel()
        param_dtype = param.dtype

        # Ceil-divide so every shard has equal size (last shard may have padding)
        chunk_size = (param_numel + self.world_size - 1) // self.world_size
        padded_numel = chunk_size * self.world_size
        flatten_data = param.data.flatten()

        if padded_numel > param_numel:
            flatten_data = torch.cat([flatten_data, flatten_data.new_zeros(padded_numel - param_numel)])

        start_idx = self.rank * chunk_size
        local_shard = flatten_data[start_idx : start_idx + chunk_size].clone()

        # Reusable full-parameter buffer, following the FairScale FSDP pattern.
        # Autograd may save views of gathered full weights during forward, so we
        # keep this tensor object stable across forward/backward. Its storage is
        # resized to zero when idle, then rematerialized in-place before compute.
        full_param_padded = torch.empty(
            padded_numel,
            dtype=self.compute_dtype if self.compute_dtype is not None else param_dtype,
            device=param.device,
        )
        _free_storage_(full_param_padded)

        # param.data is now just 1/N of the original weight
        param.data = local_shard
        self._shard_info[param] = {
            "shape": param_shape,
            "numel": param_numel,
            "chunk_size": chunk_size,
            "dtype": param_dtype,
            "full_param_padded": full_param_padded,
        }

    # ── async all-gather primitives ────────────────────────────────────

    def _full_param_buffer(self, param: nn.Parameter) -> torch.Tensor:
        """Return the reusable full-parameter buffer, allocating its storage."""
        info = self._shard_info[param]
        full_param_padded = info["full_param_padded"]
        _alloc_storage_(full_param_padded, info["chunk_size"] * self.world_size)
        return full_param_padded

    def _full_param_view(self, param: nn.Parameter) -> torch.Tensor:
        """View the reusable full-parameter buffer with the original shape."""
        info = self._shard_info[param]
        full_param_padded = info["full_param_padded"]
        return full_param_padded[: info["numel"]].view(info["shape"])

    def _gather_input(self, param: nn.Parameter) -> torch.Tensor:
        if self.compute_dtype is None:
            return param.data
        return param.data.to(self.compute_dtype)

    def wait_profile_summary(self) -> str:
        """Return a small text report for Nsight runs with FSDP_PROFILE_WAITS=1."""
        if not self._wait_profile_records:
            return "[fsdp_wait_profile] no wait records"

        lines = ["[fsdp_wait_profile] gather wait latency"]
        prefetched_total_ms = 0.0
        sync_total_ms = 0.0
        prefetched_count = 0
        sync_count = 0
        max_record = max(self._wait_profile_records, key=lambda r: float(r["wait_ms"]))
        for record in self._wait_profile_records:
            wait_ms = float(record["wait_ms"])
            prefetched = bool(record["prefetched"])
            if prefetched:
                prefetched_total_ms += wait_ms
                prefetched_count += 1
            else:
                sync_total_ms += wait_ms
                sync_count += 1
            lines.append(
                "  "
                f"{record['direction']} layer {record['layer_idx']:>3}: "
                f"{wait_ms:8.3f} ms "
                f"({'prefetched' if prefetched else 'sync fallback'})"
            )
        prefetched_avg_ms = prefetched_total_ms / max(prefetched_count, 1)
        lines.append(
            f"  prefetched_total={prefetched_total_ms:.3f} ms, "
            f"prefetched_avg={prefetched_avg_ms:.3f} ms, "
            f"sync_fallback_total={sync_total_ms:.3f} ms, "
            f"max={float(max_record['wait_ms']):.3f} ms "
            f"({max_record['direction']} layer {max_record['layer_idx']})"
        )
        return "\n".join(lines)

    def clear_wait_profile(self) -> None:
        self._wait_profile_records.clear()

    def _nvtx_push(self, label: str) -> bool:
        if not self._profile_waits or not torch.cuda.is_available():
            return False
        torch.cuda.nvtx.range_push(label)
        return True

    def _nvtx_pop(self, pushed: bool) -> None:
        if pushed:
            torch.cuda.nvtx.range_pop()

    def _async_gather_weight(self, param: nn.Parameter, direction: str, layer_idx: int):
        """Start a non-blocking all-gather for prefetching.

        Saves param.data (the local shard) to param._local_shard, allocates
        reusable full-parameter storage, and issues dist.all_gather(async_op=True).
        The actual param.data is NOT updated here — that happens in
        _ensure_gathered when the result is needed.
        """
        if param in self._pending_work or hasattr(param, "_local_shard"):
            return  # already in-flight or already gathered
        param._local_shard = param.data
        full_param_padded = self._full_param_buffer(param)
        output_chunks = list(full_param_padded.chunk(self.world_size))
        gather_input = self._gather_input(param)
        pushed = self._nvtx_push(f"fsdp_prefetch_{direction}_layer_{layer_idx}")
        try:
            work = dist.all_gather(output_chunks, gather_input, async_op=True)
        finally:
            self._nvtx_pop(pushed)
        self._pending_work[param] = (work, output_chunks, gather_input, direction, layer_idx)

    def _ensure_gathered(
        self,
        param: nn.Parameter,
        direction: str | None = None,
        layer_idx: int | None = None,
    ):
        """Block until param.data holds the full weight.

        Three cases:
          1. Prefetch completed/in-flight → wait on Work, use full buffer
          2. No prefetch and not gathered → synchronous all-gather (fallback)
          3. Already gathered (_local_shard exists) → no-op
        """
        if param in self._pending_work:
            work, _, _, pending_direction, pending_layer_idx = self._pending_work.pop(param)
            direction = direction or pending_direction
            layer_idx = pending_layer_idx if layer_idx is None else layer_idx
            pushed = self._nvtx_push(f"fsdp_wait_{direction}_layer_{layer_idx}")
            start = time.perf_counter()
            try:
                work.wait()
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                self._nvtx_pop(pushed)
            if self._profile_waits:
                self._wait_profile_records.append(
                    {
                        "direction": direction,
                        "layer_idx": layer_idx,
                        "wait_ms": elapsed_ms,
                        "prefetched": True,
                    }
                )
            full_weight = self._full_param_view(param)
            param.data = full_weight
        elif not hasattr(param, "_local_shard"):
            param._local_shard = param.data
            full_param_padded = self._full_param_buffer(param)
            output_chunks = list(full_param_padded.chunk(self.world_size))
            gather_input = self._gather_input(param)
            layer_idx = self._param_index.get(param, -1) if layer_idx is None else layer_idx
            direction = direction or "sync"
            pushed = self._nvtx_push(f"fsdp_sync_gather_{direction}_layer_{layer_idx}")
            start = time.perf_counter()
            try:
                dist.all_gather(output_chunks, gather_input)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                self._nvtx_pop(pushed)
            if self._profile_waits:
                self._wait_profile_records.append(
                    {
                        "direction": direction,
                        "layer_idx": layer_idx,
                        "wait_ms": elapsed_ms,
                        "prefetched": False,
                    }
                )
            full_weight = self._full_param_view(param)
            param.data = full_weight

    def _free_weight(self, param: nn.Parameter):
        """Restore param.data to the local shard, freeing the full weight."""
        if not hasattr(param, "_local_shard"):
            return
        full_param_padded = self._shard_info[param]["full_param_padded"]
        if full_param_padded.is_cuda:
            full_param_padded.record_stream(torch.cuda.current_stream(full_param_padded.device))
        param.data = param._local_shard
        del param._local_shard
        _free_storage_(full_param_padded)

    # ── hooks ──────────────────────────────────────────────────────────

    def _forward_pre_hook(self, module: nn.Module, args):
        """Before layer forward: materialise current weight and prefetch ahead."""
        if self._record_forward_order:
            self._observed_forward_order.append(module)
        idx = self._layer_index[id(module)]
        self._ensure_gathered(module.weight, direction="fwd", layer_idx=idx)
        if not self._has_learned_forward_order:
            return
        for offset in range(1, self.prefetch_depth + 1):
            next_idx = idx + offset
            if next_idx >= len(self._ordered_layers):
                break
            self._async_gather_weight(self._ordered_layers[next_idx].weight, "fwd", next_idx)

    def _forward_hook(self, module: nn.Module, args, result):
        """After layer forward: discard full weight, keep only local shard."""
        self._free_weight(module.weight)

    def _full_backward_pre_hook(self, module: nn.Module, args):
        """Before layer backward: materialise full weight, prefetch prev layer.

        Backward traverses layers in reverse order, so we prefetch idx-1.
        """
        idx = self._layer_index[id(module)]
        self._ensure_gathered(module.weight, direction="bwd", layer_idx=idx)
        for offset in range(1, self.prefetch_depth + 1):
            prev_idx = idx - offset
            if prev_idx < 0:
                break
            self._async_gather_weight(self._ordered_layers[prev_idx].weight, "bwd", prev_idx)

    def _post_accumulate_grad_hook(self, param: nn.Parameter):
        """After gradient is computed for a sharded param: reduce-scatter.

        Steps:
          1. Cast gradient to fp32 (master precision), flatten, pad
          2. reduce_scatter: each rank gets SUM of its gradient shard
          3. Divide by world_size to get mean
          4. Restore param.data to local shard (free full weight from backward)
          5. Assign shard-sized gradient to param.grad
        """
        info = self._shard_info[param]

        grad = param.grad.data
        if grad.dtype != torch.float32:
            grad = grad.to(torch.float32)
        grad = grad.flatten()

        padded_numel = info["chunk_size"] * self.world_size
        if padded_numel > info["numel"]:
            grad = torch.cat([grad, grad.new_zeros(padded_numel - info["numel"])])

        grad_chunks = list(grad.chunk(self.world_size))
        local_grad = torch.empty(info["chunk_size"], dtype=torch.float32, device=param.device)
        dist.reduce_scatter(local_grad, grad_chunks, op=dist.ReduceOp.SUM)
        local_grad.div_(self.world_size)

        # Restore fp32 local shard BEFORE assigning grad so shapes & dtypes match
        self._free_weight(param)

        param.grad = local_grad
