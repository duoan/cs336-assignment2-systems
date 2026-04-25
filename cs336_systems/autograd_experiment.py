from decimal import Context
from multiprocessing import context
import torch
from cs336_basics.model import RMSNorm, RotaryEmbedding, TransformerBlock

total_size_bytes = 0

def pack_hook(t: torch.Tensor):
    if isinstance(t, torch.nn.Parameter):
        return t
    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    x = torch.randn((4, 512, 2560), requires_grad=True, device="cuda")
    ln = torch.compile(RMSNorm(x.size(-1), device="cuda"))

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = ln(x)
        y.sum().backward()
    
    del x, ln, y
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("=" * 100)

    d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048
    positional_encoder = RotaryEmbedding(dim=d_model // num_heads, context_length=context_length)
    block = TransformerBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads, positional_encoder=positional_encoder).to("cuda")
    

    block = torch.compile(block, fullgraph=True)
    x = torch.randn((4, context_length, d_model), requires_grad=True, device="cuda")

    total_size_bytes = 0
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = block(x)

    print(f"Total size of saved tensors in single TansformerBlock: {total_size_bytes / (1024**2):.2f} MiB")


    def four_blocks(x):
        x = block(x)
        x = block(x)
        x = block(x)
        x = block(x)
        return x
    
    total_size_bytes = 0
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = four_blocks(x)

    print(f"Total size of saved tensors in 4 TansformerBlocks: {total_size_bytes / (1024**2):.2f} MiB")


    from torch.utils.checkpoint import checkpoint

    def two_blocks(x):
        x = block(x)
        x = block(x)
        return x
    
    def four_blocks_checkpoint(x):
        x = checkpoint(two_blocks, x, use_reentrant=False)
        x = checkpoint(two_blocks, x, use_reentrant=False)
        return x

    total_size_bytes = 0
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = four_blocks_checkpoint(x)

    print(f"Total size of saved tensors in 4 TansformerBlocks with checkpointing: {total_size_bytes / (1024**2):.2f} MiB")
