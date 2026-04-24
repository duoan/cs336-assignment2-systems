import timeit

import torch
import torch.cuda.nvtx as nvtx

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy, clip_gradient

def run_benchmark(
    batch_size: int=4,
    vocab_size: int=10_000,
    context_length: int=256,
    d_model: int=768,
    d_ff: int=3072,
    num_layers: int=12,
    num_heads: int=12,
    warmup_steps: int=5,
    repetion_steps: int=10,
    dtype: torch.dtype=torch.bfloat16
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
    optimizer = AdamW(model.parameters())
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        logits = model(inputs)
        losses = cross_entropy(logits, targets)
        losses.backward()
        optimizer.step()
    
    print(f"Warmup done, will run benchmark with {repetion_steps} repetion steps")
    
    events = [dict() for _ in range(repetion_steps)]
    mixed_percision_enabled = dtype==torch.bfloat16
    
    torch.cuda.memory._record_memory_history(max_entries=1_000_000)
    
    def trace_handler(profiler):
        print(f"\n[Step {profiler.step_num}] Profiler Report:")
        print(profiler.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
        print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=trace_handler,
        with_flops=True,
        with_stack=True,
        profile_memory=True,
        record_shapes=True,
        with_modules=True) as profiler:
        for i in range(repetion_steps):
            start_fw_time = timeit.default_timer()
            
            with torch.cuda.nvtx.range("forward_pass"):
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=mixed_percision_enabled):
                    logits = model(inputs)
            with torch.cuda.nvtx.range("loss_calculation"):
                losses = cross_entropy(logits, targets)
            events[i]['fw'] = timeit.default_timer() - start_fw_time
            
            start_bw_time = timeit.default_timer()   
            with torch.cuda.nvtx.range("backward_pass"):
                losses.backward()
            events[i]['bw'] = timeit.default_timer() - start_bw_time
            
            start_op_time = timeit.default_timer()   
            with torch.cuda.nvtx.range("optimizer.step"):
                optimizer.step()
                optimizer.zero_grad()
            events[i]['op'] = timeit.default_timer() - start_op_time
            
            torch.cuda.synchronize()
            events[i]['total'] = timeit.default_timer() - start_fw_time
            profiler.step()
            
            
            
    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)
    
    from pandas import DataFrame
    df = DataFrame.from_records(events)
    print(df.to_string())
    print("="*100)
    print(df.describe())
    

if __name__ == "__main__":
    run_benchmark()