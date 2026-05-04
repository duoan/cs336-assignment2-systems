import os
from torch._tensor import Tensor

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
def my_broadcast(tensor: torch.Tensor, src: int=0):
    if dist.get_rank() == src:
        for dst in range(dist.get_world_size()):
            if dst != src:
                dist.send(tensor, dst=dst)
    else:
        dist.recv(tensor, src=src)

def my_scatter(tensor: torch.Tensor, scatter_list: tuple[torch.Tensor, ...], src=0):
    if dist.get_rank() == src:
        if scatter_list is None or len(scatter_list) != dist.get_world_size():
            raise ValueError("Scatter list must contain 'world_size' tensors.")
        
        # keep our own chunk
        tensor.copy_(scatter_list[src])

        # send the rest
        for dst in range(dist.get_world_size()):
            if dst != src:
                dist.send(scatter_list[dst], dst=dst)
    else:
        # receive chunk from source rank
        dist.recv(tensor, src=src)

def my_gather(tensor: torch.Tensor, gather_list: tuple[torch.Tensor, ...], dst=0):
    if dist.get_rank() == dst:
        # copy our own tensor into list
        gather_list[dst].copy_(tensor)
        
        # receive from everyone else
        for src in range(dist.get_world_size()):
            if src != dst:
                dist.recv(gather_list[src], src=src)
    else:
        # send our tensor to distenation
        dist.send(tensor, dst=dst)

def my_reduce(tensor: torch.Tensor, op: dist.ReduceOp=dist.ReduceOp.SUM, dst=0):
    if dist.get_rank() == dst:
        other_tensor = torch.empty_like(tensor)
        for src in range(dist.get_world_size()):
            if src != dst:
                dist.recv(other_tensor, src=src)
                if op == dist.ReduceOp.SUM:
                    tensor.add_(other_tensor)
    else:
        dist.send(tensor, dst=dst)

def my_all_reduce(tensor: torch.Tensor):
    my_reduce(tensor, dst=0)
    my_broadcast(tensor, src=0)
    
def my_all_gather(tensor: torch.Tensor):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Ring topology: every rank sends clockwise and receives counter-clockwise.
    #
    # world_size = 4:
    #
    #           send
    #      0 --------> 1
    #      ^           |
    # recv |           | send
    #      |           v
    #      3 <-------- 2
    #           recv
    #
    # For this rank:
    #   right: the next rank clockwise, where we send.
    #   left:  the previous rank, where we receive from.
    right = (rank + 1) % world_size
    left = (rank - 1 + world_size) % world_size

    # All-gather assumes each rank already owns one final chunk. In ring
    # all-reduce, this is true after reduce-scatter: rank r owns chunk r.
    #
    # The goal here is not to add anything. The goal is only to circulate final
    # chunks until every rank has every chunk.
    chunks = list(torch.chunk(tensor, world_size))
    recv_buffer = torch.empty_like(chunks[0])

    # Example with world_size = 4 and rank = 2.
    #
    # Before all-gather, rank 2's important local result is chunk 2:
    #
    #   chunk owners:
    #     rank 0 owns chunk 0
    #     rank 1 owns chunk 1
    #     rank 2 owns chunk 2
    #     rank 3 owns chunk 3
    #
    # Each step moves already-finished chunks clockwise:
    #
    #   step 0:
    #     rank 2 sends chunk 2 -> rank 3
    #     rank 2 receives chunk 1 <- rank 1
    #
    #   step 1:
    #     rank 2 sends chunk 1 -> rank 3
    #     rank 2 receives chunk 0 <- rank 1
    #
    #   step 2:
    #     rank 2 sends chunk 0 -> rank 3
    #     rank 2 receives chunk 3 <- rank 1
    #
    # Why send_idx = rank - step?
    #   At step 0, this rank sends its own reduced chunk. On later steps, it
    #   forwards the chunk it received in the previous step.
    #
    # Why recv_idx = rank - step - 1?
    #   The left neighbor owns the previous chunk, and each step walks backward
    #   through chunk ids as data rotates around the ring.
    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size

        # Post both operations before waiting. If every rank used blocking send
        # first, the ring could deadlock because everyone would wait to send.
        req_send = dist.isend(chunks[send_idx], dst=right)
        req_recv = dist.irecv(recv_buffer, src=left)

        req_send.wait()
        req_recv.wait()

        # All-gather distributes final chunks, so receiving is a copy, not add.
        chunks[recv_idx].copy_(recv_buffer)


def my_reduce_scatter(tensor: torch.Tensor):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Same ring topology as all-gather: send clockwise, receive from the left.
    #
    # world_size = 4:
    #
    #      0 ---> 1
    #      ^      |
    #      |      v
    #      3 <--- 2
    #
    # Reduce-scatter is the first half of ring all-reduce. Its job is:
    #
    #   1. Split every rank's tensor into world_size chunks.
    #   2. Move chunks around the ring.
    #   3. Add matching chunks together as they arrive.
    #   4. End with rank i holding the reduced value for chunk i.
    #
    # Unlike all-gather, this phase is about accumulation, not distribution.
    right = (rank + 1) % world_size
    left = (rank - 1 + world_size) % world_size

    # Chunk i is the slice whose final owner is rank i.
    #
    # Example:
    #   tensor = [chunk0 | chunk1 | chunk2 | chunk3]
    #
    # At the end of reduce-scatter:
    #   rank 0 should hold reduced chunk0
    #   rank 1 should hold reduced chunk1
    #   rank 2 should hold reduced chunk2
    #   rank 3 should hold reduced chunk3
    chunks = list(torch.chunk(tensor, world_size))
    recv_buffer = torch.empty_like(chunks[0])

    # Example with world_size = 4 and rank = 2.
    #
    # Step table for rank 2:
    #
    #   step | send_idx | recv_idx | meaning
    #   -----+----------+----------+------------------------------------------
    #     0  |    1     |    0     | send chunk1, receive chunk0 and add it
    #     1  |    0     |    3     | send chunk0, receive chunk3 and add it
    #     2  |    3     |    2     | send chunk3, receive chunk2 and add it
    #
    # Why send_idx = rank - step - 1?
    #   At step 0, rank r starts by sending the chunk just before its final
    #   owner chunk. On later steps, this rank forwards the chunk it received
    #   and reduced in the previous step.
    #
    # Why recv_idx = rank - step - 2?
    #   Because the left neighbor is one rank behind us, and we offset the
    #   schedule so the final receive lands on chunk r.
    #
    # The modulo wraps around the ring:
    #   for rank 0, recv_idx at step 0 is -2 % world_size, i.e. chunk world_size-2.
    for step in range(world_size - 1):
        send_idx = (rank - step - 1) % world_size
        recv_idx = (rank - step - 2) % world_size

        # Non-blocking send/recv keeps the whole ring from deadlocking.
        req_send = dist.isend(chunks[send_idx], dst=right)
        req_recv = dist.irecv(recv_buffer, src=left)

        req_send.wait()
        req_recv.wait()

        # Reduce-scatter accumulates partial sums into the chunk that arrived.
        chunks[recv_idx].add_(recv_buffer)
    

def my_ring_all_reduce(tensor: torch.Tensor):
    # Stage 1: reduce-scatter
    my_reduce_scatter(tensor)
    # Stage 2: All gather
    my_all_gather(tensor)


def alternate_ring_all_reduce(tensor: torch.Tensor):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    right = (rank + 1) % world_size
    left = (rank - 1 + world_size) % world_size
    # y <- x^(i)
    # tensor itself stores the running partial sum y.
    send_buffer = tensor.clone()
    recv_buffer = torch.empty_like(tensor)
    for _ in range(1, world_size):
        # At step t, rank i sends the full tensor x^((i - t + 1) mod N)
        # that it currently holds in send_buffer.
        req_send = dist.isend(send_buffer, dst=right)
        # It receives the full tensor x^((i - t) mod N) from the left.
        req_recv = dist.irecv(recv_buffer, src=left)
        req_send.wait()
        req_recv.wait()
        # y <- y + received x
        tensor.add_(recv_buffer)
        # The tensor we just received is what we forward in the next step.
        send_buffer.copy_(recv_buffer)
        

def all_to_all(output_tensor_list: list[torch.Tensor], input_tensor_list: list[torch.Tensor]):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # all to all
    # input_tensor_list
    # rank#0: 1, 1, 1, 1
    # rank#1: 2, 2, 2, 2
    # rank#2: 3, 3, 3, 3
    # rank#3: 4, 4, 4, 4

    # output_tensor_list:
    # rank#0: 1, 2, 3, 4
    # rank#1: 1, 2, 3, 4
    # rank#2: 1, 2, 3, 4
    # rank#3: 1, 2, 3, 4

    assert len(output_tensor_list) == world_size
    assert len(input_tensor_list) == world_size

    for t in input_tensor_list:
        assert t.is_contiguous()
    
    for t in output_tensor_list:
        assert t.is_contiguous()

    # local rank: input[rank] -> output[rank]
    if output_tensor_list[rank].data_ptr() != input_tensor_list[rank].data_ptr():
        output_tensor_list[rank].copy_(input_tensor_list[rank])

    # Pairwise exchange to avoid deedlokc
    for peer in range(world_size):
        if peer == rank:
            continue
        
        if rank < peer:
            # Lower rank sends first, then receives.
            dist.send(input_tensor_list[peer], dst=peer)
            dist.recv(output_tensor_list[peer], src=peer)
        else:
            # Higher rank receives first, then sends.
            dist.recv(output_tensor_list[peer], src=peer)
            dist.send(input_tensor_list[peer], dst=peer)

def app(rank, world_size: int, input_size: int):
    setup(rank, world_size)
    
    if rank == 0:
        print("=" * 100)
    
    data = torch.randn(input_size)
    print(f"rank {rank} data (before broadcast): {data}")
    my_broadcast(data, src=0)
    
    print(f"rank {rank} data (after broadcast): {data}")
    dist.barrier()
    
    if rank == 0:
        print("=" * 100)
    scatter_tensor = torch.empty(data.size(0) // world_size)
    my_scatter(scatter_tensor, scatter_list=data.chunk(world_size), src=0)
    print(f"rank {rank} scatter_tensor (after scatter): {scatter_tensor}")
    dist.barrier()
    
    if rank == 0:
        print("=" * 100)
    gather_list = [torch.zeros_like(scatter_tensor) for _ in range(world_size)]
    my_gather(data.chunk(world_size)[dist.get_rank()], gather_list=gather_list, dst=0)
    print(f"rank {rank} gather_list (after gather): {gather_list}")
    dist.barrier()
    
    if rank == 0:
        print("=" * 100)
    data = torch.tensor([dist.get_rank() + 1 for _ in range(4)])
    print(f"rank {rank} data (before reduce): {data}")
    my_reduce(data, dst=0)
    print(f"rank {rank} data (after reduce): {data}")
    dist.barrier()
    
    if rank == 0:
        print("=" * 100)
        
    data = torch.tensor([dist.get_rank() + 1 for _ in range(4)])
    print(f"rank {rank} data (before all-reduce): {data}")
    my_all_reduce(data)
    print(f"rank {rank} data (after all-reduce): {data}")
    dist.barrier()
    
    if rank == 0:
        print("=" * 100) 
    data = torch.tensor([dist.get_rank() + 1 for _ in range(4)])
    print(f"rank {rank} data (before ring-all-reduce): {data}")
    alternate_ring_all_reduce(data)
    print(f"rank {rank} data (after ring-all-reduce): {data}")
    dist.barrier()

    if rank == 0:
        print("=" * 100) 
    # Example:
    # rank 0: [1, 1, 1, 1]
    # rank 1: [2, 2, 2, 2]
    # rank 2: [3, 3, 3, 3]
    # rank 3: [4, 4, 4, 4]
    x = torch.full((world_size,), rank + 1, dtype=torch.float32)

    # Split input into world_size chunks.
    # input_chunks[i] is sent to rank i.
    input_chunks = list(x.chunk(world_size, dim=0))

    # Allocate output chunks.
    # output_chunks[i] receives data from rank i.
    output_chunks = [
        torch.empty_like(input_chunks[0])
        for _ in range(world_size)
    ]

    print(f"rank {rank} input_chunks (before all-to-all: {torch.cat(input_chunks)})")
    all_to_all(output_chunks, input_chunks)
    print(f"rank {rank} output_chunks (after all to all): {torch.cat(output_chunks)}")
    dist.barrier()

    dist.destroy_process_group()
    

def main():
    world_size = 4
    input_size = 16
    
    print(f"Testing {world_size=} {input_size=}")
    
    mp.spawn(
        fn=app,
        args=(
            world_size,
            input_size,
        ),
        nprocs=world_size,
        join=True,
    )
    
if __name__ == "__main__":
    main()