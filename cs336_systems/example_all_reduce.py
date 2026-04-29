import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import modal

app = modal.App("dist_comm_benchmark")

BACKEND = "gloo" if modal.is_local else "nccl"


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(BACKEND, rank=rank, world_size=world_size)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def distributed_demo(rank, world_size, input_size):
    setup(rank, world_size)
    data = torch.randn((input_size,), device=get_device())
    # print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False)
    # print(f"rank {rank} data (after all-reduce): {data}")


@app.local_entrypoint()
def main():
    world_sizes = [2, 4, 6]
    input_sizes = [
        1024 * 1024,  # 1MB
        10 * 1024 * 1024,  # 10MB
        100 * 1024 * 1024,  # 100MB
        1024 * 1024 * 1024,  # 1GB
    ]
    for world_size in world_sizes:
        for input_size in input_sizes:
            print(f"Testing {world_size=} {input_size=}")
            mp.spawn(
                fn=distributed_demo,
                args=(
                    world_size,
                    input_size,
                ),
                nprocs=world_size,
                join=True,
            )


if __name__ == "__main__":
    main()
