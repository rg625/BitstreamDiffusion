"""All-reduce bandwidth across every rank of a torchrun job.

Run before trusting a multi-node training run: a job whose NCCL fell back to
TCP sockets trains correctly but slowly, and nothing else would say so. Prints
the transport NCCL chose (with NCCL_DEBUG=INFO) and the measured bus
bandwidth at the gradient size of the token model (~160M params, fp32 grads).
"""
import os
import time

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    n = int(os.environ.get("NCCL_CHECK_NUMEL", 160_000_000))
    x = torch.ones(n, dtype=torch.float32, device="cuda")

    for _ in range(3):                      # warm-up builds the communicators
        dist.all_reduce(x)
    torch.cuda.synchronize()
    iters = 10
    t = time.time()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.time() - t) / iters

    # Correctness: ones summed `world` times per all-reduce, 13 all-reduces.
    expect = float(world) ** 13
    ok = torch.allclose(x[:1000], torch.full_like(x[:1000], expect), rtol=1e-3)
    size = n * 4
    busbw = size * 2 * (world - 1) / world / dt / 1e9
    if rank == 0:
        print(f"[nccl_check] world={world} size={size / 1e9:.2f} GB "
              f"time={dt * 1e3:.1f} ms busbw={busbw:.1f} GB/s correct={bool(ok)}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
