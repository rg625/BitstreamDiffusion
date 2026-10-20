"""Verify a CoBit Python environment before it is trusted with grant hours.

Two halves, because an Isambard login node has no GPU driver:

  build-time  (any node)   the wheel is a CUDA build, of the pinned torch, with
                           sm_90 kernels compiled in. Answerable without a GPU.
  device      (GPU node)   CUDA initialises, the device is Hopper, bf16 works,
                           and a real kernel produces the right numbers.

  python scripts/site/verify_env.py                 # build-time; device half if a GPU is visible
  python scripts/site/verify_env.py --require-gpu   # gate 1: fails if the device half cannot run

The distinction matters because torch.cuda.is_available() is False both for a
CPU-only wheel and for a CUDA wheel on a node without a driver. The first is a
broken env; the second is a login node. Reporting them the same way sends you
to rebuild an env that is fine, or to trust one that is not.
"""
from __future__ import annotations

import argparse
import sys

# Every result in this repo was produced on torch 2.8.0. On aarch64 that
# version exists only as +cu129 (download.pytorch.org publishes no aarch64
# 2.8.0+cu128 wheel), so the CUDA minor version is the one thing that moves.
EXPECTED_TORCH = "2.8.0"
EXPECTED_CUDA = "12.9"


def build_checks(torch) -> list[str]:
    fails = []
    ver = torch.__version__
    print(f"torch {ver}  cuda-build {torch.version.cuda}  python {sys.version.split()[0]}")
    if torch.version.cuda is None:
        fails.append("CPU-only torch wheel (torch.version.cuda is None) -- install from the "
                     "download.pytorch.org cu129 index, not PyPI")
    elif torch.version.cuda != EXPECTED_CUDA:
        fails.append(f"CUDA build {torch.version.cuda}, expected {EXPECTED_CUDA}")
    if ver.split("+")[0] != EXPECTED_TORCH:
        fails.append(f"torch {ver}, expected {EXPECTED_TORCH} (the version behind every result)")
    if torch.version.cuda is not None:
        flags = torch._C._cuda_getArchFlags() or ""
        print(f"      compiled arch flags: {flags}")
        if "sm_90" not in flags:
            fails.append(f"no sm_90 kernels compiled into this wheel ({flags!r})")
    try:
        import triton
        print(f"ok    triton {triton.__version__} (token_sm fast path available)")
    except Exception as e:  # documented eager fallback exists
        print(f"WARN  no triton ({e}); token_sm falls back to the chunked kernel")
    return fails


def device_checks(torch) -> list[str]:
    fails = []
    n = torch.cuda.device_count()
    for i in range(n):
        cap = torch.cuda.get_device_capability(i)
        gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"ok    cuda:{i} {torch.cuda.get_device_name(i)} sm_{cap[0]}{cap[1]} {gb:.0f} GB")
        if cap != (9, 0):
            fails.append(f"cuda:{i} is sm_{cap[0]}{cap[1]}, not sm_90 -- not a GH200 node?")
    if not torch.cuda.is_bf16_supported():
        fails.append("bf16 unsupported")
    else:
        print("ok    bf16 supported")

    # A real kernel, checked against the CPU. "Imports cleanly, then fails at the
    # first CUDA call" is the failure this environment has to rule out, and a
    # wrong-but-silent kernel is worse than a crash.
    g = torch.Generator().manual_seed(0)
    a = torch.randn(256, 256, generator=g)
    b = torch.randn(256, 256, generator=g)
    ref = a @ b
    got32 = (a.cuda() @ b.cuda()).cpu()
    err32 = (got32 - ref).abs().max().item()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        got16 = (a.cuda() @ b.cuda()).float().cpu()
    rel16 = ((got16 - ref).norm() / ref.norm()).item()
    print(f"      matmul fp32 max|err| {err32:.2e}   bf16 autocast rel err {rel16:.2e}")
    # fp32 on Hopper may use TF32 if enabled; 1e-2 absolute over K=256 bounds both.
    if err32 > 1e-2:
        fails.append(f"fp32 matmul disagrees with CPU by {err32:.2e}")
    if rel16 > 2e-2:
        fails.append(f"bf16 matmul relative error {rel16:.2e} (expected ~4e-3)")

    q = torch.randn(2, 4, 64, 32, generator=g)
    k = torch.randn(2, 4, 64, 32, generator=g)
    v = torch.randn(2, 4, 64, 32, generator=g)
    sdpa = torch.nn.functional.scaled_dot_product_attention
    ref_att = sdpa(q, k, v)
    got_att = sdpa(q.cuda(), k.cuda(), v.cuda()).cpu()
    err_att = (got_att - ref_att).abs().max().item()
    print(f"      SDPA max|err| vs CPU {err_att:.2e}")
    if err_att > 1e-3:
        fails.append(f"SDPA disagrees with CPU by {err_att:.2e}")
    return fails


def nccl_check() -> int:
    """Run under torchrun: every rank all-reduces its rank id and checks the sum.

    Exercises NCCL over this node's interconnect before gate 4 relies on it.
    """
    import os

    import torch
    import torch.distributed as dist
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    x = torch.full((1 << 20,), float(dist.get_rank()), device="cuda")
    dist.all_reduce(x)
    want = world * (world - 1) / 2
    ok = bool((x == want).all().item())
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"{'ok  ' if ok else 'FAIL'}  NCCL all_reduce over {world} ranks "
              f"(got {x[0].item()}, want {want}) nccl {torch.cuda.nccl.version()}")
    dist.destroy_process_group()
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-gpu", action="store_true",
                    help="fail if no GPU is visible (use inside the gate-1 job)")
    ap.add_argument("--nccl", action="store_true",
                    help="all-reduce check; launch under torch.distributed.run")
    args = ap.parse_args(argv)
    if args.nccl:
        return nccl_check()

    import torch
    fails = build_checks(torch)
    if torch.version.cuda is not None and torch.cuda.is_available():
        fails += device_checks(torch)
    elif args.require_gpu:
        fails.append("no GPU visible, and --require-gpu was given")
    else:
        print("SKIP  device checks: no GPU visible (login node?). Run the gate-1 job:")
        print("      bash scripts/site/submit.sh scripts/site/verify_gpu.slurm")

    for f in fails:
        print(f"FAIL  {f}")
    print("PASS" if not fails else f"{len(fails)} check(s) failed")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
