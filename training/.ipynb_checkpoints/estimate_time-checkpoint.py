import os
import time
import math
import argparse
import torch
import torch.distributed as dist

def get_int(name, default):
    v = os.environ.get(name, None)
    return int(v) if v is not None else default

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_samples", type=int, required=True)
    p.add_argument("--per_device_batch", type=int, required=True)
    p.add_argument("--grad_accum", type=int, required=True)
    p.add_argument("--measure_updates", type=int, default=20)
    args = p.parse_args()

    world = get_int("WORLD_SIZE", 1)
    rank = get_int("RANK", 0)

    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    effective_batch = world * args.per_device_batch * args.grad_accum
    total_updates = math.ceil(args.num_samples / effective_batch)

    device = torch.device(f"cuda:{get_int('LOCAL_RANK', 0)}" if torch.cuda.is_available() else "cpu")

    d = 8192
    w = torch.randn(d, d, device=device, dtype=torch.bfloat16)
    x = torch.randn(d, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    opt = torch.optim.AdamW([x], lr=1e-4)

    measure_updates = args.measure_updates
    warmup_updates = max(2, measure_updates // 4)

    barrier()
    if rank == 0:
        print(f"WORLD_SIZE={world}")
        print(f"N={args.num_samples}, B={args.per_device_batch}, A={args.grad_accum}")
        print(f"Effective batch per update = {effective_batch}")
        print(f"Estimated optimizer updates per epoch = {total_updates}")
        print(f"Measuring {measure_updates} optimizer updates (with {warmup_updates} warmup)...", flush=True)

    barrier()
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    updates_done = 0
    t0 = time.time()

    for upd in range(measure_updates + warmup_updates):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            y = (x @ w).float().mean()
            y.backward()
        opt.step()
        updates_done += 1

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.time()

    total = t1 - t0
    measured = measure_updates
    measured_time = total * (measured / (measure_updates + warmup_updates))
    sec_per_update = measured_time / measured

    barrier()
    if rank == 0:
        est_seconds = sec_per_update * total_updates
        print(f"Seconds per optimizer update ~= {sec_per_update:.3f}")
        print(f"Estimated 1-epoch time ~= {est_seconds/3600:.2f} hours")
        print(f"Estimated updates/hour ~= {3600/sec_per_update:.1f}")
        print("Note: This is a compute-only proxy. Real training can be slower due to dataloading + tokenization.", flush=True)

if __name__ == "__main__":
    main()
