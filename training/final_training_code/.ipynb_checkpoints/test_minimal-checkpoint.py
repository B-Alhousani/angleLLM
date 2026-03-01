import os
import sys

print(f"[0] Script started, PID={os.getpid()}", flush=True)
sys.stdout.flush()
sys.stderr.flush()

try:
    print("[1] Importing torch...", flush=True)
    import torch
    
    print(f"[2] Torch version: {torch.__version__}", flush=True)
    print(f"[3] CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"[4] CUDA devices: {torch.cuda.device_count()}", flush=True)
    
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print(f"[5] Rank {rank}/{world_size}, Local Rank {local_rank}", flush=True)
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_name = torch.cuda.get_device_name(local_rank)
        print(f"[6] Rank {rank} using GPU {local_rank}: {device_name}", flush=True)
    
    print(f"[7] Rank {rank} - All checks passed!", flush=True)
    
except Exception as e:
    print(f"[ERROR] {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
