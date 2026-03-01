import torch
import gc
import os

def clear_gpu():
    """Cleans VRAM GPU memory to avoid OOM errors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("GPU memory cleared.")

def setup_env():
    os.makedirs('models', exist_ok=True)