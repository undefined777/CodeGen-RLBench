# debug_memory.py
"""Tiny helpers to trace GPU memory step‑by‑step.

Import `log_mem` and sprinkle it at critical spots to find leaks / spikes:

    from debug_memory import log_mem, mem_guard

    log_mem("after model load")

or wrap any block:

    with mem_guard("generate"):
        preds = model.generate(...)

All numbers are GB.
"""

import contextlib
import torch

__all__ = ["log_mem", "mem_guard"]

def _mem_gb():
    alloc = torch.cuda.memory_allocated() / 1024 ** 3
    reserved = torch.cuda.memory_reserved() / 1024 ** 3
    return alloc, reserved


def log_mem(tag: str):
    """Print current allocated / reserved GPU memory (GB) with a tag."""
    if not torch.cuda.is_available():
        print(f"[MEM][{tag}] CUDA not available")
        return
    torch.cuda.synchronize()
    alloc, reserved = _mem_gb()
    print(f"[MEM][{tag}] alloc {alloc:.2f} GB | reserved {reserved:.2f} GB")


@contextlib.contextmanager
def mem_guard(tag: str):
    """Context manager to measure peak memory inside a code block."""
    if not torch.cuda.is_available():
        yield
        return

    torch.cuda.reset_peak_memory_stats()
    start_alloc, _ = _mem_gb()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        end_alloc, reserved = _mem_gb()
        print(
            f"[MEM][{tag}] start {start_alloc:.2f} GB | "
            f"peak {peak:.2f} GB | end {end_alloc:.2f} GB | reserved {reserved:.2f} GB"
        )
