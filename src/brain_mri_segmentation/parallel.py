"""Run several training runs at the same time: each run is a separate process pinned to one GPU slot.

A slot is (GPU index, position); with runs_per_gpu = 2 two runs share each GPU. Every child process trains exactly one
run with the same script (so results are identical to sequential training) and sees only its GPU through
CUDA_VISIBLE_DEVICES. Failed runs are reported at the end; finished runs are never retrained.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time

LOGGER = logging.getLogger(__name__)
POLL_SECONDS = 5


def visible_gpu_indices() -> list[int]:
    import torch

    return list(range(torch.cuda.device_count()))


def run_in_parallel(
    run_names: list[str], gpu_indices: list[int], runs_per_gpu: int, command_for_run, log_path_for_run
) -> list[str]:
    """Launch command_for_run(name) for every run across the GPU slots; return the names of runs that failed."""
    slots = [gpu for gpu in gpu_indices for _ in range(runs_per_gpu)]
    pending = list(run_names)
    active: dict[int, tuple[str, subprocess.Popen, object]] = {}
    failed: list[str] = []
    while pending or active:
        for slot, gpu in enumerate(slots):
            if slot not in active and pending:
                run_name = pending.pop(0)
                log_file = open(log_path_for_run(run_name), "a")
                environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                process = subprocess.Popen(
                    command_for_run(run_name), env=environment, stdout=log_file, stderr=subprocess.STDOUT
                )
                active[slot] = (run_name, process, log_file)
                LOGGER.info("started %s on GPU %d (slot %d, pid %d)", run_name, gpu, slot, process.pid)
        for slot, (run_name, process, log_file) in list(active.items()):
            if process.poll() is not None:
                log_file.close()
                del active[slot]
                if process.returncode == 0:
                    LOGGER.info("finished %s", run_name)
                else:
                    failed.append(run_name)
                    LOGGER.error("%s failed (exit code %d); see its console log", run_name, process.returncode)
        time.sleep(POLL_SECONDS)
    return failed
