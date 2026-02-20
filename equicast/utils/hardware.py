"""Hardware info for experiment tracking."""

import os
import socket

import torch


def get_hardware_info() -> dict:
    info = {
        "host": socket.gethostname(),
        "num_cpus": os.cpu_count(),
        "num_gpus": torch.cuda.device_count(),
    }
    if info["num_gpus"] > 0:
        info["gpu_type"] = torch.cuda.get_device_name(0)
    return info
