#!/usr/bin/env python3
"""
Utility functions and classes for RL+SMPC Training Pipeline
"""

import numpy as np
import torch
import math
import os

def simple_minmax_normalize(x: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
    """
    Simple min-max normalization: (x - min) / (max - min)
    
    Args:
        x: Input array to normalize
        min_vals: Minimum values for each dimension
        max_vals: Maximum values for each dimension
        
    Returns:
        Normalized array in range [0, 1]
    """
    # Ensure inputs are numpy arrays
    x = np.asarray(x, dtype=np.float32)
    min_vals = np.asarray(min_vals, dtype=np.float32)
    max_vals = np.asarray(min_vals, dtype=np.float32)
    
    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals = np.where(range_vals == 0, 1.0, range_vals)
    
    # Apply min-max normalization
    normalized = (x - min_vals) / range_vals
    
    # Clip to [0, 1] range
    normalized = np.clip(normalized, 0.0, 1.0)
    
    return normalized

def thrust_to_cmd(T_total: float) -> int:
    """
    Convert total thrust [N] into Crazyflie thrust command (0..65535).
    
    Args:
        T_total (float): Desired total thrust in Newtons.
    
    Returns:
        int: Command value in range [0, 65535].
    """
    a2 = 2.130295e-11
    a1 = 1.032633e-6
    a0 = 5.484560e-4
    # per-motor thrust
    f_per = max(0.0, T_total / 4.0)

    # solve quadratic for cmd (positive root)
    disc = (a1/(2*a2))**2 - (a0 - f_per)/a2
    if disc < 0:
        cmd = 0.0
    else:
        cmd = -a1/(2*a2) + math.sqrt(disc)

    # clip to Crazyflie range
    return int(min(65535, max(0, cmd)))


def _select_device() -> torch.device:
    """Select the best available device (CUDA if available, CPU otherwise)"""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        _ = torch.ones(1, device="cuda") * 1.0
        torch.cuda.synchronize()
        return torch.device("cuda")
    except Exception as e:
        print(f"WARNING: CUDA detected but not usable ({e}). Falling back to CPU.")
        return torch.device("cpu")


def setup_device():
    """Setup device and configure PyTorch settings"""
    device = _select_device()
    
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    else:
        # Ensure no accidental CUDA usage when falling back
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.backends.cudnn.benchmark = False
    
    # Conservative default thread caps to avoid CPU oversubscription
    try:
        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
    except Exception:
        pass
    
    return device 