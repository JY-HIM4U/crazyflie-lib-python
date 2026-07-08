#!/usr/bin/env python3
"""Crazyflie flight controller using a JAX/BRACE-trained policy.

Loads a Flax checkpoint from the BRACE training pipeline and deploys it on
real hardware via the Crazyflie radio link.  The observation format, LQR
matrices, and belief filter match the PhoenixPhysicalJAX environment used
during training.

Key design choices
------------------
- **Attitude-control LQR** – u_hat[1:3] are *target roll/pitch/yaw angles*
  (not rates).  The Crazyflie's on-board PID tracks these.
- **simple_states 8-D observation** – [x, y, vx, vy, gx, gy, rel_x, rel_y]
  normalised by [W, W, 1, 1, W, W, R, R], then 5-D belief appended → 13-D.
- **JAX inference** – policy forward-pass is pure JAX; no PyTorch dependency.
"""
import logging
import time
from threading import Thread, Barrier
import math
import sys
import os
from collections import defaultdict
import threading
import csv
import pickle
import json

sys.path.insert(0, '/home/realm/jaeyoun/Phoenix_smpc/BRACE/src')
sys.path.insert(0, '/home/realm/jaeyoun/Phoenix_smpc/Stochastic_Hierarchies_Code')
sys.path.insert(0, '/home/realm/jaeyoun/Phoenix_smpc')

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from scipy.linalg import expm as scipy_expm

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig

# IMM belief filter from BRACE (JAX-native)
from model.belief import (
    default_config as default_imm_config,
    init_state as init_imm_state,
    predict as imm_predict,
    update as imm_update,
    get_features as imm_get_features,
)

# =============================================================================
# Config — loaded from the checkpoint's config.json
# =============================================================================
# Switch which trained model gets loaded. Each architecture differs:
#   M0        — base (8-D obs, scratch) + residual + gate, MU/SIGMA scaled
#   M1        — base only, no belief, no residual          (13-D obs+zeros)
#   M2        — base (13-D obs+zeros) + residual + gate, MU/SIGMA scaled (frozen base)
#   M5        — base only, belief CONCAT into obs          (13-D obs+belief)
#   M1_term   — like M1, retrained with terminal-V^h reward (seed 4)
#   M0nl_term — like M0, retrained with terminal-V^h + nonlinear predictor (seed 4)
MODEL_TAG = "M0nl_term"   # one of: "M0", "M1", "M2", "M5", "M1_term", "M0nl_term"
# MODEL_TAG = "M1_term"   # one of: "M0", "M1", "M2", "M5", "M1_term", "M0nl_term"

_MODELS_ROOT = "/home/realm/jaeyoun/crazyflie-lib-python/models"
MODEL_DIRS = {
    "M0":        f"{_MODELS_ROOT}/trained_20260430_002921_s0_M0_res_track_rew_scratch_PhoenixPhysicalJAX_DCBF_belief_residual_seed0",
    "M1":        f"{_MODELS_ROOT}/trained_20260430_003922_s0_M1_nobelief_PhoenixPhysicalJAX_DCBF_nobelief_seed0",
    "M2":        f"{_MODELS_ROOT}/trained_s0_M2_res_track_rew_frz_20260428_143340_PhoenixPhysicalJAX_DCBF_belief_residual_seed0",
    "M5":        f"{_MODELS_ROOT}/trained_20260430_013140_s0_M5_concat_PhoenixPhysicalJAX_DCBF_belief_seed0",
    "M1_term":   f"{_MODELS_ROOT}/trained_20260504_191611_s4_M1_term_nobelief_PhoenixPhysicalJAX_DCBF_nobelief_termVh_seed4",
    "M0nl_term": f"{_MODELS_ROOT}/trained_20260504_191611_s4_M0nl_term_res_track_rew_scratch_nlpred_PhoenixPhysicalJAX_DCBF_belief_residual_termVh_seed4",
}
MODEL_DIR       = MODEL_DIRS[MODEL_TAG]
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_checkpoint.pkl")

with open(os.path.join(MODEL_DIR, "config.json"), "r") as _f:
    _cfg = json.load(_f)

MAP_NAME       = _cfg.get("MAP_NAME", "ra_jy2")
ACTION_SCALE   = float(_cfg.get("ACTION_SCALE", 0.1))
MU_SCALE       = float(_cfg.get("MU_SCALE", 20.0))
SIGMA_SCALE    = float(_cfg.get("SIGMA_SCALE", 5.0))
GATE_ACTIVATION = _cfg.get("GATE_ACTIVATION", "none")
OBS_MODE       = _cfg.get("OBS_MODE", "simple_states")
SENSING_RADIUS = float(_cfg.get("SENSING_RADIUS", 0.5))
BELIEF_MODE    = _cfg.get("BELIEF_MODE", "residual")  # 'residual', 'concat', 'none'
USE_RESIDUAL   = bool(_cfg.get("USE_RESIDUAL", True))
BELIEF_DIM     = 5
ACTION_SCALE =0.08 
RESIDUAL_SCALE= 0.25 # 0.5
print(f"[config] MODEL_TAG={MODEL_TAG}, BELIEF_MODE={BELIEF_MODE}, "
      f"USE_RESIDUAL={USE_RESIDUAL}, MU_SCALE={MU_SCALE}, SIGMA_SCALE={SIGMA_SCALE}")

# =============================================================================
# Map loading (ra_jy2)
# =============================================================================
_MAP_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "Phoenix_smpc", "BRACE", "src", "env"
)
# Normalise so os.path.join produces a clean path
_MAP_DIR = os.path.normpath(_MAP_DIR)

_WORLD_SIZE = 2.0

def _load_map(map_name):
    map_path = os.path.join(_MAP_DIR, f"{map_name}_map.npz")
    data = np.load(map_path)
    safe = np.asarray(data["safe"], dtype=np.float32)
    target = np.asarray(data["target"], dtype=np.float32)
    avoid = np.asarray(data["avoid"], dtype=np.float32)
    dist_field = np.asarray(data["dist_field"], dtype=np.float32)
    map_size = safe.shape[0]
    grid_res = _WORLD_SIZE / map_size

    tys, txs = np.where(target > 0.5)
    goal_x = float(np.mean(txs) * grid_res + grid_res / 2)
    goal_y = float(np.mean(tys) * grid_res + grid_res / 2)

    start_overrides = {
        "ra_2": [0.2, 0.3],
        "ra_jy": [0.4, 0.4],
        "ra_jy2": [1.0, 1.825],
        "slit_10": [1.0, 1.75],
    }
    start = start_overrides.get(map_name, [0.2, 0.3])

    return {
        "safe": safe, "target": target, "avoid": avoid,
        "dist_field": dist_field,
        "map_size": map_size, "grid_res": grid_res,
        "world_size": _WORLD_SIZE,
        "goal_xy": np.array([goal_x, goal_y], dtype=np.float64),
        "start_xy": np.array(start, dtype=np.float64),
    }

_MAP = _load_map(MAP_NAME)
GOAL_XY   = _MAP["goal_xy"]
START_XY  = _MAP["start_xy"]

print(f"[config] MAP={MAP_NAME}, GOAL_XY={GOAL_XY}, START_XY={START_XY}, "
      f"ACTION_SCALE={ACTION_SCALE}, OBS_MODE={OBS_MODE}")

# =============================================================================
# LQR system — matches PhoenixPhysicalJAX._build_system()
# =============================================================================
_MASS    = 0.027
_GRAVITY = 9.81
_DRAG    = 0.006          # tuned for real CF; NOT the old 0.1735
_T_MAX   = _MASS * _GRAVITY * 2.25
_TAU_ATT = 0.005          # attitude-coded model time constant
_DT_LQR  = 0.025          # 25 ms per LQR step
_HORIZON = 20             # 20 × 0.025 = 0.5 s

def _build_lqr_system():
    """Build discretised A, B, Q, R matching PhoenixPhysicalJAX."""
    I3 = np.eye(3); Z3 = np.zeros((3, 3)); Z34 = np.zeros((3, 4))
    inv_tau = 1.0 / _TAU_ATT
    rate_block = np.zeros((3, 4))
    rate_block[0:3, 1:4] = inv_tau * np.eye(3)
    A_c = np.block([
        [Z3,  I3,                    Z3],
        [Z3, -(_DRAG / _MASS) * I3, np.array([[0, _GRAVITY, 0],
                                               [-_GRAVITY, 0, 0],
                                               [0, 0, 0]])],
        [Z3,  Z3,                   -inv_tau * I3],
    ])
    B_c = np.block([
        [Z34],
        [np.array([[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [(1 / _MASS) * _T_MAX, 0, 0, 0]])],
        [rate_block],
    ])
    n, m = 9, 4
    aug = np.block([[A_c, B_c], [np.zeros((m, n)), np.zeros((m, m))]])
    result = scipy_expm(aug * _DT_LQR)
    A_dt = result[:n, :n]
    B_dt = result[:n, n:n + m]
    Q = np.zeros((9, 9))
    Q[:3, :3]  = 4 * np.eye(3)   # position
    Q[3:6, 3:6] = 1 * np.eye(3)  # velocity (lowered 2→1: less tilt-fight)
    Q[6:, 6:]   = np.eye(3)      # attitude
    R = 0.01 * np.eye(4)
    return A_dt, B_dt, Q, R

A_dt, B_dt, Q_lqr, R_lqr = _build_lqr_system()


# ── u_hat[0] saturation bounds ──────────────────────────────────────────────
# Mirror BRACE training (phoenix_physical_jax.py:_compute_u_hat0_bounds).
# The on-board attitude cascade clamps each motor's PWM to [35000, 55000],
# i.e. physical per-motor thrust ∈ [0.251 N, 0.487 N] (≈ 0.95 mg .. 1.84 mg).
# In u_hat[0] units (= (T_total − m·g) / (m·g·THRUST2WEIGHT_RATIO)):
#   _U_HAT0_MIN ≈ -0.023,  _U_HAT0_MAX ≈ +0.373
# Without this clip the nonlinear predictor lets T_total go negative when
# the LQR over-brakes a climb, which flips ax/ay sign and corrupts the IMM
# residual — exactly the bug the BRACE commit (9573b20) calls out.
def _compute_u_hat0_bounds():
    a2 = 2.130295e-11; a1 = 1.032633e-6; a0 = 5.484560e-4
    def _pwm_to_T(cmd):
        return 4.0 * (a0 + a2 * cmd * cmd + a1 * cmd)
    T_min, T_max_real = _pwm_to_T(35000.0), _pwm_to_T(55000.0)
    T_max_lqr = _MASS * _GRAVITY * 2.25       # THRUST2WEIGHT_RATIO
    return ((T_min - _MASS * _GRAVITY) / T_max_lqr,
            (T_max_real - _MASS * _GRAVITY) / T_max_lqr)

_U_HAT0_MIN, _U_HAT0_MAX = _compute_u_hat0_bounds()


def nonlinear_step(state_9, u_hat, dt=_DT_LQR, n_substeps=4):
    """Closed-form nonlinear forward integration of one LQR-step's dynamics.

    Used ONLY for the belief residual (state_9_post − x_pred_next). The LQR
    controller still uses the linear A_dt/B_dt for gain scheduling.

    Models:
      - Trig coupling: vx_dot = (T/m)·sin(pitch)·cos(roll)
                       vy_dot = -(T/m)·sin(roll)
                       vz_dot = (T/m)·cos(pitch)·cos(roll) − g
      - Linear drag (same as LQR's _DRAG/_MASS).
      - Attitude relaxation: rpy_dot = (target − rpy)/τ_att.
      - All in Phoenix convention (positive pitch = nose down → +x accel).

    Args
    ----
    state_9 : np.ndarray (9,) [x, y, z, vx, vy, vz, roll, pitch, yaw]
    u_hat   : np.ndarray (4,) [thrust_frac, target_roll, target_pitch, target_yaw]
    dt      : LQR step duration
    n_substeps : RK1 substeps per LQR step (4 is plenty for 25 ms)
    """
    h = dt / n_substeps
    s = np.array(state_9, dtype=np.float64).copy()
    target_rpy = np.asarray(u_hat[1:4], dtype=np.float64)
    inv_tau = 1.0 / _TAU_ATT
    drag_per_m = _DRAG / _MASS
    # Clip u_hat[0] to the SAME range the controller uses ([-0.3, +1.0]),
    # not BRACE's tighter [-0.023, +0.373] PWM-cascade bound. Reason: the
    # CF firmware accepts send_setpoint with PWM all the way down to 0
    # (verified by past flight data: thrust_pwm range was [0, 65535]),
    # so real motors DO go below the PWM 35000 floor that BRACE's
    # attitude_cascade enforced in simulation. Using a tighter clip here
    # would make the predictor under-predict descent and bias the IMM
    # residual. This clip's only job is to prevent T_total < 0 (which
    # would flip ax/ay sign): with [-0.3, +1.0], T_total ∈ [0.086, 0.86] N
    # — always positive.
    u0 = float(np.clip(u_hat[0], -0.3, 1.0))
    T_total = _MASS * _GRAVITY + u0 * _T_MAX
    T_over_m = T_total / _MASS
    for _ in range(n_substeps):
        roll, pitch = s[6], s[7]
        c_r = math.cos(roll);  s_r = math.sin(roll)
        c_p = math.cos(pitch); s_p = math.sin(pitch)
        # Translational accel — full nonlinear thrust projection at yaw=0
        ax = T_over_m * s_p * c_r           - drag_per_m * s[3]
        ay = -T_over_m * s_r                - drag_per_m * s[4]
        az = T_over_m * c_p * c_r - _GRAVITY - drag_per_m * s[5]
        # Attitude relaxation toward commanded target
        att_dot = (target_rpy - s[6:9]) * inv_tau
        # Forward Euler update
        s[0:3] += s[3:6] * h
        s[3:6] += np.array([ax, ay, az]) * h
        s[6:9] += att_dot * h
    return s


def compute_lqt_gains(A, B, Q, R, r_seq, Q_terminal=None):
    """Finite-horizon LQT backward recursion (numpy)."""
    nx = A.shape[0]
    horizon = r_seq.shape[1] - 1
    terminal_Q = Q if Q_terminal is None else Q_terminal
    P = [None] * (horizon + 1)
    s = [None] * (horizon + 1)
    K = [None] * horizon
    F = [None] * horizon
    P[horizon] = terminal_Q
    s[horizon] = -terminal_Q @ r_seq[:, horizon]
    Bt = B.T; At = A.T
    for k in range(horizon - 1, -1, -1):
        M = R + Bt @ P[k + 1] @ B
        F[k] = np.linalg.solve(M, Bt)
        K[k] = F[k] @ P[k + 1] @ A
        Acl = A - B @ K[k]
        P[k] = Q + At @ P[k + 1] @ Acl
        s[k] = -Q @ r_seq[:, k] + Acl.T @ s[k + 1]
    return K, F, s

def lqt_control_step(K_k, F_k, s_next, x_k):
    return -K_k @ x_k - F_k @ s_next

# =============================================================================
# Nearest-obstacle relative vector (matches PhoenixPhysicalJAX)
# =============================================================================
def nearest_obstacle_rel(wx, wy, dist_field, map_size, grid_res, sensing_radius):
    """Compute relative vector to nearest obstacle, gated by sensing_radius."""
    cpx = (wx / grid_res) - 0.5
    cpy = (wy / grid_res) - 0.5
    cpx = np.clip(cpx, 0.0, map_size - 1.0)
    cpy = np.clip(cpy, 0.0, map_size - 1.0)

    def _bilin(field, fx_pix, fy_pix):
        ix0 = int(np.floor(fx_pix))
        iy0 = int(np.floor(fy_pix))
        ix1 = min(ix0 + 1, map_size - 1)
        iy1 = min(iy0 + 1, map_size - 1)
        fx = fx_pix - np.floor(fx_pix)
        fy = fy_pix - np.floor(fy_pix)
        return ((1 - fx) * (1 - fy) * field[iy0, ix0]
                + fx * (1 - fy) * field[iy0, ix1]
                + (1 - fx) * fy * field[iy1, ix0]
                + fx * fy * field[iy1, ix1])

    d_pix = _bilin(dist_field, cpx, cpy)
    d_m = d_pix * grid_res

    eps = 1.0
    g_x = (_bilin(dist_field, min(cpx + eps, map_size - 1), cpy)
           - _bilin(dist_field, max(cpx - eps, 0), cpy)) / (2 * eps)
    g_y = (_bilin(dist_field, cpx, min(cpy + eps, map_size - 1))
           - _bilin(dist_field, cpx, max(cpy - eps, 0))) / (2 * eps)
    gnorm = np.sqrt(g_x * g_x + g_y * g_y) + 1e-6
    dir_x = -g_x / gnorm
    dir_y = -g_y / gnorm

    rel_x = dir_x * d_m
    rel_y = dir_y * d_m
    gate = 1.0 if d_m <= sensing_radius else 0.0
    return np.array([rel_x * gate, rel_y * gate], dtype=np.float32)

# =============================================================================
# Observation construction (simple_states 8-D → normalise → +belief → 13-D)
# =============================================================================
def build_obs_simple_states(x, y, vx, vy):
    """Build 8-D normalised observation for simple_states mode."""
    gx, gy = GOAL_XY
    rel = nearest_obstacle_rel(
        x, y,
        _MAP["dist_field"], _MAP["map_size"], _MAP["grid_res"],
        SENSING_RADIUS,
    )
    W = _WORLD_SIZE
    R = SENSING_RADIUS
    obs_raw = np.array([x, y, vx, vy, gx, gy, rel[0], rel[1]], dtype=np.float32)
    variance = np.array([W, W, 1.0, 1.0, W, W, R, R], dtype=np.float32)
    obs_norm = obs_raw / variance          # mean is zero
    return obs_norm

# =============================================================================
# JAX policy forward pass (Residual_Policy_FromPretrained)
# =============================================================================
def _load_checkpoint(path):
    with open(path, "rb") as f:
        ck = pickle.load(f)
    print(f"[model] Loaded checkpoint: best_iteration={ck.get('best_iteration')}, "
          f"success_rate={ck.get('best_eval_success_rate', 'N/A')}")
    return ck

def _layer_keys_in_order(params):
    """Return Dense-like layer keys ('Dense_X' or 'layers_X') sorted by suffix.
    Skips non-dict entries like 'log_std'."""
    return sorted(
        [k for k, v in params.items()
         if isinstance(v, dict) and "kernel" in v],
        key=lambda k: int(k.rsplit("_", 1)[1]),
    )


def _dense_forward(params, x, activation="relu"):
    """Forward through Dense layers in order. Handles both 'Dense_X' and
    'layers_X' (Sequential-style with skipped activation indices)."""
    keys = _layer_keys_in_order(params)
    for i, key in enumerate(keys):
        w = jnp.array(params[key]["kernel"])
        b = jnp.array(params[key]["bias"])
        x = x @ w + b
        if i < len(keys) - 1:
            if activation == "relu":
                x = jax.nn.relu(x)
            elif activation == "tanh":
                x = jnp.tanh(x)
    return x


def _base_input(obs_8d_norm, belief_5d, policy_params):
    """Build the base_net input by inspecting its first kernel shape.
        8-D first kernel  → obs only         (M0)
        13-D first kernel → obs + slot       (slot = belief if concat else zeros)
    """
    base_params = policy_params.get("base_net", policy_params)
    first_key = _layer_keys_in_order(base_params)[0]
    in_dim = base_params[first_key]["kernel"].shape[0]
    if in_dim == 8:
        return obs_8d_norm
    if BELIEF_MODE == "concat":
        return jnp.concatenate([obs_8d_norm, belief_5d])
    return jnp.concatenate([obs_8d_norm, jnp.zeros(5)])


def policy_forward(policy_params, obs_8d_norm, belief_5d):
    """Deterministic action. Branches on params structure to support:
        - flat policy (M1, M5):  policy_params == top-level Dense_*
        - residual policy (M0, M2): base_net + res_mean_net + gate_net
    Returns (action_mean, base_mean, residual_mean*gate).
    """
    has_residual = ("res_mean_net" in policy_params
                    and "base_net" in policy_params)

    if not has_residual:
        # Flat policy_network (M1 / M5). For M5 the belief is concatenated;
        # for M1 (BELIEF_MODE='none') the slot is zeros — matches training.
        x_input = _base_input(obs_8d_norm, belief_5d, policy_params)
        action_mean = _dense_forward(policy_params, x_input, activation="relu")
        zero = jnp.zeros_like(action_mean)
        return action_mean, action_mean, zero

    # Residual policy (M0 / M2). Base ignores belief; residual head consumes
    # the scaled belief mean; gate consumes scaled belief uncertainty.
    base_params = policy_params["base_net"]
    x_base = _base_input(obs_8d_norm, belief_5d, policy_params)
    base_mean = _dense_forward(base_params, x_base, activation="relu")

    belief_mean = belief_5d[:2] * MU_SCALE
    belief_unc = jnp.concatenate([
        belief_5d[2:4] * SIGMA_SCALE,
        belief_5d[4:5],
    ])

    pos_xy = obs_8d_norm[:2]
    vel_xy = obs_8d_norm[2:4]
    goal_xy = obs_8d_norm[4:6]
    goal_delta = goal_xy - pos_xy
    res_input = jnp.concatenate([
        jax.lax.stop_gradient(base_mean),
        pos_xy, vel_xy, goal_delta,
        belief_mean,
    ])
    residual_mean = _dense_forward(
        policy_params["res_mean_net"], res_input, activation="relu"
    )
    gate = _dense_forward(
        policy_params["gate_net"], belief_unc, activation="relu"
    )

    action_mean = base_mean + residual_mean * gate
    return action_mean, base_mean, residual_mean * gate


# JIT-compile the forward pass for speed
_policy_forward_jit = jax.jit(policy_forward, static_argnums=())

# =============================================================================
# Hardware constants
# =============================================================================
URIS = [
    'radio://0/80/2M/E7E7E7E710',
]
DEFAULT_HEIGHT = 0.5       # m — hover altitude for real CF
HOVER_Z_SIM    = 1.0        # m — hover altitude used during training
RL_DECISION_INTERVAL = 0.5  # s — matches training
MPC_PLANNING_INTERVAL = _DT_LQR  # 0.025 s

FLIGHT_DURATION = 40.0      # s — total flight time

logging.basicConfig(level=logging.ERROR)

# =============================================================================
# Global state containers
# =============================================================================
fullstate: Dict[str, Dict[str, float]] = {}
crazyflies: List[SyncCrazyflie] = []
start_barrier = Barrier(len(URIS))


LOG_DIR = os.path.join(os.getcwd(), "cf_logs", time.strftime("%Y%m%d_%H%M%S"))
os.makedirs(LOG_DIR, exist_ok=True)

_measurements: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
_rl_decisions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
_meas_lock = threading.Lock()
_rl_lock = threading.Lock()

def _sanitize_uri(uri: str) -> str:
    return uri.replace("://", "_").replace("/", "_").replace(":", "_")

# =============================================================================
# Logging helpers
# =============================================================================
def record_rl_decision(uri, rl_t, ref_mpc, belief_5d, current_xy):
    """Snapshot one RL tick: full reference horizon (positions only) + belief
    + the drone position when the decision was issued. Used for overlaying
    the planned references on the trajectory plot afterwards."""
    rec = {
        "rl_t": float(rl_t),
        "ref_x": np.asarray(ref_mpc[0, :], dtype=np.float64).copy(),
        "ref_y": np.asarray(ref_mpc[1, :], dtype=np.float64).copy(),
        "belief": np.asarray(belief_5d, dtype=np.float64).copy(),
        "actual_x": float(current_xy[0]),
        "actual_y": float(current_xy[1]),
    }
    with _rl_lock:
        _rl_decisions[uri].append(rec)


def record_sample(uri, state_vec, thrust_pwm, extra=None):
    row = {
        "t": time.time(),
        "x": float(state_vec[0]), "y": float(state_vec[1]), "z": float(state_vec[2]),
        "qx": float(state_vec[3]), "qy": float(state_vec[4]),
        "qz": float(state_vec[5]), "qw": float(state_vec[6]),
        "vx": float(state_vec[7]), "vy": float(state_vec[8]), "vz": float(state_vec[9]),
        "gyro_x": float(state_vec[10]), "gyro_y": float(state_vec[11]),
        "gyro_z": float(state_vec[12]),
        "thrust_pwm": float(thrust_pwm),
    }
    if extra:
        row.update(extra)
    with _meas_lock:
        _measurements[uri].append(row)


def save_measurements_to_disk():
    if not _measurements:
        print("[logger] No measurements collected; nothing to save.")
        return
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    for uri, rows in _measurements.items():
        if not rows:
            continue
        csv_path = os.path.join(LOG_DIR, f"{current_datetime}_{_sanitize_uri(uri)}_meas.csv")
        fieldnames = list(rows[0].keys())
        t0 = rows[0]["t"]
        for r in rows:
            r["t_rel"] = r["t"] - t0
        if "t_rel" not in fieldnames:
            fieldnames.append("t_rel")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"[logger] Saved {len(rows)} samples to {csv_path}")

    # Save per-RL-decision references + beliefs to CSV (one row per RL tick,
    # ref_x/ref_y stored as semicolon-joined strings for CSV friendliness).
    for uri, recs in _rl_decisions.items():
        if not recs:
            continue
        rl_csv = os.path.join(LOG_DIR, f"{current_datetime}_{_sanitize_uri(uri)}_rl.csv")
        with open(rl_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rl_t", "actual_x", "actual_y",
                        "belief_mu_x", "belief_mu_y",
                        "belief_sigma_x", "belief_sigma_y", "belief_p_fast",
                        "ref_x", "ref_y"])
            for r in recs:
                w.writerow([
                    r["rl_t"], r["actual_x"], r["actual_y"],
                    r["belief"][0], r["belief"][1],
                    r["belief"][2], r["belief"][3], r["belief"][4],
                    ";".join(f"{v:.4f}" for v in r["ref_x"]),
                    ";".join(f"{v:.4f}" for v in r["ref_y"]),
                ])
        print(f"[logger] Saved {len(recs)} RL decisions to {rl_csv}")

    # Trajectory plot — 1×4 layout: trajectory (with ref overlays), distance,
    # actions, belief over RL time.
    for uri, rows in _measurements.items():
        if not rows:
            continue
        # Snapshot — under Ctrl-C the test() thread can still be appending to
        # _measurements[uri]; without the local copy, list comprehensions
        # later in this block see an array length 1 longer than `flight_times`.
        with _meas_lock:
            rows = list(rows)
        xs = np.array([r["x"] for r in rows])
        ys = np.array([r["y"] for r in rows])
        flight_times = np.array([r.get("flight_time", i * 0.025) for i, r in enumerate(rows)])
        actions_x = np.array([r.get("action_x", 0.0) for r in rows])
        actions_y = np.array([r.get("action_y", 0.0) for r in rows])
        goal_x, goal_y = GOAL_XY
        rl_recs = _rl_decisions.get(uri, [])

        fig, axes = plt.subplots(1, 5, figsize=(30, 6))

        # Subplot 1: XY trajectory + per-RL-decision reference horizons.
        ax = axes[0]
        m = _MAP
        h, w = m["safe"].shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[m["safe"] > 0.5] = [0.0, 0.8, 0.0, 0.25]
        rgba[m["avoid"] > 0.5] = [0.8, 0.0, 0.0, 0.35]
        rgba[m["target"] > 0.5] = [0.0, 0.0, 0.8, 0.35]
        ax.imshow(rgba, extent=[0, _WORLD_SIZE, 0, _WORLD_SIZE],
                  origin='lower', aspect='equal', zorder=0)

        # Reference horizons (one per RL tick) — thin red lines + dots.
        for i, r in enumerate(rl_recs):
            lab = 'Reference (planned)' if i == 0 else None
            ax.plot(r["ref_x"], r["ref_y"], color='red', linewidth=1.0,
                    alpha=0.55, zorder=4, label=lab)
            ax.scatter(r["ref_x"], r["ref_y"], c='red', s=8, alpha=0.5, zorder=5)

        # Real path (40 Hz LQR sampling).
        ax.plot(xs, ys, 'b-', linewidth=1.8, alpha=0.85, label='Real path', zorder=6)
        ax.scatter(xs, ys, c='blue', s=4, alpha=0.6, zorder=6)

        # Pair markers — each reference's terminal point (red square) with
        # where the drone actually was 0.5 s later (blue square). A grey line
        # connects the pair so the prediction gap is visible at a glance.
        for i, r in enumerate(rl_recs):
            ref_end_x = float(r["ref_x"][-1])
            ref_end_y = float(r["ref_y"][-1])
            if i + 1 < len(rl_recs):
                real_x = float(rl_recs[i + 1]["actual_x"])
                real_y = float(rl_recs[i + 1]["actual_y"])
            else:
                real_x, real_y = float(xs[-1]), float(ys[-1])

            lab_ref = 'Ref end (planned)' if i == 0 else None
            lab_real = 'Real @ ref end (t+0.5s)' if i == 0 else None
            ax.plot([ref_end_x, real_x], [ref_end_y, real_y],
                    color='gray', linewidth=0.6, alpha=0.55, zorder=10)
            ax.scatter(ref_end_x, ref_end_y, marker='s', s=55,
                       facecolors='red', edgecolors='darkred', linewidths=1.0,
                       alpha=0.95, zorder=11, label=lab_ref)
            ax.scatter(real_x, real_y, marker='s', s=55,
                       facecolors='blue', edgecolors='navy', linewidths=1.0,
                       alpha=0.95, zorder=11, label=lab_real)
            # Tiny step number annotation next to the ref-end square so you
            # can map a square back to the time it was planned.
            ax.text(ref_end_x + 0.02, ref_end_y + 0.02, f"{i}",
                    fontsize=6, color='darkred', alpha=0.85, zorder=12)

        # Belief mean as magenta arrows at each RL decision.
        if rl_recs:
            ax_xs = np.array([r["actual_x"] for r in rl_recs])
            ax_ys = np.array([r["actual_y"] for r in rl_recs])
            mu_x = np.array([r["belief"][0] for r in rl_recs])
            mu_y = np.array([r["belief"][1] for r in rl_recs])
            mu_mag = float(np.max(np.hypot(mu_x, mu_y)) or 1e-8)
            scale = (0.15 / mu_mag) if mu_mag < 0.05 else 8.0
            ax.quiver(ax_xs, ax_ys, scale * mu_x, scale * mu_y,
                      angles='xy', scale_units='xy', scale=1.0,
                      color='magenta', width=0.004, alpha=0.85, zorder=7)
            ax.plot([], [], color='magenta', label='Belief μ')

        ax.plot(xs[0], ys[0], 'go', markersize=10, label='Start', zorder=8)
        ax.plot(goal_x, goal_y, 'r*', markersize=15, label='Goal', zorder=8)
        ax.set_xlim(-0.05, _WORLD_SIZE + 0.05)
        ax.set_ylim(-0.05, _WORLD_SIZE + 0.05)
        ax.set_aspect('equal')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_title(f'Trajectory  ({MODEL_TAG})')

        # Subplot 2: distance to goal
        ax2 = axes[1]
        dist = np.sqrt((xs - goal_x) ** 2 + (ys - goal_y) ** 2)
        ax2.plot(flight_times, dist, 'b-')
        ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Distance (m)')
        ax2.set_title('Distance to Goal')

        # Subplot 3: actions
        ax3 = axes[2]
        ax3.plot(flight_times, actions_x, label='ax')
        ax3.plot(flight_times, actions_y, label='ay')
        ax3.legend(); ax3.set_xlabel('Time (s)'); ax3.set_title('Actions')

        # Subplot 4: belief features over RL time (mu on left, sigma/p_fast on right).
        ax4 = axes[3]
        if rl_recs:
            rl_t = np.array([r["rl_t"] for r in rl_recs])
            bel = np.stack([r["belief"] for r in rl_recs], axis=0)  # (N, 5)
            l1, = ax4.plot(rl_t, bel[:, 0], 'r-', linewidth=1.4, label='μ_x')
            l2, = ax4.plot(rl_t, bel[:, 1], 'b-', linewidth=1.4, label='μ_y')
            ax4.set_ylabel('Belief μ')
            ax4.set_xlabel('RL time (s)')

            ax4r = ax4.twinx()
            l3, = ax4r.plot(rl_t, bel[:, 2], 'r--', linewidth=1.0, alpha=0.8, label='σ_x')
            l4, = ax4r.plot(rl_t, bel[:, 3], 'b--', linewidth=1.0, alpha=0.8, label='σ_y')
            l5, = ax4r.plot(rl_t, bel[:, 4], 'g-', linewidth=1.0, alpha=0.8, label='p_fast')
            ax4r.set_ylabel('σ / p_fast', color='gray')
            ax4r.tick_params(axis='y', labelcolor='gray')

            ax4.legend([l1, l2, l3, l4, l5],
                       [h.get_label() for h in [l1, l2, l3, l4, l5]],
                       fontsize=6, loc='upper right')
        else:
            ax4.text(0.5, 0.5, 'no belief data', ha='center', va='center',
                     transform=ax4.transAxes)
        ax4.set_title('Belief over RL time')

        # Subplot 5: commanded vs actual roll/pitch — diagnoses on-board PID
        # overshoot or lag. If actual (solid) overshoots commanded (dashed),
        # the inner attitude-loop gain is too high.
        ax5 = axes[4]
        cmd_roll = np.array([r.get("target_roll_deg", 0.0) for r in rows])
        cmd_pitch = np.array([r.get("target_pitch_deg", 0.0) for r in rows])
        act_roll = np.array([r.get("actual_roll_deg", 0.0) for r in rows])
        act_pitch = np.array([r.get("actual_pitch_deg", 0.0) for r in rows])
        ax5.plot(flight_times, cmd_roll, 'r--', linewidth=1.0, alpha=0.7, label='cmd roll')
        ax5.plot(flight_times, act_roll, 'r-', linewidth=1.4, alpha=0.9, label='actual roll')
        ax5.plot(flight_times, cmd_pitch, 'b--', linewidth=1.0, alpha=0.7, label='cmd pitch')
        ax5.plot(flight_times, act_pitch, 'b-', linewidth=1.4, alpha=0.9, label='actual pitch')
        ax5.axhline(25, color='gray', linestyle=':', linewidth=0.5)
        ax5.axhline(-25, color='gray', linestyle=':', linewidth=0.5)
        ax5.set_xlabel('Time (s)'); ax5.set_ylabel('deg')
        ax5.set_title('Roll/Pitch: commanded vs actual')
        ax5.legend(fontsize=6, loc='upper right')

        plt.tight_layout()
        plot_path = os.path.join(LOG_DIR, f"{current_datetime}_{_sanitize_uri(uri)}_traj.png")
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[logger] Saved plot to {plot_path}")

# =============================================================================
# Crazyflie logging & connection
# =============================================================================
def make_logconfs(uri):
    logconfs = []
    lc1 = LogConfig(name='PosVel', period_in_ms=25)
    for v in ('stateEstimateZ.x', 'stateEstimateZ.y', 'stateEstimateZ.z',
              'stateEstimateZ.vx', 'stateEstimateZ.vy', 'stateEstimateZ.vz'):
        lc1.add_variable(v, 'float')
    lc1.data_received_cb.add_callback(_log_cb_factory(uri))
    logconfs.append(lc1)

    # Attitude at 20 Hz — only used for the LQR's roll/pitch state, which
    # the on-board cascade tracks far faster than we can read it back.
    lc2 = LogConfig(name='Attitude', period_in_ms=50)
    for v in ('stabilizer.roll', 'stabilizer.pitch', 'stabilizer.yaw'):
        lc2.add_variable(v, 'float')
    lc2.add_variable('stabilizer.thrust', 'float')
    lc2.data_received_cb.add_callback(_log_cb_factory(uri))
    logconfs.append(lc2)

    # Quaternion stream removed: state_13_to_state_9 only uses Euler from
    # the stabilizer, and the quaternion was only retained for CSV logging.

    return logconfs

def _log_cb_factory(uri):
    def _cb(ts, data, logconf):
        d = fullstate.get(uri)
        if d is None:
            fullstate[uri] = dict(data)
        else:
            d.update(data)
    return _cb


def connect(uri):
    scf = SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache'))
    scf.open_link()
    time.sleep(0.2)

    scf.cf.param.set_value("stabilizer.estimator", 2)   # Kalman
    scf.cf.param.set_value("commander.enHighLevel", 1)   # high-level for takeoff
    # Attitude mode: roll/pitch commands are ANGLES (not rates)
    scf.cf.param.set_value("flightmode.stabModeRoll", 1)
    scf.cf.param.set_value("flightmode.stabModePitch", 1)
    scf.cf.param.set_value("flightmode.stabModeYaw", 0)  # yaw stays rate
    scf.cf.param.set_value("stabilizer.controller", 1)    # Mellinger

    logconfs = make_logconfs(uri)
    scf._logconfs = logconfs
    for lc in logconfs:
        try:
            scf.cf.log.add_config(lc)
            lc.start()
        except AttributeError as e:
            sys.exit(f"[{uri}] Failed to start log '{lc.name}': {e}")
    crazyflies.append(scf)


def get_state_SI(uri):
    """13-D state: [x,y,z, qx,qy,qz,qw, vx,vy,vz, roll_rad,pitch_rad,yaw_rad]
    plus scalar thrust."""
    s = fullstate.get(uri, {})
    x  = float(s.get('stateEstimateZ.x', 0)) * 1e-3
    y  = float(s.get('stateEstimateZ.y', 3)) * 1e-3
    z  = float(s.get('stateEstimateZ.z', 0)) * 1e-3
    vx = float(s.get('stateEstimateZ.vx', 0)) * 1e-3
    vy = float(s.get('stateEstimateZ.vy', 0)) * 1e-3
    vz = float(s.get('stateEstimateZ.vz', 0)) * 1e-3
    qx = float(s.get('stateEstimate.qx', 0))
    qy = float(s.get('stateEstimate.qy', 0))
    qz = float(s.get('stateEstimate.qz', 0))
    qw = float(s.get('stateEstimate.qw', 1))
    thrust = float(s.get('stabilizer.thrust', 0))
    # Euler from stabilizer (degrees → radians)
    roll_deg  = float(s.get('stabilizer.roll', 0))
    pitch_deg = float(s.get('stabilizer.pitch', 0))
    yaw_deg   = float(s.get('stabilizer.yaw', 0))
    roll  = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw   = math.radians(yaw_deg)

    return np.array([x, y, z, qx, qy, qz, qw, vx, vy, vz,
                     roll, pitch, yaw], dtype=np.float32), thrust


def state_13_to_state_9(s13):
    """Convert 13-D CF state to 9-D LQR state [x,y,z, vx,vy,vz, roll,pitch,yaw].

    Pitch sign is NEGATED here: the Crazyflie firmware reports `stabilizer.pitch`
    in aerospace convention (positive = nose up → drone moves -x_body), while the
    Phoenix LQR was built in the opposite convention (positive pitch = nose down
    → +x_body). Negating once on input and once on output keeps the LQR's
    internal model consistent with what the hardware actually does (verified
    by debug_axis.py: cmd_pitch=+5° produced Δx_body=-0.30 m).
    Roll convention matches between CF and Phoenix, so it's left alone.
    """
    return np.array([
        s13[0], s13[1], s13[2],     # pos
        s13[7], s13[8], s13[9],     # vel
        s13[10], -s13[11], s13[12]  # roll, -pitch, yaw  (pitch sign-flip)
    ], dtype=np.float64)

# =============================================================================
# Thrust conversion
# =============================================================================
def thrust_to_cmd(T_total):
    """Total thrust [N] → CF thrust command (0..65535)."""
    a2 = 2.130295e-11; a1 = 1.032633e-6; a0 = 5.484560e-4
    f_per = max(0.0, T_total / 4.0)
    disc = (a1 / (2 * a2))**2 - (a0 - f_per) / a2
    cmd = 0.0 if disc < 0 else -a1 / (2 * a2) + math.sqrt(disc)
    return int(min(65535, max(0, cmd)))

# =============================================================================
# Smooth hover / land helpers
# =============================================================================
def smooth_send_hover(cf, start, target, duration):
    steps = int(duration / 0.025)
    for i in range(steps):
        f = (i + 1) / steps
        vx = start[0] + (target[0] - start[0]) * f
        vy = start[1] + (target[1] - start[1]) * f
        yr = start[2] + (target[2] - start[2]) * f
        z  = start[3] + (target[3] - start[3]) * f
        cf.commander.send_hover_setpoint(vx, vy, yr, z)
        time.sleep(0.025)

def smooth_land(cf, height, duration=3.0):
    steps = int(duration / 0.025)
    for i in range(steps):
        z = height * (1.0 - (i + 1) / steps)
        cf.commander.send_hover_setpoint(0, 0, 0, max(z, 0.02))
        time.sleep(0.025)
    cf.commander.send_stop_setpoint()
    time.sleep(0.1)

# =============================================================================
# Position status check (simple OOB / obstacle / goal check)
# =============================================================================
def check_position_status(x, y):
    m = _MAP
    ws = m["world_size"]
    if x < 0 or x > ws or y < 0 or y > ws:
        return {"terminate": True, "status": "oob", "message": "Out of bounds"}
    ms = m["map_size"]; gr = m["grid_res"]
    px = int(np.clip(np.floor(x / gr), 0, ms - 1))
    py = int(np.clip(np.floor(y / gr), 0, ms - 1))
    if m["avoid"][py, px] > 0.5:
        return {"terminate": True, "status": "collision", "message": "Hit obstacle"}
    if m["target"][py, px] > 0.5:
        return {"terminate": True, "status": "reached", "message": "Reached target"}
    return {"terminate": False, "status": "safe", "message": ""}


# =============================================================================
# Main flight loop
# =============================================================================
def test(cf, uri):
    """Main test function — one RL+LQR flight."""
    # ── Load model ──
    ck = _load_checkpoint(CHECKPOINT_PATH)
    policy_params = ck["policy"]["params"]

    # ── IMM belief filter ──
    imm_cfg = default_imm_config()
    imm_state = init_imm_state()
    belief_features = np.zeros(BELIEF_DIM, dtype=np.float32)

    # ── Pre-warm ALL JAX-jitted functions before takeoff ──
    # Without this, the first call to each jit'd function compiles for
    # ~0.5–1 s, blocking the LQR loop and dropping the drone. We hit this
    # with policy_forward_jit on the first RL tick AND with imm_predict /
    # imm_update on the second RL tick (verified by 0.79 s LQR blackout
    # in 20260504_230828; drone fell from z=0.50 to z=0.01 during the
    # gap). Compile everything here so the in-flight loop is steady-state.
    print("[warmup] pre-compiling JAX functions (≈1 s)…")
    _t0_warm = time.time()
    _dummy_obs = jnp.zeros(8, dtype=jnp.float32)
    _dummy_belief = jnp.zeros(BELIEF_DIM, dtype=jnp.float32)
    _ = _policy_forward_jit(policy_params, _dummy_obs, _dummy_belief)
    _warm_state = imm_predict(imm_state, imm_cfg)
    _warm_state = imm_update(_warm_state, jnp.zeros(2, dtype=jnp.float32), imm_cfg)
    _warm_feat = imm_get_features(_warm_state)
    # Block until all JAX dispatches finish (compile is async on the device).
    jax.block_until_ready(_warm_state.mu_models)
    jax.block_until_ready(_warm_state.model_probs)
    jax.block_until_ready(_warm_feat)
    print(f"[warmup] done in {time.time() - _t0_warm:.2f} s")

    # ── Take off with high-level commander ──
    cf.commander.send_setpoint(0, 0, 0, 0)  # unlock
    time.sleep(0.1)
    smooth_send_hover(cf, (0, 0, 0, 0), (0, 0, 0, DEFAULT_HEIGHT), 3.0)

    # ── Compute origin offset (shift real position → training frame) ──
    initial_state, _ = get_state_SI(uri)
    origin_offset = START_XY - initial_state[0:2].astype(np.float64)
    print(f"[origin] Real XY={initial_state[0:2]}, offset={origin_offset}, "
          f"→ training XY={START_XY}")

    # ── Switch to low-level commander for attitude control ──
    cf.param.set_value("commander.enHighLevel", 0)
    time.sleep(0.05)

    # ── State tracking ──
    current_action = np.zeros(2, dtype=np.float32)
    base_action_np = np.zeros(2, dtype=np.float32)
    residual_np = np.zeros(2, dtype=np.float32)
    K_seq = F_seq = s_seq = None
    reference_mpc = None
    lqr_step_idx = 0
    residual_xy_acc = np.zeros(2, dtype=np.float64)
    rl_decision_count = 0    # used to skip the JIT-poisoned first RL window
    termination_cause = "timeout"

    start_time = time.time()
    last_action_time = -RL_DECISION_INTERVAL  # trigger immediate first action
    step = 0
    rate_anchor_step = 0
    rate_anchor_time = start_time

    while time.time() - start_time < FLIGHT_DURATION:
        loop_start = time.time()
        current_time = time.time() - start_time

        # ── Read CF state ──
        state_13, thrust = get_state_SI(uri)
        state_13[0:2] += origin_offset.astype(np.float32)  # shift to training frame
        state_9 = state_13_to_state_9(state_13)
        # Map real z (around DEFAULT_HEIGHT) into the training frame
        # (around HOVER_Z_SIM). When the drone is at the intended hover
        # altitude, state_9[2] == HOVER_Z_SIM and z error is zero; if it
        # drifts down, the LQR sees the position error and pushes thrust.
        state_9[2] = float(state_13[2]) + (HOVER_Z_SIM - DEFAULT_HEIGHT)

        # ── Terminal check ──
        pos_status = check_position_status(float(state_9[0]), float(state_9[1]))
        if pos_status["terminate"]:
            termination_cause = pos_status["message"]
            print(f"[terminal] {pos_status['status']}: {pos_status['message']} "
                  f"at ({state_9[0]:.3f}, {state_9[1]:.3f})")
            break

        # ── RL decision (every 0.5 s) ──
        if current_time - last_action_time >= RL_DECISION_INTERVAL - 0.02:
            # SKIP belief update for the FIRST RL window. The first call to
            # _policy_forward_jit triggers JAX JIT compile (~0.5–1 s wall
            # clock), during which the LQR loop sleeps for 0 ms but real
            # time still elapses. The first window's residual_xy_acc thus
            # compares ~800 ms of real motion against ~25 ms of predicted
            # motion → garbage residual that swings the IMM into a phantom
            # wind direction (verified in 20260504_230324: μ_x jumped to
            # -0.061 on the first update with no actual disturbance).
            if rl_decision_count > 0 and np.all(np.isfinite(residual_xy_acc)):
                imm_state = imm_predict(imm_state, imm_cfg)
                imm_state = imm_update(imm_state, jnp.array(residual_xy_acc*RESIDUAL_SCALE), imm_cfg)
                belief_features = np.asarray(imm_get_features(imm_state), dtype=np.float32)
            residual_xy_acc[:] = 0.0
            rl_decision_count += 1

            # Build observation
            obs_8d = build_obs_simple_states(
                float(state_9[0]), float(state_9[1]),
                float(state_9[3]), float(state_9[4]),
            )

            # Policy forward pass
            action_jax, base_jax, res_jax = _policy_forward_jit(
                policy_params,
                jnp.array(obs_8d),
                jnp.array(belief_features),
            )
            current_action = np.asarray(action_jax, dtype=np.float32)
            base_action_np = np.asarray(base_jax, dtype=np.float32)
            residual_np = np.asarray(res_jax, dtype=np.float32)

            # Cap policy outputs. Lowered from 5 → 3 with the tilt clip
            # tightened to ±15°: caps commanded velocity at 0.3 m/s, which
            # is achievable in one RL window without saturating attitude.
            ACTION_CLIP = 15.0
            current_action = np.clip(current_action, -ACTION_CLIP, ACTION_CLIP)

            print(f"[RL t={current_time:.1f}s] action={current_action} "
                  f"(base={base_action_np}, res={residual_np})")
            last_action_time = current_time

            # ── Build reference trajectory from velocity command ──
            # Ramp the reference velocity from the current measured velocity
            # to the policy's target over the first half of the LQR horizon
            # (~0.25 s), then hold the target for the remaining half. Gives
            # the LQR a settled reference at the horizon tail.
            cur_vel_xy = state_9[3:5].astype(np.float64).copy()
            target_vel_xy = current_action.astype(np.float64) * ACTION_SCALE
            RAMP_STEPS = _HORIZON // 4

            ref = np.zeros((9, _HORIZON + 1), dtype=np.float64)
            ref[:, 0] = state_9.copy()
            for i in range(_HORIZON):
                # alpha = min(1.0, (i + 1) / RAMP_STEPS)
                alpha =1.0
                ramped_vx = (1.0 - alpha) * cur_vel_xy[0] + alpha * target_vel_xy[0]
                ramped_vy = (1.0 - alpha) * cur_vel_xy[1] + alpha * target_vel_xy[1]
                ref[0, i + 1] = ref[0, i] + ramped_vx * _DT_LQR
                ref[1, i + 1] = ref[1, i] + ramped_vy * _DT_LQR
                ref[2, i + 1] = HOVER_Z_SIM
                ref[3, i + 1] = ramped_vx
                ref[4, i + 1] = ramped_vy
                ref[5, i + 1] = 0.0
                ref[6:9, i + 1] = 0.0
            reference_mpc = ref

            # Snapshot this RL decision for post-flight plotting.
            record_rl_decision(
                uri, current_time, reference_mpc, belief_features,
                (state_9[0], state_9[1]),
            )

            # Compute LQR gains for whole horizon
            K_seq, F_seq, s_seq = compute_lqt_gains(
                A_dt, B_dt, Q_lqr, R_lqr, reference_mpc
            )
            lqr_step_idx = 0

        # ── LQR tracking step ──
        if K_seq is None:
            # No action yet — send zero attitude, hover thrust
            hover_cmd = thrust_to_cmd(_MASS * _GRAVITY)
            cf.commander.send_setpoint(0, 0, 0, hover_cmd)
            time.sleep(MPC_PLANNING_INTERVAL)
            step += 1
            continue

        k = min(lqr_step_idx, _HORIZON - 1)
        u_hat = lqt_control_step(K_seq[k], F_seq[k], s_seq[k + 1], state_9)

        # Predicted next state (for belief residual). Uses the nonlinear
        # quadrotor dynamics rather than the LQR's linear A_dt/B_dt — the
        # 3% trig error and 10% cos(tilt) thrust loss in the linear model
        # were being mistaken for wind by the IMM filter. The LQR controller
        # itself still uses linear A_dt/B_dt for gain scheduling.
        x_pred_next = nonlinear_step(state_9, u_hat)

        # ── Convert u_hat to CF commands ──
        # u_hat[0] = thrust fraction, u_hat[1:4] = target roll/pitch/yaw (rad)
        # Tilt-aware hover: at tilt θ the vertical thrust is T·cos(θ), so
        # commanding mg yields mg·cos(θ) of lift — losing ~10% at ±25°.
        # Compensate so steady-state hover holds altitude even when tilted.
        tilt_rad = math.sqrt(state_9[6] ** 2 + state_9[7] ** 2)
        cos_tilt = max(math.cos(tilt_rad), 0.5)        # floor for safety
        hover_thrust = (_MASS * _GRAVITY) / cos_tilt

        # u_hat[0] clip [-0.3, +1.0] — same range used by the nonlinear
        # predictor. Floor of -0.3 keeps T_total > 0 (prevents predictor
        # ax/ay sign flip BRACE warned about) while still giving the LQR
        # enough descent authority (~6.6 m/s² max downward accel). This
        # is INTENTIONALLY looser than BRACE's [-0.023, +0.373]: that
        # bound mirrors Phoenix-sim's attitude_cascade PWM clamp at
        # 35000–55000, but real CF via send_setpoint accepts PWM 0..65535
        # without on-board clamping (verified by past CSV: thrust_pwm
        # range was [0, 65535]).
        u_thrust = float(np.clip(u_hat[0], -0.3, 1.0))
        thrust_force = u_thrust * _T_MAX
        thrust_cmd = thrust_to_cmd(hover_thrust + thrust_force)
        thrust_cmd = int(np.clip(thrust_cmd, 0, 65535))

        # Target angles from LQR (radians → degrees for CF).
        # Clip to safe range (±25 deg).
        # NO sign flip on output: the send_setpoint API uses the same convention
        # as Phoenix (+pitch → +x motion). State_9[7] IS sign-flipped on input
        # (see state_13_to_state_9) because stabilizer.pitch logs in the
        # opposite convention to send_setpoint. Verified by inspecting CSV:
        # target_pitch_deg=+25 produced vx → +2.4 m/s while stabilizer.pitch
        # logged -25 — i.e. send_setpoint and the log have OPPOSITE signs.
        # Tilt clip lowered 25→15°: keeps the LQR inside its hover-linearization
        # validity range, cuts the cos(tilt) thrust loss to ~3% (vs 10% at 25°),
        # and reduces the model-vs-real residuals that the IMM filter mistakes
        # for wind. Penalty: lateral acceleration capped at g·sin(15°)≈2.5 m/s².
        target_roll_deg  = float(np.clip(np.degrees(u_hat[1]), -15.0, 15.0))
        target_pitch_deg = float(np.clip(np.degrees(u_hat[2]), -15.0, 15.0))
        # Yaw is uncontrolled (policy doesn't use it); hold the rate at 0
        # and let the on-board stabilizer maintain heading.
        target_yawrate_deg = 0.0

        # send_setpoint: (roll_deg, pitch_deg, yawrate_deg_s, thrust_pwm)
        # With stabModeRoll=1, stabModePitch=1: roll/pitch are ANGLE commands
        cf.commander.send_setpoint(
            target_roll_deg, target_pitch_deg,
            target_yawrate_deg, thrust_cmd
        )

        # ── Record & advance ──
        record_sample(uri, state_13, thrust_cmd, {
            "step": step,
            "flight_time": float(current_time),
            "action_x": float(current_action[0]),
            "action_y": float(current_action[1]),
            "base_action_x": float(base_action_np[0]),
            "base_action_y": float(base_action_np[1]),
            "residual_action_x": float(residual_np[0]),
            "residual_action_y": float(residual_np[1]),
            "Belief_x": float(belief_features[0]),
            "Belief_y": float(belief_features[1]),
            "Belief_sigma_x": float(belief_features[2]),
            "Belief_sigma_y": float(belief_features[3]),
            "Belief_p_fast": float(belief_features[4]),
            "goal_x": float(GOAL_XY[0]),
            "goal_y": float(GOAL_XY[1]),
            "u_hat_thrust": float(u_hat[0]),
            "u_hat_roll": float(u_hat[1]),
            "u_hat_pitch": float(u_hat[2]),
            "u_hat_yaw": float(u_hat[3]),
            "target_roll_deg": target_roll_deg,
            "target_pitch_deg": target_pitch_deg,
            "actual_roll_deg": float(math.degrees(state_13[10])),
            # Pitch logged with sign FLIPPED relative to stabilizer.pitch so
            # this column matches the target_pitch_deg convention (send_setpoint
            # API / Phoenix). Without the flip, cmd and actual mirror each
            # other in the plot because the firmware reports stabilizer.pitch
            # in the opposite sign to what send_setpoint accepts.
            "actual_pitch_deg": -float(math.degrees(state_13[11])),
            "actual_yaw_deg": float(math.degrees(state_13[12])),
            "lqr_step": k,
            "trajectory_x_end": float(reference_mpc[0, -1]),
            "trajectory_y_end": float(reference_mpc[1, -1]),
        })

        # Maintain loop rate
        elapsed = time.time() - loop_start
        sleep_time = max(0, MPC_PLANNING_INTERVAL - elapsed)
        time.sleep(sleep_time)

        # Read state after control was applied; compute residual
        state_13_post, _ = get_state_SI(uri)
        state_13_post[0:2] += origin_offset.astype(np.float32)
        state_9_post = state_13_to_state_9(state_13_post)
        residual_xy_acc += (state_9_post[0:2] - x_pred_next[0:2]).astype(np.float64)

        lqr_step_idx += 1
        step += 1

        # Loop-rate sanity check — rolling window so the JAX JIT compile on
        # the first RL decision doesn't poison the average. Should reach
        # ~40 Hz at steady state to match LQR _DT_LQR=25 ms.
        if step > 0 and step % 50 == 0:
            now = time.time()
            d_steps = step - rate_anchor_step
            d_time = now - rate_anchor_time
            rate = d_steps / max(d_time, 1e-3)
            print(f"[loop] last {d_steps} steps: {rate:.1f} Hz")
            rate_anchor_step = step
            rate_anchor_time = now

    print(f"[flight] Ended: {termination_cause} after {step} steps")

    # ── Land ──
    cf.param.set_value("commander.enHighLevel", 1)
    time.sleep(0.05)
    state_now, _ = get_state_SI(uri)
    smooth_send_hover(cf, (0, 0, 0, state_now[2]), (0, 0, 0, DEFAULT_HEIGHT), 1.0)
    smooth_land(cf, DEFAULT_HEIGHT, duration=3.0)


def fly_all():
    threads = []
    for scf in crazyflies:
        t = Thread(target=test, args=(scf.cf, scf._link_uri))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    try:
        cflib.crtp.init_drivers()
        for uri in URIS:
            connect(uri)
        fly_all()
    except KeyboardInterrupt:
        print("\nInterrupted; saving logs...")
    finally:
        save_measurements_to_disk()
        for scf in crazyflies:
            try:
                scf.close_link()
            except Exception:
                pass
