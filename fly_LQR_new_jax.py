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
MODEL_DIR = (
    "/home/realm/jaeyoun/crazyflie-lib-python/models/"
    "trained_s0_M2_res_track_rew_frz_20260428_143340_"
    "PhoenixPhysicalJAX_DCBF_belief_residual_seed0"
)
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
BELIEF_DIM     = 5

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
    Q[3:6, 3:6] = 2 * np.eye(3)  # velocity
    Q[6:, 6:]   = np.eye(3)      # attitude
    R = 0.01 * np.eye(4)
    return A_dt, B_dt, Q, R

A_dt, B_dt, Q_lqr, R_lqr = _build_lqr_system()

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

def _dense_forward(params, x, activation="relu"):
    """Forward through a sequence of Dense_0, Dense_1, ... layers."""
    idx = 0
    while True:
        key = f"Dense_{idx}"
        if key not in params:
            break
        w = jnp.array(params[key]["kernel"])
        b = jnp.array(params[key]["bias"])
        x = x @ w + b
        # Apply activation to all but last layer
        next_key = f"Dense_{idx + 1}"
        if next_key in params and activation == "relu":
            x = jax.nn.relu(x)
        elif next_key in params and activation == "tanh":
            x = jnp.tanh(x)
        idx += 1
    return x


def policy_forward(policy_params, obs_8d_norm, belief_5d):
    """Deterministic action from the residual policy.

    Parameters
    ----------
    policy_params : dict  – checkpoint['policy']['params']
    obs_8d_norm   : (8,) normalised simple_states observation
    belief_5d     : (5,) [mu_x, mu_y, sigma_x, sigma_y, p_fast]

    Returns
    -------
    action_mean : (2,) deterministic action in [-1, 1]
    base_action : (2,)
    residual    : (2,) gated residual correction
    """
    # Base net expects 13-D input: [obs_8d, zeros_5d_belief_slot]
    # (trained without belief, so belief slot is always zero for the base)
    x_base = jnp.concatenate([obs_8d_norm, jnp.zeros(5)])
    base_mean = _dense_forward(policy_params["base_net"], x_base, activation="relu")

    # Belief processing
    belief_mean = belief_5d[:2] * MU_SCALE
    belief_unc = jnp.concatenate([
        belief_5d[2:4] * SIGMA_SCALE,
        belief_5d[4:5],
    ])

    # Residual input: state_geom mode
    # [stop_grad(base_mean), pos_xy, vel_xy, goal_delta, belief_mean]
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

    # Gate (activation="none" → linear, bias_init=1.0)
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
DEFAULT_HEIGHT = 0.35       # m — hover altitude for real CF
HOVER_Z_SIM    = 1.0        # m — hover altitude used during training
RL_DECISION_INTERVAL = 0.5  # s — matches training
MPC_PLANNING_INTERVAL = _DT_LQR  # 0.025 s

FLIGHT_DURATION = 30.0      # s — total flight time

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
_meas_lock = threading.Lock()

def _sanitize_uri(uri: str) -> str:
    return uri.replace("://", "_").replace("/", "_").replace(":", "_")

# =============================================================================
# Logging helpers
# =============================================================================
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

    # Trajectory plot
    for uri, rows in _measurements.items():
        if not rows:
            continue
        xs = np.array([r["x"] for r in rows])
        ys = np.array([r["y"] for r in rows])
        flight_times = np.array([r.get("flight_time", i * 0.025) for i, r in enumerate(rows)])
        actions_x = np.array([r.get("action_x", 0.0) for r in rows])
        actions_y = np.array([r.get("action_y", 0.0) for r in rows])
        goal_x, goal_y = GOAL_XY

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        # Subplot 1: XY trajectory
        ax = axes[0]
        m = _MAP
        h, w = m["safe"].shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[m["safe"] > 0.5] = [0.0, 0.8, 0.0, 0.25]
        rgba[m["avoid"] > 0.5] = [0.8, 0.0, 0.0, 0.35]
        rgba[m["target"] > 0.5] = [0.0, 0.0, 0.8, 0.35]
        ax.imshow(rgba, extent=[0, _WORLD_SIZE, 0, _WORLD_SIZE],
                  origin='lower', aspect='equal', zorder=0)
        ax.plot(xs, ys, 'b-', linewidth=2, alpha=0.7, label='Path')
        ax.plot(xs[0], ys[0], 'go', markersize=10, label='Start')
        ax.plot(goal_x, goal_y, 'r*', markersize=15, label='Goal')
        ax.set_xlim(-0.05, _WORLD_SIZE + 0.05)
        ax.set_ylim(-0.05, _WORLD_SIZE + 0.05)
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        ax.set_title('Trajectory')

        # Subplot 2: distance to goal
        ax2 = axes[1]
        dist = np.sqrt((xs - goal_x)**2 + (ys - goal_y)**2)
        ax2.plot(flight_times, dist, 'b-')
        ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Distance (m)')
        ax2.set_title('Distance to Goal')

        # Subplot 3: actions
        ax3 = axes[2]
        ax3.plot(flight_times, actions_x, label='ax')
        ax3.plot(flight_times, actions_y, label='ay')
        ax3.legend(); ax3.set_xlabel('Time (s)'); ax3.set_title('Actions')

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

    lc2 = LogConfig(name='Attitude', period_in_ms=25)
    for v in ('stabilizer.roll', 'stabilizer.pitch', 'stabilizer.yaw'):
        lc2.add_variable(v, 'float')
    lc2.add_variable('stabilizer.thrust', 'float')
    lc2.data_received_cb.add_callback(_log_cb_factory(uri))
    logconfs.append(lc2)

    lc3 = LogConfig(name='Quaternion', period_in_ms=25)
    for v in ('stateEstimate.qx', 'stateEstimate.qy',
              'stateEstimate.qz', 'stateEstimate.qw'):
        lc3.add_variable(v, 'float')
    lc3.data_received_cb.add_callback(_log_cb_factory(uri))
    logconfs.append(lc3)

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
    y  = float(s.get('stateEstimateZ.y', 0)) * 1e-3
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
    """Convert 13-D CF state to 9-D LQR state [x,y,z, vx,vy,vz, roll,pitch,yaw]."""
    return np.array([
        s13[0], s13[1], s13[2],   # pos
        s13[7], s13[8], s13[9],   # vel
        s13[10], s13[11], s13[12] # roll, pitch, yaw (already in radians)
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
    termination_cause = "timeout"

    start_time = time.time()
    last_action_time = -RL_DECISION_INTERVAL  # trigger immediate first action
    step = 0

    while time.time() - start_time < FLIGHT_DURATION:
        loop_start = time.time()
        current_time = time.time() - start_time

        # ── Read CF state ──
        state_13, thrust = get_state_SI(uri)
        state_13[0:2] += origin_offset.astype(np.float32)  # shift to training frame
        state_9 = state_13_to_state_9(state_13)
        state_9[2] = HOVER_Z_SIM  # z fixed to training hover altitude for LQR

        # ── Terminal check ──
        pos_status = check_position_status(float(state_9[0]), float(state_9[1]))
        if pos_status["terminate"]:
            termination_cause = pos_status["message"]
            print(f"[terminal] {pos_status['status']}: {pos_status['message']} "
                  f"at ({state_9[0]:.3f}, {state_9[1]:.3f})")
            break

        # ── RL decision (every 0.5 s) ──
        if current_time - last_action_time >= RL_DECISION_INTERVAL - 0.02:
            # Update belief with accumulated residual
            if np.all(np.isfinite(residual_xy_acc)):
                imm_state = imm_predict(imm_state, imm_cfg)
                imm_state = imm_update(imm_state, jnp.array(residual_xy_acc), imm_cfg)
                belief_features = np.asarray(imm_get_features(imm_state), dtype=np.float32)
            residual_xy_acc[:] = 0.0

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

            print(f"[RL t={current_time:.1f}s] action={current_action} "
                  f"(base={base_action_np}, res={residual_np})")
            last_action_time = current_time

            # ── Build reference trajectory from velocity command ──
            vel_cmd = np.array([current_action[0] * ACTION_SCALE,
                                current_action[1] * ACTION_SCALE,
                                0.0], dtype=np.float64)
            ref = np.zeros((9, _HORIZON + 1), dtype=np.float64)
            ref[:, 0] = state_9.copy()
            for i in range(_HORIZON):
                ref[0:3, i + 1] = ref[0:3, i] + vel_cmd * _DT_LQR
                ref[3:6, i + 1] = vel_cmd
                ref[6:9, i + 1] = 0.0     # desired attitude = 0
                ref[2, i + 1] = HOVER_Z_SIM
                ref[5, i + 1] = 0.0       # vz = 0
            reference_mpc = ref

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

        # Predicted next state (for belief residual)
        x_pred_next = A_dt @ state_9 + B_dt @ u_hat

        # ── Convert u_hat to CF commands ──
        # u_hat[0] = thrust fraction, u_hat[1:4] = target roll/pitch/yaw (rad)
        hover_thrust = _MASS * _GRAVITY
        thrust_force = u_hat[0] * _T_MAX
        thrust_cmd = thrust_to_cmd(hover_thrust + thrust_force)
        thrust_cmd = int(np.clip(thrust_cmd, 0, 65535))

        # Target angles from LQR (radians → degrees for CF)
        # Clip to safe range (±25 deg)
        target_roll_deg  = float(np.clip(np.degrees(u_hat[1]), -25.0, 25.0))
        target_pitch_deg = float(np.clip(np.degrees(u_hat[2]), -25.0, 25.0))
        target_yawrate_deg = float(np.clip(np.degrees(u_hat[3]) / _DT_LQR, -200.0, 200.0))
        # Note: for yaw, u_hat[3] is a target angle; convert to rate as
        # (target_yaw - current_yaw) / dt for the CF's rate-mode yaw channel
        current_yaw = state_9[8]
        yaw_error = u_hat[3] - current_yaw
        target_yawrate_deg = float(np.clip(np.degrees(yaw_error) / _DT_LQR, -200.0, 200.0))

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
            "goal_x": float(GOAL_XY[0]),
            "goal_y": float(GOAL_XY[1]),
            "u_hat_thrust": float(u_hat[0]),
            "u_hat_roll": float(u_hat[1]),
            "u_hat_pitch": float(u_hat[2]),
            "u_hat_yaw": float(u_hat[3]),
            "target_roll_deg": target_roll_deg,
            "target_pitch_deg": target_pitch_deg,
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
