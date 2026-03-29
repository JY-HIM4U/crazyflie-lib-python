#!/usr/bin/env python3
"""
Hardware flight script for Crazyflie using STL-Rollout trajectory planning.

Modes:
  MODE = "baseline"  -> MILP base policy trajectory
  MODE = "rollout"   -> MILP rollout-improved trajectory

At each planning step the policy computes the next action from the real
drone state, the drone flies to the target, and the loop repeats.
Position setpoints are sent via send_position_setpoint(x, y, z, yaw=0).
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

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig

# ============================================================
# Rollout-STL imports
# ============================================================
ROLLOUT_STL_DIR = os.path.join(os.path.dirname(__file__), "..", "Rollout-STL", "drone exp", "drone")
sys.path.insert(0, os.path.abspath(ROLLOUT_STL_DIR))

from drone_dynamics import DroneParams, clamp_vec, control_bounds, step_quad_simplified
from milp_base_policy import MILPBasePolicy
from rollout_policy_cd import CandidateGrid, coordinate_descent_one_step
from stl_task_eval import STLEvaluator
from task_spec import A3, get_task, mu_not_poly

# =========================
# Config
# =========================
URIS = [
    'radio://0/80/2M/E7E7E7E710',
]

# ---------- Mode selection ----------
# "baseline" = MILP base policy
# "rollout"  = MILP rollout-improved policy
MODE = "rollout"

# ---------- STL task config ----------
TASK_ID = "task10"
DT_PLAN = 0.5          # planning timestep (seconds) — must match rollout_env
TMAX = 20.0             # total planning horizon
WIN_STEPS = 4
LAMBDA_U = 0.1
STL_TAU = 50.0
STL_HARD = True
GRID_POINTS_XY = 4
GRID_POINTS_Z = 4

# ---------- Flight config ----------
DEFAULT_HEIGHT = 0.35   # m
POSITION_SEND_RATE = 0.05  # send setpoint every 50 ms (20 Hz)

# ---------- Workspace mapping (planner <-> real world) ----------
# Real-world coordinates are obtained from planner coordinates by:
#   x_real = x_start_real + POS_SCALE_XY * (x_plan - x_start_plan)
#   y_real = y_start_real + POS_SCALE_XY * (y_plan - y_start_plan)
#   z_real = z_start_real + POS_SCALE_Z  * (z_plan - z_start_plan) + Z_SHIFT
# and conversely when mapping real -> planner.
POS_SCALE_XY = 1.0   # < 1.0 shrinks the workspace in X/Y, > 1.0 enlarges it
POS_SCALE_Z = 1.0    # vertical scale
Z_SHIFT = 0.0        # constant offset added in metres to all real-world z

logging.basicConfig(level=logging.ERROR)

# =========================
# Global state containers
# =========================
fullstate: Dict[str, Dict[str, float]] = {}
crazyflies: List[SyncCrazyflie] = []
start_barrier = Barrier(len(URIS))
_stop_event = threading.Event()

# =========================
# Logging setup
# =========================
LOG_DIR = os.path.join(os.getcwd(), "cf_logs", time.strftime("%Y%m%d_%H%M%S"))
os.makedirs(LOG_DIR, exist_ok=True)

_measurements: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
_meas_lock = threading.Lock()


def _sanitize_uri(uri: str) -> str:
    return uri.replace("://", "_").replace("/", "_").replace(":", "_")


def record_sample(uri: str, state_vec: np.ndarray, extra: Dict[str, Any] | None = None):
    row = {
        "t": time.time(),
        "x": float(state_vec[0]),
        "y": float(state_vec[1]),
        "z": float(state_vec[2]),
        "qx": float(state_vec[3]),
        "qy": float(state_vec[4]),
        "qz": float(state_vec[5]),
        "qw": float(state_vec[6]),
        "vx": float(state_vec[7]),
        "vy": float(state_vec[8]),
        "vz": float(state_vec[9]),
        "gyro_x": float(state_vec[10]),
        "gyro_y": float(state_vec[11]),
        "gyro_z": float(state_vec[12]),
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

    plt.figure(figsize=(6, 6))
    for uri, rows in _measurements.items():
        if not rows:
            continue
        xs = [r["x"] for r in rows]
        ys = [r["y"] for r in rows]
        plt.plot(xs, ys, label=uri)
        uri_fig = os.path.join(LOG_DIR, f"{current_datetime}_{_sanitize_uri(uri)}_xy.png")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.plot(xs, ys)
        ax2.set_aspect("equal", adjustable="box")
        ax2.set_xlabel("x [m]")
        ax2.set_ylabel("y [m]")
        ax2.set_title(f"XY Trajectory: {uri}")
        fig2.savefig(uri_fig, dpi=200, bbox_inches="tight")
        plt.close(fig2)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"XY Trajectory ({MODE})")
    plt.legend()
    combined_fig = os.path.join(LOG_DIR, f"{current_datetime}_all_xy.png")
    plt.savefig(combined_fig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[logger] Saved combined XY plot to {combined_fig}")


# =========================
# Crazyflie helpers
# =========================
def make_logconfs(uri: str):
    logconfs = []
    lc1 = LogConfig(name='PosVel', period_in_ms=25)
    lc1.add_variable('stateEstimateZ.x', 'float')
    lc1.add_variable('stateEstimateZ.y', 'float')
    lc1.add_variable('stateEstimateZ.z', 'float')
    lc1.add_variable('stateEstimateZ.vx', 'float')
    lc1.add_variable('stateEstimateZ.vy', 'float')
    lc1.add_variable('stateEstimateZ.vz', 'float')
    lc1.data_received_cb.add_callback(log_cb_factory(uri))
    logconfs.append(lc1)

    lc2 = LogConfig(name='Attitude', period_in_ms=25)
    lc2.add_variable('gyro.x', 'float')
    lc2.add_variable('gyro.y', 'float')
    lc2.add_variable('gyro.z', 'float')
    lc2.add_variable('stabilizer.thrust', 'float')
    lc2.data_received_cb.add_callback(log_cb_factory(uri))
    logconfs.append(lc2)

    lc3 = LogConfig(name='Quaternion', period_in_ms=25)
    lc3.add_variable('stateEstimate.qx', 'float')
    lc3.add_variable('stateEstimate.qy', 'float')
    lc3.add_variable('stateEstimate.qz', 'float')
    lc3.add_variable('stateEstimate.qw', 'float')
    lc3.data_received_cb.add_callback(log_cb_factory(uri))
    logconfs.append(lc3)
    return logconfs


def log_cb_factory(uri: str):
    def _cb(ts, data, logconf):
        d = fullstate.get(uri)
        if d is None:
            fullstate[uri] = dict(data)
        else:
            d.update(data)
    return _cb


def connect(uri: str):
    scf = SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache'))
    scf.open_link()
    time.sleep(0.2)
    scf.cf.param.set_value("stabilizer.estimator", 2)
    scf.cf.param.set_value("commander.enHighLevel", 1)
    scf.cf.param.set_value("flightmode.stabModeRoll", 0)
    scf.cf.param.set_value("flightmode.stabModePitch", 0)
    scf.cf.param.set_value("flightmode.stabModeYaw", 0)
    scf.cf.param.set_value("stabilizer.controller", 1)
    logconfs = make_logconfs(uri)
    scf._logconfs = logconfs
    for lc in logconfs:
        try:
            scf.cf.log.add_config(lc)
            lc.start()
        except AttributeError as e:
            print(f"[{uri}] Failed to add or start log config '{lc.name}': {e}")
            sys.exit(1)
    crazyflies.append(scf)


def get_state_SI(uri: str):
    s = fullstate.get(uri, {})
    x     = float(s.get('stateEstimateZ.x', 0.0)) * 1e-3
    y     = float(s.get('stateEstimateZ.y', 0.0)) * 1e-3
    z     = float(s.get('stateEstimateZ.z', 0.0)) * 1e-3
    vx    = float(s.get('stateEstimateZ.vx', 0.0)) * 1e-3
    vy    = float(s.get('stateEstimateZ.vy', 0.0)) * 1e-3
    vz    = float(s.get('stateEstimateZ.vz', 0.0)) * 1e-3
    quatx = float(s.get('stateEstimate.qx', 0.0))
    quaty = float(s.get('stateEstimate.qy', 0.0))
    quatz = float(s.get('stateEstimate.qz', 0.0))
    quatw = float(s.get('stateEstimate.qw', 0.0))
    thrust = float(s.get('stabilizer.thrust', 0.0))
    gyro_x = float(s.get('gyro.x', 0.0)) / 180.0 * np.pi
    gyro_y = float(s.get('gyro.y', 0.0)) / 180.0 * np.pi
    gyro_z = float(s.get('gyro.z', 0.0)) / 180.0 * np.pi
    return np.array([x, y, z, quatx, quaty, quatz, quatw, vx, vy, vz,
                     gyro_x, gyro_y, gyro_z], dtype=np.float32), thrust


def smooth_send_hover(cf, start_setpoint, target_setpoint, duration):
    steps = int(duration / 0.025)
    for i in range(steps):
        f = (i + 1) / steps
        vx = start_setpoint[0] + (target_setpoint[0] - start_setpoint[0]) * f
        vy = start_setpoint[1] + (target_setpoint[1] - start_setpoint[1]) * f
        yr = start_setpoint[2] + (target_setpoint[2] - start_setpoint[2]) * f
        z  = start_setpoint[3] + (target_setpoint[3] - start_setpoint[3]) * f
        cf.commander.send_hover_setpoint(vx, vy, yr, z)
        time.sleep(0.025)


# ============================================================
# STL planner setup (called once, returns objects used each step)
# ============================================================
def build_stl_planner():
    """
    Build all planner objects once.  Returns a dict with everything the
    online loop needs to call step-by-step.
    """
    task = get_task(task_id=TASK_ID, win_steps_override=WIN_STEPS)
    params = DroneParams()
    u_min, u_max = control_bounds(params)
    t_steps = int(round(TMAX / DT_PLAN))
    win_len = task.win_len(DT_PLAN)

    def step_wrapped(x, u):
        return step_quad_simplified(x, u, dt=DT_PLAN,
                                    world_min=task.world_min, world_max=task.world_max, params=params)

    def next_state_safe(x, u, mrg=0.02):
        x2 = step_wrapped(x, u)
        pos = x2[:3]
        if not (task.world_min + mrg <= pos[0] <= task.world_max - mrg and
                task.world_min + mrg <= pos[1] <= task.world_max - mrg and
                task.world_min + mrg <= pos[2] <= task.world_max - mrg):
            return False
        for obs_name in task.obstacle_names:
            if not (mu_not_poly(A3, task.obstacle_b(obs_name), pos) > 0.0):
                return False
        return True

    def safe_apply(x, u):
        u = clamp_vec(u, u_min, u_max)
        if next_state_safe(x, u):
            return step_wrapped(x, u)
        scale = 1.0
        for _ in range(10):
            scale *= 0.5
            u_try = clamp_vec(scale * u, u_min, u_max)
            if next_state_safe(x, u_try):
                return step_wrapped(x, u_try)
        return x.copy()

    stl_eval = STLEvaluator(task=task, dt=DT_PLAN, tau=STL_TAU, hard=STL_HARD)
    policy = MILPBasePolicy(
        task=task, params=params, u_min=u_min, u_max=u_max,
        win_len=win_len, dt=DT_PLAN, horizon_steps=t_steps,
    )

    ax_grid = np.linspace(float(u_min[0]), float(u_max[0]), GRID_POINTS_XY)
    ay_grid = np.linspace(float(u_min[1]), float(u_max[1]), GRID_POINTS_XY)
    az_grid = np.linspace(float(u_min[2]), float(u_max[2]), GRID_POINTS_Z)
    cand_grid = CandidateGrid(per_dim_values=[ax_grid, ay_grid, az_grid])

    def ctx_step_from_x(x, ctx):
        return policy.phase_step_update(x, ctx)

    def base_action_from_ctx(x, ctx):
        return policy.base_control(x, ctx)

    def eval_scores_from_traj(xs):
        return stl_eval.eval_traj(xs)

    def is_satisfied_scores(sc, ju, eps=0.0):
        return float(sc[-1]) >= -eps

    def total_cost_scores(sc, ju):
        return -float(sc[-1]) + LAMBDA_U * ju

    def optimize_one_step(traj_prefix, x, phase, step_idx):
        return coordinate_descent_one_step(
            traj_prefix=traj_prefix, x=x, ctx=phase, step_idx=step_idx,
            T_total=t_steps, base_action=base_action_from_ctx,
            ctx_step=ctx_step_from_x, safe_apply=safe_apply,
            eval_scores=eval_scores_from_traj, is_satisfied=is_satisfied_scores,
            total_cost=total_cost_scores, cand_grid=cand_grid,
            u_min=u_min, u_max=u_max, rollout_enabled=True, rollout_filter=lambda ctx: True,
        )

    def choose_no_worse(traj_prefix, x, phase, step_idx, ju):
        u_base = policy.base_control(x, phase)
        u_roll = optimize_one_step(traj_prefix, x, phase, step_idx)
        if np.allclose(u_roll, u_base, atol=1e-8, rtol=0.0):
            return u_base
        x_base = safe_apply(x, u_base)
        x_roll = safe_apply(x, u_roll)
        sc_base = eval_scores_from_traj(list(traj_prefix) + [x_base])
        sc_roll = eval_scores_from_traj(list(traj_prefix) + [x_roll])
        j_base = total_cost_scores(sc_base, ju + float(u_base @ u_base))
        j_roll = total_cost_scores(sc_roll, ju + float(u_roll @ u_roll))
        tol = 1e-10
        if j_roll <= j_base + tol and float(sc_roll[-1]) >= float(sc_base[-1]) - tol:
            return u_roll
        return u_base

    return {
        "task": task,
        "policy": policy,
        "safe_apply": safe_apply,
        "step_wrapped": step_wrapped,
        "stl_eval": stl_eval,
        "choose_no_worse": choose_no_worse,
        "t_steps": t_steps,
    }


def drone_state_to_planner_state(
    state_13: np.ndarray,
    plan_start: np.ndarray,
    drone_start: np.ndarray,
) -> np.ndarray:
    """
    Convert 13D Crazyflie state (real world) to 12D planner state.

    This uses the exact inverse of the affine mapping used in
    `planner_pos_to_drone_pos`:

      x_real = x_start_real + POS_SCALE_XY * (x_plan - x_start_plan)
      y_real = y_start_real + POS_SCALE_XY * (y_plan - y_start_plan)
      z_real = z_start_real + POS_SCALE_Z  * (z_plan - z_start_plan) + Z_SHIFT
    """
    x_real = float(state_13[0])
    y_real = float(state_13[1])
    z_real = float(state_13[2])
    vx_real = float(state_13[7])
    vy_real = float(state_13[8])
    vz_real = float(state_13[9])

    x0_p, y0_p, z0_p = float(plan_start[0]), float(plan_start[1]), float(plan_start[2])
    x0_r, y0_r, z0_r = float(drone_start[0]), float(drone_start[1]), float(drone_start[2])

    x_plan = np.zeros(12, dtype=np.float64)
    # Invert affine XY mapping
    x_plan[0] = (x_real - x0_r) / POS_SCALE_XY + x0_p
    x_plan[1] = (y_real - y0_r) / POS_SCALE_XY + y0_p
    # Invert affine Z mapping
    x_plan[2] = (z_real - z0_r - Z_SHIFT) / POS_SCALE_Z + z0_p

    # Velocities: inverse scaling (no offset)
    x_plan[3] = vx_real / POS_SCALE_XY
    x_plan[4] = vy_real / POS_SCALE_XY
    x_plan[5] = vz_real / POS_SCALE_Z
    return x_plan


def planner_pos_to_drone_pos(
    plan_pos: np.ndarray,
    plan_start: np.ndarray,
    drone_start: np.ndarray,
) -> np.ndarray:
    """
    Convert planner [x,y,z] back to drone world [x,y,z] using the affine mapping:

      x_real = x_start_real + POS_SCALE_XY * (x_plan - x_start_plan)
      y_real = y_start_real + POS_SCALE_XY * (y_plan - y_start_plan)
      z_real = z_start_real + POS_SCALE_Z  * (z_plan - z_start_plan) + Z_SHIFT
    """
    x0_p, y0_p, z0_p = float(plan_start[0]), float(plan_start[1]), float(plan_start[2])
    x0_r, y0_r, z0_r = float(drone_start[0]), float(drone_start[1]), float(drone_start[2])

    dx_p = float(plan_pos[0]) - x0_p
    dy_p = float(plan_pos[1]) - y0_p
    dz_p = float(plan_pos[2]) - z0_p

    x_real = x0_r + POS_SCALE_XY * dx_p
    y_real = y0_r + POS_SCALE_XY * dy_p
    z_real = z0_r + POS_SCALE_Z * dz_p + Z_SHIFT

    return np.array([x_real, y_real, z_real], dtype=np.float64)


# ============================================================
# Main flight routine (online step-by-step planning)
# ============================================================
def test(cf, uri: str):
    # --- 1. Build planner ---
    print(f"[STL] Building planner (task={TASK_ID}, mode={MODE})...")
    planner = build_stl_planner()
    task = planner["task"]
    policy = planner["policy"]
    safe_apply = planner["safe_apply"]
    choose_no_worse = planner["choose_no_worse"]
    stl_eval = planner["stl_eval"]
    t_steps = planner["t_steps"]
    print(f"[flight] T={t_steps} steps, dt={DT_PLAN}s, total ~{t_steps * DT_PLAN:.1f}s")

    # --- 2. Take off ---
    current_setpoint = (0.0, 0.0, 0.0, 0.0)
    target_setpoint = (0.0, 0.0, 0.0, DEFAULT_HEIGHT)
    smooth_send_hover(cf, current_setpoint, target_setpoint, 3.0)
    time.sleep(1.0)

    # --- 3. Compute frame offset (planner <-> drone) ---
    state_13, _ = get_state_SI(uri)
    drone_start_xy = np.array([state_13[0], state_13[1]], dtype=np.float64)
    drone_start_z = float(state_13[2])
    plan_start = task.x0[0:3]
    drone_start = np.array([drone_start_xy[0], drone_start_xy[1], drone_start_z], dtype=np.float64)
    # For debugging: offsets in planner frame corresponding to initial alignment
    offset_xy = plan_start[0:2] - drone_start_xy
    offset_z = plan_start[2] - drone_start_z
    print(f"[flight] Drone start = ({drone_start[0]:.3f}, {drone_start[1]:.3f}, {drone_start[2]:.3f})")
    print(f"[flight] Plan  start = ({plan_start[0]:.3f}, {plan_start[1]:.3f}, {plan_start[2]:.3f})")
    print(f"[flight] Offsets XY=({offset_xy[0]:.3f},{offset_xy[1]:.3f}), Z={offset_z:.3f}")
    print(f"[flight] Scales XY={POS_SCALE_XY}, Z={POS_SCALE_Z}, Z_SHIFT={Z_SHIFT}")

    # --- 4. Online planning + flight loop ---
    phase = policy.initial_phase()
    traj_prefix = [drone_state_to_planner_state(state_13, plan_start, drone_start)]
    ju = 0.0
    flight_start_time = time.time()

    for k in range(t_steps):
        if _stop_event.is_set():
            print(f"[flight] Stop requested at step {k+1}/{t_steps}; landing...")
            break

        step_start = time.time()

        # Read current drone state, convert to planner frame (inverse affine mapping)
        state_13, _ = get_state_SI(uri)
        x_plan = drone_state_to_planner_state(state_13, plan_start, drone_start)

        # Update phase and compute action
        phase = policy.phase_step_update(x_plan, phase)
        if MODE == "rollout":
            u = choose_no_worse(traj_prefix, x_plan, phase, k, ju)
        else:
            u = policy.base_control(x_plan, phase)
        ju += float(u @ u)

        # Predict next state in planner frame (for STL evaluation and trajectory prefix)
        x_plan_next = safe_apply(x_plan, u)
        traj_prefix.append(x_plan_next.copy())

        # Convert predicted target position to drone frame
        target_drone = planner_pos_to_drone_pos(x_plan_next[0:3], plan_start, drone_start)

        compute_time = time.time() - step_start
        print(f"[flight] Step {k+1}/{t_steps}  compute={compute_time*1000:.0f}ms  "
              f"target=({target_drone[0]:.3f},{target_drone[1]:.3f},{target_drone[2]:.3f})  "
              f"u=({u[0]:.2f},{u[1]:.2f},{u[2]:.2f})    "
              f"Current state: {state_13[0]:.3f},{state_13[1]:.3f},{state_13[2]:.3f}")

        # Send position setpoint for DT_PLAN seconds
        segment_start = time.time()
        while time.time() - segment_start < DT_PLAN and not _stop_event.is_set():
            cf.commander.send_position_setpoint(
                float(target_drone[0]),
                float(target_drone[1]),
                float(target_drone[2]),
                0.0,
            )
            state_13_log, thrust = get_state_SI(uri)
            record_sample(uri, state_13_log, {
                "step": k,
                "ref_x": float(target_drone[0]),
                "ref_y": float(target_drone[1]),
                "ref_z": float(target_drone[2]),
                "u_ax": float(u[0]),
                "u_ay": float(u[1]),
                "u_az": float(u[2]),
                "flight_time": time.time() - flight_start_time,
                "mode": MODE,
            })
            time.sleep(POSITION_SEND_RATE)

    if not _stop_event.is_set():
        # --- 5. Evaluate STL robustness on the actual trajectory prefix ---
        scores = stl_eval.eval_traj(traj_prefix)
        print(f"[STL] Final robustness ({MODE}) = {float(scores[-1]):.3f}")

        # --- 6. Hover at final position for 2 seconds ---
        final_target = planner_pos_to_drone_pos(traj_prefix[-1][0:3], plan_start, drone_start)
        hover_end = time.time() + 2.0
        while time.time() < hover_end and not _stop_event.is_set():
            cf.commander.send_position_setpoint(
                float(final_target[0]), float(final_target[1]), float(final_target[2]), 0.0)
            time.sleep(POSITION_SEND_RATE)

    # Always land the drone
    try:
        state_13, _ = get_state_SI(uri)
        current_z = float(state_13[2])
    except Exception:
        current_z = DEFAULT_HEIGHT
    smooth_send_hover(cf, (0.0, 0.0, 0.0, current_z), (0.0, 0.0, 0.0, 0.05), 3.0)
    cf.commander.send_stop_setpoint()


def fly_all():
    threads = []
    for scf in crazyflies:
        t = Thread(target=test, args=(scf.cf, scf._link_uri))
        t.daemon = True
        t.start()
        threads.append(t)

    try:
        for t in threads:
            while t.is_alive():
                t.join(timeout=0.2)
    except KeyboardInterrupt:
        print("\n[main] Ctrl+C received — signalling flight threads to stop...")
        _stop_event.set()
        for t in threads:
            t.join(timeout=10.0)


# =========================
# Main
# =========================
if __name__ == '__main__':
    try:
        cflib.crtp.init_drivers()
        for uri in URIS:
            connect(uri)
        fly_all()
    finally:
        save_measurements_to_disk()
        for scf in crazyflies:
            try:
                scf.close_link()
            except Exception:
                pass
