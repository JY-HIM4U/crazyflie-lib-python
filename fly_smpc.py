#!/usr/bin/env python3
import logging
import time
from threading import Thread, Barrier
import random
import math
import sys
import os
from collections import defaultdict
import threading
import csv

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
import json
import torch

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig

# --- Your modules (leave as-is if you use them) ---
from parameters import load_parameters
from smpc import shrinking_horizon_SMPC, DAQP_fast, QP
from quadcopter_dynamics import quadcopter_dynamics_single_step_linear
# --------------------------------------------------
from RL_smpc_rl_agent import RLAgent, simple_minmax_normalize
from RL_smpc_world_map import WorldMap
from RL_smpc_config import *
from RL_smpc_simulation import PhoenixSimulator
# Fix argv[0] in some embedded/REPL environments
if not sys.argv:
    sys.argv = ['train_RL_smpc.py']
elif not sys.argv[0]:
    sys.argv[0] = 'train_RL_smpc.py'

# =========================
# Config
# =========================
URIS = [
    'radio://0/80/2M/E7E7E7E710',
    # add more URIs if needed
]
disturbance_enabled = 1
disturbance_strength = 0.3

agent = RLAgent(state_dim=46, action_dim=2)
# MODEL_PATH = "/home/realm/Jaeyoun/models/model_iter_531.pth"  # Set your model path here
# MODEL_PATH = "/home/realm/Jaeyoun/models/model_iter_100_RLMPC_1e-2noise.pth"  # Set your model path here
# MODEL_PATH = "/home/realm/Jaeyoun/models/model_iter_326_RLMPC_1e-4_Final.pth"  # Set your model path here
# MODEL_PATH = "/home/realm/Jaeyoun/models/model_iter_270_RLMPC_Final_1e-4.pth"  # Set your model path here
# MODEL_PATH = "/home/realm/Jaeyoun/models/model_iter_280_RLMPC_Final_1e-4.pth"  # Set your model path here
# MODEL_PATH = "/home/realm/Jaeyoun/models/model_iter_300_RLMPC_1e-4.pth"  # Set your model path here
MODEL_PATH = "/home/realm/Jaeyoun/models/model_iter_625_belief.pth"  # Set your model path here



# MODEL_PATH = "/home/realm/Jaeyoun/models/model_iter_178_RLMPC_final_nodisturb.pth"  # Set your model path here

# MODEL_PATH = "/home/realm/Jaeyoun/models/model_iter_600.pth"  # Set your model path here

DEFAULT_HEIGHT = 0.35          # 0.35 m (35 cm)
VELOCITY = 0.1                 # m/s
HOVER_PWM= 30000

RL_DECISION_INTERVAL = 1.00    # RL decides every 0.25s
MPC_PLANNING_INTERVAL = 0.05   # MPC plans every 0.01s  
PHYSICS_TIME_STEP = 0.005      # Physics runs at 500Hz (much faster!)
TARGET_POSITION = np.array([1, 2, 1])
REPEATS=1
START_TIME = time.time()  # Global start time for tracking overall execution
END_TIME = time.time()    # Global end time for tracking overall execution

logging.basicConfig(level=logging.ERROR)

# =========================
# Global state containers
# =========================
# Per-CF latest log packet: { uri: {var_name: value, ...} }
fullstate: Dict[str, Dict[str, float]] = {}
crazyflies: List[SyncCrazyflie] = []
start_barrier = Barrier(len(URIS))

# =========================
# Logging setup
# =========================
LOG_DIR = os.path.join(os.getcwd(), "cf_logs", time.strftime("%Y%m%d_%H%M%S"))
os.makedirs(LOG_DIR, exist_ok=True)

# Per-URI list of recorded samples
_measurements: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
_meas_lock = threading.Lock()  # in case multiple CF threads append concurrently

def _sanitize_uri(uri: str) -> str:
    return uri.replace("://", "_").replace("/", "_").replace(":", "_")


class DisturbanceKF2D:
    """
    Lightweight 2D disturbance estimator using a Kalman-style update.

    State: d ~ N(mu, Sigma), mu in R^2, Sigma in R^{2x2}
    Dynamics: d_{k+1} = d_k + w_k,     w_k ~ N(0, Q)
    Measurement: r_k = d_k + v_k,      v_k ~ N(0, R)
    """

    def __init__(self, mu0=None, Sigma0=None, Q=None, R=None):
        self.dim = 2
        self._I = np.eye(self.dim, dtype=np.float64)
        self.eps = 1e-8

        if mu0 is None:
            self.mu = np.zeros(self.dim, dtype=np.float64)
        else:
            self.mu = np.asarray(mu0, dtype=np.float64).reshape(self.dim)

        if Sigma0 is None:
            self.Sigma = (1e-4) * np.eye(self.dim, dtype=np.float64)
        else:
            self.Sigma = np.asarray(Sigma0, dtype=np.float64).reshape(self.dim, self.dim)

        if Q is None:
            self.Q = (1e-5) * np.eye(self.dim, dtype=np.float64)
        else:
            self.Q = np.asarray(Q, dtype=np.float64).reshape(self.dim, self.dim)

        if R is None:
            self.R = (1e-4) * np.eye(self.dim, dtype=np.float64)
        else:
            self.R = np.asarray(R, dtype=np.float64).reshape(self.dim, self.dim)

    def predict(self):
        """Time update: Sigma <- Sigma + Q (mu unchanged)."""
        self.Sigma = self.Sigma + self.Q

    def update(self, r):
        """
        Measurement update with residual r (2D).

        K = Sigma (Sigma + R)^{-1}
        mu <- mu + K (r - mu)
        Sigma <- (I - K) Sigma
        """
        r = np.asarray(r, dtype=np.float64).reshape(self.dim)
        try:
            S = self.Sigma + self.R + self.eps * self._I
            S_inv = np.linalg.inv(S)
            K = self.Sigma @ S_inv
            innovation = r - self.mu
            self.mu = self.mu + K @ innovation
            self.Sigma = (self._I - K) @ self.Sigma
            if not np.all(np.isfinite(self.mu)) or not np.all(np.isfinite(self.Sigma)):
                raise FloatingPointError("Non-finite Kalman state")
        except (np.linalg.LinAlgError, FloatingPointError):
            # If inversion or update fails, keep previous state and slightly inflate covariance
            self.Sigma = self.Sigma + self.eps * self._I

    def get_mu(self):
        """Return current mean estimate as float32 vector."""
        return self.mu.astype(np.float32)


def record_sample(uri: str,
                  state_vec: np.ndarray,
                  thrust_pwm: float,
                  extra: Dict[str, Any] | None = None):
    """
    Store one time-stamped sample of measured state & thrust.
    state_vec is the 13D vector returned by get_state_SI (x,y,z,qx,qy,qz,qw,vx,vy,vz,gyro_x,gyro_y,gyro_z)
    """
    row = {
        "t": time.time(),        # absolute timestamp (s since epoch)
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
        "thrust_pwm": float(thrust_pwm),
    }
    if extra:
        row.update(extra)
    with _meas_lock:
        _measurements[uri].append(row)

def save_measurements_to_disk():
    """
    Write CSV per URI and save XY trajectory plot(s).
    """
    if not _measurements:
        print("[logger] No measurements collected; nothing to save.")
        return

    # Get current datetime for file prefix
    current_datetime = time.strftime("%Y%m%d_%H%M%S")

    # 1) CSVs
    for uri, rows in _measurements.items():
        if not rows:
            continue
        csv_path = os.path.join(LOG_DIR, f"{current_datetime}_{_sanitize_uri(uri)}_meas.csv")
        fieldnames = list(rows[0].keys())
        # Convert timestamps to relative time (t0 = first sample)
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

    # 2) XY plots
    # Single combined plot for all URIs
    plt.figure(figsize=(6, 6))
    made_any = False
    for uri, rows in _measurements.items():
        if not rows:
            continue
        xs = [r["x"] for r in rows]
        ys = [r["y"] for r in rows]
        plt.plot(xs, ys, label=uri)
        made_any = True

        # Also save per-URI XY
        uri_fig = os.path.join(LOG_DIR, f"{current_datetime}_{_sanitize_uri(uri)}_xy.png")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.plot(xs, ys)
        ax2.set_aspect("equal", adjustable="box")
        ax2.set_xlabel("x [m]")
        ax2.set_ylabel("y [m]")
        ax2.set_title(f"XY Trajectory: {uri}")
        fig2.savefig(uri_fig, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"[logger] Saved XY plot to {uri_fig}")

    if made_any:
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("XY Trajectories (all)")
        plt.legend()
        combined_fig = os.path.join(LOG_DIR, f"{current_datetime}_all_xy.png")
        plt.savefig(combined_fig, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[logger] Saved combined XY plot to {combined_fig}")

def load_model(agent:RLAgent, model_path: str):
    """Load the trained model weights into the existing RLAgent"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the trained weights into the existing actor_critic
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    agent.actor_critic.load_state_dict(state_dict)
    
    print(f"🎯 Model architecture:")
    print(f"   - Policy network: {agent.actor_critic.pi}")
    print(f"   - Value network: {agent.actor_critic.v}")
    

def state_to_input(state17, target_position, world_map):
    drone_pos_xy = state17[:2]
    local_grid = world_map.encode_local_grid(drone_pos_xy)
    augmented_obs = np.concatenate([state17, target_position[:2]-state17[0:2], local_grid], axis=0).astype(np.float32)
    return augmented_obs
def make_logconfs(uri: str):
    """
    Returns a list of LogConfigs (split into safe sizes) and attaches callbacks.
    """
    logconfs = []

    # Position + Velocity
    lc1 = LogConfig(name='PosVel', period_in_ms=25)
    lc1.add_variable('stateEstimateZ.x', 'float')
    lc1.add_variable('stateEstimateZ.y', 'float')
    lc1.add_variable('stateEstimateZ.z', 'float')
    lc1.add_variable('stateEstimateZ.vx', 'float')
    lc1.add_variable('stateEstimateZ.vy', 'float')
    lc1.add_variable('stateEstimateZ.vz', 'float')
    
    lc1.data_received_cb.add_callback(log_cb_factory(uri))
    logconfs.append(lc1)

    # Attitude + Angular rate
    lc2 = LogConfig(name='Attitude', period_in_ms=25)
    # lc2.add_variable('stabilizer.roll', 'float')
    # lc2.add_variable('stabilizer.pitch', 'float')
    # lc2.add_variable('stabilizer.yaw', 'float')
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
    """Creates a callback that merges latest log values per-URI."""
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

    # === Estimator & commander modes ===
    # Kalman on; use high-level if you’ll call hover setpoints
    scf.cf.param.set_value("stabilizer.estimator", 2)     # 2 = Kalman
    scf.cf.param.set_value("commander.enHighLevel", 1)    # 1 = High-level for hover setpoints
    scf.cf.param.set_value("flightmode.stabModeRoll", 0)    # 1 = High-level for hover setpoints
    scf.cf.param.set_value("flightmode.stabModePitch", 0)    # 1 = High-level for hover setpoints
    scf.cf.param.set_value("flightmode.stabModeYaw", 0)    # 1 = High-level for hover setpoints
    # Optional: set controller (0=PID, 1=INE, etc., depends on firmware)
    scf.cf.param.set_value("stabilizer.controller", 1)
    # posCtlNoise params may not exist on all firmware (e.g. stock cf2 firmware)
    try:
        scf.cf.param.set_value("posCtlNoise.enabled", disturbance_enabled)
        scf.cf.param.set_value("posCtlNoise.stddev", disturbance_strength)
    except KeyError:
        print(f"[{uri}] posCtlNoise params not in firmware TOC; disturbance injection disabled.")

    # If you insisted on low-level (enHighLevel=0), then DO NOT call send_hover_setpoint anywhere.

    # Register all log configs and KEEP references
    logconfs = make_logconfs(uri)
    scf._logconfs = logconfs            # <<< keep alive
    for lc in logconfs:
        try:
            scf.cf.log.add_config(lc)
            lc.start()
        except AttributeError as e:
            print(f"[{uri}] Failed to add or start log config '{lc.name}': {e}")
            sys.exit(f"[{uri}] Failed to add or start log config '{lc.name}': {e}")

    crazyflies.append(scf)

# =========================
# Helpers
# =========================


def get_state_SI(uri: str) -> np.ndarray:
    """
    Returns the latest state for the given CF URI as a 12-dim numpy array:
    [x, y, z, vx, vy, vz, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
    in SI units: m, m/s, rad, rad/s.
    Defaults to zeros if no data yet.
    """
    s = fullstate.get(uri, {})
    # Safe fetch with defaults
    x     = float(s.get('stateEstimateZ.x', 0.0)) * 1e-3
    y     = float(s.get('stateEstimateZ.y', 0.0)) * 1e-3
    z     = float(s.get('stateEstimateZ.z', 0.0)) * 1e-3
    vx    = float(s.get('stateEstimateZ.vx', 0.0)) * 1e-3 ## Body frame
    vy    = float(s.get('stateEstimateZ.vy', 0.0)) * 1e-3
    vz    = float(s.get('stateEstimateZ.vz', 0.0)) * 1e-3
    quatx  = float(s.get('stateEstimate.qx', 0.0))
    quaty  = float(s.get('stateEstimate.qy', 0.0))
    quatz  = float(s.get('stateEstimate.qz', 0.0))
    quatw  = float(s.get('stateEstimate.qw', 0.0))
    thrust = float(s.get('stabilizer.thrust', 0.0))
    gyro_x = float(s.get('gyro.x', 0.0)) / 180.0 * np.pi
    gyro_y = float(s.get('gyro.y', 0.0)) / 180.0 * np.pi
    gyro_z = float(s.get('gyro.z', 0.0)) / 180.0 * np.pi

    return np.array([x, y, z, quatx, quaty, quatz, quatw,  vx, vy, vz,
                     gyro_x, gyro_y, gyro_z], dtype=np.float32), thrust


def fly_all():
    threads = []
    for scf in crazyflies:
        t = Thread(target=test, args=(scf.cf, scf._link_uri))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

# =========================
# MPC wrapper that reads CF state
# =========================
def send_trpy_command_pwm(cf, thrust_pwm: int, rollrate: float, pitchrate: float, yawrate: float, duration: float):
    """Send one RPY-rate+thrust PWM command then sleep for duration."""
    cf.commander.send_setpoint(rollrate, pitchrate, yawrate, int(thrust_pwm))
    time.sleep(duration)
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
def shrinking_horizon_SMPC_phoenix(cf, uri: str,
                                   dp_params, mpc_params, sim_params,
                                   system_state, reference, Q, R,
                                   phoenix_sim=None, upper_layer_constraints=None):
    """
    Example MPC loop that:
      - solves your QP
      - sends RPYT (with PWM thrust)
      - reads the latest Crazyflie state (SI units) via logs
      - accumulates reward
    """
    global START_TIME, END_TIME  # Declare global variables for timing across iterations
    
    reward = 0.0
    x_trajectory = [np.asarray(system_state, dtype=np.float32)]
    control_actions = np.zeros((4, mpc_params['J']), dtype=np.float32)
    control_outputs = []
    
    isfirst=True
    # print("Starting MPC")
    for j in range(mpc_params['J']):
        # === Solve your QP (user-provided) ===
        u_sol, x_sol, epsilon_sol = QP(dp_params, mpc_params, x_trajectory[-1], Q, R, reference, j)
        if u_sol is None or x_sol is None:
            raise ValueError("QP solver did not return a valid solution.")

        # u_sol: [thrust(N), tau_x, tau_y, tau_z] (your convention)
        T_max= 0.027*9.81*2.25
        thrust_force = u_sol[0, 0] * T_max  # First control input is thrust
        hover_thrust = 0.027 * 9.81  # 0.265 N
        thrust_cmd = thrust_to_cmd(hover_thrust+thrust_force)

        

        # x_sol contains states; angular velocity (rad/s) in indices 9:12 (your convention)
        next_state_pred = x_sol[:, 1]

        # Build control (SI)
        current_control = np.array([thrust_cmd, u_sol[1,0]/ (np.pi/3), u_sol[2,0]/ (np.pi/3), u_sol[3,0]/ (np.pi/3)], dtype=np.float32)
        control_actions[:, j] = current_control

        # Map thrust to PWM for Crazyflie low-level command (rough linear map)
        thrust_pwm = int(np.clip(thrust_cmd, 0, 48000))

        # Apply control to CF for one MPC step
        # print("Sending command to CF")
        END_TIME = time.time()
        if (END_TIME - START_TIME < MPC_PLANNING_INTERVAL and isfirst == False ):
            time.sleep(MPC_PLANNING_INTERVAL-(END_TIME - START_TIME) )
        print("Time", END_TIME - START_TIME)
        
        send_trpy_command_pwm(cf,
                              thrust_pwm=thrust_pwm,
                              rollrate=float(u_sol[1,0]/ (np.pi/3)),   # rad/s, crazyflie expects deg/s for gyro? NO: send_setpoint uses rad/s for rates? Actually it expects deg/s-equivalent internal. Keep small rates or convert if needed.
                              pitchrate=float(u_sol[2,0]/ (np.pi/3)),
                              yawrate=float(u_sol[3,0]/ (np.pi/3)),
                              duration=0.0)
        isfirst=False
        print("Thrust",thrust_pwm)
        START_TIME = time.time()
        # Read the latest *measured* state from Crazyflie logs (SI)
        smpc_next_state, thrust = get_state_SI(uri)  # shape (9,)
        # print("SMPC next state",smpc_next_state)

        # Log the SMPC state and control
        record_sample(uri, smpc_next_state, thrust_pwm, {
            "mpc_step": j,
            "control_thrust": float(thrust_force),
            "control_rollrate": float(u_sol[1,0]/ (np.pi/3)),
            "control_pitchrate": float(u_sol[2,0]/ (np.pi/3)),
            "control_yawrate": float(u_sol[3,0]/ (np.pi/3)),
            "reward_step": float(reward_step)
        })

        # Append trajectory
        x_trajectory.append(smpc_next_state.copy())

        # Reward
        err = smpc_next_state - reference[:, j]
        reward_step = (err.T @ Q @ err) + (current_control.T @ R @ current_control)
        reward += float(reward_step) / sim_params['simulation_steps_per_input']

        control_outputs.append({
            'step': j,
            'control_action': current_control.copy(),
            'measured_state': smpc_next_state.copy(),
        })

    first_control_action = control_actions[:, 0] if control_actions.shape[1] > 0 else np.zeros(4, dtype=np.float32)
    return x_trajectory[-1], x_trajectory, reward, first_control_action, control_outputs


def smooth_send_hover(cf, start_setpoint, target_setpoint, duration):
    """
    Smoothly transition from start_setpoint to target_setpoint over the specified duration.
    
    Both setpoints are tuples: (vx, vy, yawrate, z)
    """
    steps = int(duration / 0.025)  # 20Hz update rate
    for i in range(steps):
        factor = (i + 1) / steps
        vx = start_setpoint[0] + (target_setpoint[0] - start_setpoint[0]) * factor
        vy = start_setpoint[1] + (target_setpoint[1] - start_setpoint[1]) * factor
        yawrate = start_setpoint[2] + (target_setpoint[2] - start_setpoint[2]) * factor
        z = start_setpoint[3] + (target_setpoint[3] - start_setpoint[3]) * factor
        print(vx,vy,yawrate,z)
        cf.commander.send_hover_setpoint(vx, vy, yawrate, z)
        time.sleep(0.025)

def test_SMPC(cf, uri: str, initial_system_state=np.zeros((9,), dtype=np.float32)):
    repeats = REPEATS
    com_params, dp_params, mpc_params, sim_params = load_parameters()
    system_state = initial_system_state.copy()

    # Build a simple helix reference
    reference = np.zeros( (9,(mpc_params['J']+1)*repeats), dtype=np.float32)  # Reference trajectory
    radius = 0.5

    REFERENCE_TYPE = "Hover"
    if REFERENCE_TYPE == "HELIX":
        radius = .5
        for j in range((mpc_params['J']+1)*repeats):
            theta = 2 * np.pi * j / mpc_params['J'] / repeats
            reference[0, j] = radius * np.cos(theta)
            reference[1, j] = radius * np.sin(theta)
            reference[2, j] = DEFAULT_HEIGHT + 0.01*j / repeats

    if REFERENCE_TYPE == "Hover":
        for j in range((mpc_params['J']+1)*repeats):
            reference[0, j] = 0 # Example ramp reference for the x position
            reference[1, j] = 0
            reference[2, j] = DEFAULT_HEIGHT
    if REFERENCE_TYPE == "XMOVE":
        for j in range((mpc_params['J']+1)*repeats):
            reference[0, j] = j * 0.01  # Example ramp reference for the x position
            reference[1, j] = 0
            reference[2, j] = DEFAULT_HEIGHT
    

    Q = mpc_params['MPC_Q']
    R = mpc_params['MPC_R']
    for i in range(repeats):
        reference_segment = reference[:, i*(mpc_params['J']+1):(i+1)*(mpc_params['J']+1)]
        terminal_state, x_trajectory, reward, first_control_action, control_outputs= shrinking_horizon_SMPC_phoenix(
            cf=cf,
            uri=uri,
            dp_params=dp_params,
            mpc_params=mpc_params,
            sim_params=sim_params,
            system_state=system_state,
            reference=reference_segment,
            Q=Q,
            R=R,
            phoenix_sim=None,
            upper_layer_constraints=None
        )
        system_state=terminal_state


def test(cf, uri: str, initial_system_state=np.zeros((9,), dtype=np.float32)):
    send_trpy_command_pwm(cf,
                              thrust_pwm=0,
                              rollrate=0,   # rad/s, crazyflie expects deg/s for gyro? NO: send_setpoint uses rad/s for rates? Actually it expects deg/s-equivalent internal. Keep small rates or convert if needed.
                              pitchrate=0,
                              yawrate=0,
                              duration=0.0)
    current_setpoint = (0.0, 0.0, 0.0, 0.0)

    agent= RLAgent(state_dim=46, action_dim=2)
    load_model(agent, MODEL_PATH)
    world_map = WorldMap(world_name=WORLD_MAP_NAME)
    goal_position = world_map.goal_center.copy()

    sim = PhoenixSimulator(WORLD_MAP_NAME)
    # plot_action_field(agent, world_map, goal_position, 0 , True, None,  True, sim) 
    
    target_setpoint = (0.0, 0.0, 0.0, DEFAULT_HEIGHT)
    smooth_send_hover(cf, current_setpoint, target_setpoint, 3.0)


    ### Start ####
    prev_action = np.zeros((4,), dtype=np.float32)
    
    # Continuous logging at 5Hz for a specified duration
    FLIGHT_DURATION = 30.0  # seconds - adjust as needed
    LOG_INTERVAL = 0.2      # 5Hz logging (1/5 = 0.2 seconds)
    
    start_time = time.time()
    last_action_time = -5.0
    current_action = np.array([0.0, 0.0], dtype=np.float32)  # Initialize with zero action
    current_state_value = 0.0
    trajectory = None  # Initialize trajectory
    step = 0
    q_scale = 1e-5
    r_scale = 1e-4
    Q = q_scale * np.eye(2, dtype=np.float64)
    R = r_scale * np.eye(2, dtype=np.float64)
    belief_filter = DisturbanceKF2D(
        mu0=np.zeros(2, dtype=np.float64),
        Sigma0=(1e-4) * np.eye(2, dtype=np.float64),
        Q=Q,
        R=R,
    )
    belief_vec = belief_filter.get_mu()
    prev_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    while time.time() - start_time < FLIGHT_DURATION:
        loop_start_time = time.time()
        current_time = time.time() - start_time
        
        current_state, thrust = get_state_SI(uri)  # shape (9,)
        current_state[2] = DEFAULT_HEIGHT
        prev_action = [thrust, current_state[-3], current_state[-2], current_state[-1]]
        
        # Generate new action only every RL_DECISION_INTERVAL seconds
        if current_time - last_action_time >= RL_DECISION_INTERVAL-0.1:
            current_state_input = np.concatenate([current_state, prev_action], axis=0)
            current_state_input = state_to_input(current_state_input, goal_position, world_map)
            augmented_obs = np.concatenate([current_state_input, belief_vec], axis=0).astype(np.float32)
            if(disturbance_enabled):
                current_state_input[0:2] += np.random.multivariate_normal(
                    np.zeros(2), 
                    disturbance_strength * np.eye(2)
                )
            with torch.no_grad():
                    normalized_state = simple_minmax_normalize(augmented_obs)
                    obs_tensor = torch.FloatTensor(normalized_state).unsqueeze(0)
                    # Get the policy distribution and use the mean (most probable action)
                    dist = agent.actor_critic.pi.dist(obs_tensor)
                    current_state_value = agent.actor_critic.v(obs_tensor).cpu().numpy().squeeze()
                    current_action = dist.mean.cpu().numpy().squeeze().astype(np.float32)
            
            print(f"New action generated: {current_action} at time {current_time:.1f}s")
            last_action_time = current_time

                # Apply the current action
            velocity_cmd = np.array([current_action[0], current_action[1], 0.0], dtype=float)
            
            mpc_horizon = int(RL_DECISION_INTERVAL / MPC_PLANNING_INTERVAL)
            trajectory = np.zeros((6, mpc_horizon + 1))
            trajectory[:, 0] = np.concatenate([current_state[0:3], current_state[7:10]])
            for i in range(mpc_horizon):
                # Update position based on velocity command
                trajectory[0:3, i+1] = trajectory[0:3, i] + velocity_cmd * 0.025 * ACTION_SCALE
                trajectory[3:6, i+1] = velocity_cmd * ACTION_SCALE 
                # Keep other states constant (simplified)
                trajectory[6:, i+1] = 0
        
            # Log the current state and action at 5Hz (using the latest action)
            pos = trajectory[0:3, -1]
            vel = trajectory[3:6, -1]
            acc = (vel - prev_vel) / 0.025
            record_sample(uri, current_state, thrust, {
                "step": step,
                "action_x": float(current_action[0]),
                "action_y": float(current_action[1]),
                "trajectory_x_start": float(trajectory[0,0]),
                "trajectory_y_start": float(trajectory[1,0]),
                "trajectory_x_end": float(trajectory[0,-1]),
                "trajectory_y_end": float(trajectory[1,-1]),
                "Belief_x": float(belief_vec[0]),
                "Belief_y": float(belief_vec[1]),
                "state_value": float(current_state_value),
                "goal_x": float(goal_position[0]),
                "goal_y": float(goal_position[1]),
                "pos_x": float(pos[0]),
                "pos_y": float(pos[1]),
                "pos_z": float(pos[2]),
                "vel_x": float(vel[0]),
                "vel_y": float(vel[1]),
                "vel_z": float(vel[2]),
                "acc_x": float(acc[0]),
                "acc_y": float(acc[1]),
                "acc_z": float(acc[2]),
                "flight_time": float(current_time),
                "action_age": float(current_time - last_action_time)  # How old the current action is
            })
            if(current_state[0] > 1.4 and current_state[1] < 0.2):
                break
        # Only send position setpoint if trajectory is available
        if trajectory is not None:
            # cf.commander.send_position_setpoint(trajectory[0,-1], trajectory[1,-1], DEFAULT_HEIGHT, 0.0)
            # cf.commander.send_hover_setpoint(velocity_cmd[0] * ACTION_SCALE, velocity_cmd[1] * ACTION_SCALE, 0.0, DEFAULT_HEIGHT)
            pos = trajectory[0:3, -1]
            vel = trajectory[3:6, -1]
            acc = (vel - prev_vel) / 0.025
            # Commander packs pos/vel/acc as int16 (value*1000); keep within ±32.767
            # LIM = 32.767
            # pos = np.clip(pos, -LIM, LIM)
            # vel = np.clip(vel, -LIM, LIM)
            acc = np.clip(acc, -2, 2)
            cf.commander.send_full_state_setpoint(pos, vel, acc, np.array([0.0, 0.0, 0.0, 1.0]), 0.0, 0.0, 0.0)
            print(f"Sent full state setpoint pos vell acc: {pos}, {vel}, {acc}")
            
        
        # Sleep to maintain 5Hz logging rate
        elapsed = time.time() - loop_start_time
        sleep_time = max(0, LOG_INTERVAL - elapsed)
        time.sleep(sleep_time)
        current_state, thrust = get_state_SI(uri)  # shape (9,)
        # residual_xy = current_state[0:2] - trajectory[0:2, -1]
        residual_xy = (current_state[3:5] - velocity_cmd[0:2]*ACTION_SCALE) * 0.025
        r_k = np.asarray(residual_xy, dtype=np.float64).reshape(2)
        if np.all(np.isfinite(r_k)):
            belief_filter.update(r_k)
            belief_vec = belief_filter.get_mu()
            print(f"Updated belief: {belief_vec}")
            print(f"Current State: {current_state[0:5]}")
            print(f"Velocity Command: {velocity_cmd[0:2]*ACTION_SCALE}")
        step += 1
        print(f"Step {step}, Flight time: {current_time:.1f}s, Action age: {current_time - last_action_time:.1f}s")
        prev_vel = velocity_cmd

    current_state, _ = get_state_SI(uri)  # shape (9,)
    current_setpoint= (0.0, 0.0, 0.0, current_state[2])
    target_setpoint = (0.0, 0.0, 0.0, DEFAULT_HEIGHT)
    smooth_send_hover(cf, current_setpoint, target_setpoint, 1.0)

# =========================
# Main
# =========================
if __name__ == '__main__':

    # current_setpoint = (0.0, 0.0, 0.0, 0.0)

    # agent= RLAgent(state_dim=44, action_dim=2)
    # load_model(agent, MODEL_PATH)
    # world_map = WorldMap(world_name=WORLD_MAP_NAME)
    # goal_position = world_map.goal_center.copy()

    # sim = PhoenixSimulator(WORLD_MAP_NAME)
    # plot_action_field(agent, world_map, goal_position, 0 , True, "Test_results",  True, sim) 
    


    cflib.crtp.init_drivers()

    # Open links and start per-CF loggers (once)
    for uri in URIS:
        connect(uri)
    
    
    # Example: fly simple routine (reads/updates state in background)
    fly_all()

    # If you want to run MPC with the first CF:
    # scf0 = crazyflies[0]
    # terminal_state, traj, reward, first_u, logs = test_smpc_phoenix(scf0.cf, URIS[0])

    # Save all logged measurements to disk
    save_measurements_to_disk()
    
    # Close links cleanly
    for scf in crazyflies:
        try:
            scf.close_link()
        except Exception:
            pass
