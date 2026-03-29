#!/usr/bin/env python3
"""
Configuration file for RL+SMPC Training Pipeline
Contains all global constants, parameters, and configuration settings
"""

import os
import numpy as np
from datetime import datetime

# ===============================================
# ==  WORLD MAP CONFIGURATION  ==
# ===============================================

# Define start position using YOUR coordinate system - put it in a safe area
# Image (5,5) → World (6,6) → Physical [1.0~1.2m] × [1.0~1.2m] (safe area)
# START_POSITION = [0.2, 1.2, 1.0]  # Center of safe area
START_POSITION = [0.3, 0.2, 1.0]  # Center of safe area
# START_POSITION = [0.3, 1.8, 1.0]  # Slit
# START_POSITION = [0.3, 1.6, 1.0]  # DoubleSlit
# WORLD_MAP_NAME = "ra_50 _simple_but_exciting"
# WORLD_MAP_NAME = "slit_10"
# WORLD_MAP_NAME = "doubleslit_10"
WORLD_MAP_NAME = "ra_10"

# ===============================================
# ==  TRAINING CONFIGURATION  ==
# ===============================================

COST_LIMIT = 2.0
LAMBDA_LR = 1e-3

# Episode and training parameters
EPISODE_LENGTH = 15
REPEATS = int(os.environ.get("REPEATS", "1"))
ACTION_SCALE = 0.10
# ===============================================
# ==  TIMING CONFIGURATION  ==
# ===============================================
# MPC_PLANNING_INTERVAL must equal parameters.load_parameters() delta_t (LQR model step).
# mpc_horizon = RL_DECISION_INTERVAL / MPC_PLANNING_INTERVAL should match parameters J for fly_LQR.
# Default timing parameters (can be overridden by command line)
RL_DECISION_INTERVAL = 0.5
MPC_PLANNING_INTERVAL = 0.025
PHYSICS_TIME_STEP = 0.005
ENTROPY_CONSTANT = 0.01 #0.1
# ===============================================
# ==  DISTURBANCE CONFIGURATION  ==
# ===============================================

# Default disturbance settings (can be overridden by command line)
ENABLE_POSITION_DISTURBANCE = False
POSITION_DISTURBANCE_STRENGTH = 1e-6 #2e-3
DISTURBANCE_VERBOSE = False

# ===============================================
# ==  WIND DISTURBANCE CONFIGURATION  ==
# ===============================================
#
# Wind is applied as an external force in Newtons in the physics layer
# (see `phoenix_drone_simulation/envs/physics.py`).
#
# These parameters control the sampling ranges used to generate a fixed
# per-region (3 regions by X) wind configuration for each simulator instance.
#
# Wind enable flag (also exported to env var ENABLE_WIND_DISTURBANCE)
ENABLE_WIND_DISTURBANCE = True
WIND_VERBOSE = False
#
# Mean force magnitude is sampled as a fraction of weight (m*g):
#   frac_mean ~ U(WIND_MEAN_FRAC_MIN, WIND_MEAN_FRAC_MAX)
WIND_MEAN_FRAC_MIN = 0.10#0.02
WIND_MEAN_FRAC_MAX = 0.50#0.10
#
# Magnitude scale bounds for wind (direction preserved):
#   scale ~ U(WIND_SIGMA_FRAC_MIN, WIND_SIGMA_FRAC_MAX)  (used as [scale_min, scale_max])
WIND_SIGMA_FRAC_MIN = 0.9   # scale_min
WIND_SIGMA_FRAC_MAX = 1.1   # scale_max

# ===============================================
# ==  WANDB CONFIGURATION  ==
# ===============================================

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "phoenix-drone-rl-smpc")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)
WANDB_ENABLED = os.environ.get("WANDB_ENABLED", "true").lower() == "true"

# ===============================================
# ==  MODEL SAVING CONFIGURATION  ==
# ===============================================

# Create timestamped folder for model saving (will be created by main process only)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_SAVE_DIR = f"trained_model_{TIMESTAMP}"
# Don't create directory here - let main process handle it

# ===============================================
# ==  DEVICE CONFIGURATION  ==
# ===============================================

# Conservative default thread caps to avoid CPU oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Suppress PyBullet physics messages to reduce output noise
os.environ.setdefault("PYBULLET_QUIET", "1")
os.environ.setdefault("PHOENIX_QUIET", "1")

# Export wind disturbance controls for physics layer (force in N)
os.environ.setdefault("ENABLE_WIND_DISTURBANCE", "true" if ENABLE_WIND_DISTURBANCE else "false")
os.environ.setdefault("WIND_MEAN_FRAC_MIN", str(WIND_MEAN_FRAC_MIN))
os.environ.setdefault("WIND_MEAN_FRAC_MAX", str(WIND_MEAN_FRAC_MAX))
os.environ.setdefault("WIND_SIGMA_FRAC_MIN", str(WIND_SIGMA_FRAC_MIN))
os.environ.setdefault("WIND_SIGMA_FRAC_MAX", str(WIND_SIGMA_FRAC_MAX))

# ===============================================
# ==  RL AGENT CONFIGURATION  ==
# ===============================================

# PPO training parameters - More conservative for stability
CLIP_RATIO = 0.2
TARGET_KL = 0.01
TRAIN_PI_ITERATIONS = 24
TRAIN_V_ITERATIONS = 24
MAX_GRAD_NORM = 0.5
GAMMA = 0.95#0.95
LAM = 0.95#0.95  # GAE lambda

# Optimizers - Much lower learning rates for stability
PI_LEARNING_RATE = 1e-3
VF_LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = float(os.environ.get("MIN_LR", "1e-6"))

# Learning rate schedulers - More gradual decay
PI_SCHEDULER_STEP_SIZE = 100
PI_SCHEDULER_GAMMA = 0.95
VF_SCHEDULER_STEP_SIZE = 100
VF_SCHEDULER_GAMMA = 0.95

# ===============================================
# ==  DISTRIBUTED TRAINING CONFIGURATION  ==
# ===============================================

# Training parameters
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "50"))
DEFAULT_NUM_WORKERS = max(1, min(16, (os.cpu_count() or 2) - 1))

# Reward structure
TARGET_REWARD = 5.0
AVOID_PENALTY = -5.0
SURVIVE_REWARD = 1.0
ACTION_PENALTY_WEIGHT = 0.5

# ===============================================
# ==  NETWORK ARCHITECTURE  ==
# ===============================================

# State and action dimensions
PHOENIX_STATE_DIM = 17  # [pos(3), quat(4), vel(3), ang_vel(3), last_action(4)]
TOTAL_OBS_DIM = 44      # 17D state + 2D target + 25D local grid
ACTION_DIM = 2           # [vx, vy] horizontal velocity commands

# ActorCritic network configuration
# ACTOR_HIDDEN_SIZES = [64, 64]
# ACTOR_ACTIVATION = 'tanh'
# VALUE_HIDDEN_SIZES = [64, 64]
# VALUE_ACTIVATION = 'tanh'

ACTOR_HIDDEN_SIZES = [128, 128]
ACTOR_ACTIVATION = 'relu'
VALUE_HIDDEN_SIZES = [128, 128]
VALUE_ACTIVATION = 'relu'

# ===============================================
# ==  DISTRIBUTED TRAINING CONFIGURATION  ==
# ===============================================

# Default distributed training parameters

DEFAULT_ROLLOUT_LEN = 100
DEFAULT_SAVE_INTERVAL = 10
MAX_ITERATIONS = 100000
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "50"))

# ===============================================
# ==  PHYSICS CONFIGURATION  ==
# ===============================================

# PyBullet physics parameters
PHYSICS_SOLVER_ITERATIONS = 5
PHYSICS_DETERMINISTIC_OVERLAPPING_PAIRS = 1
PHYSICS_NUM_SUB_STEPS = 1

# Drone control parameters
MAX_ANGULAR_RATE = 2.0  # Maximum angular rate in rad/s
RATE_SCALE_FACTOR = 1.0

# ===============================================
# ==  REWARD CONFIGURATION  ==
# ===============================================

# World map rewards
TARGET_REWARD = 5.0
AVOID_PENALTY = -5.0
SURVIVE_REWARD = 1.0
ACTION_PENALTY_WEIGHT = 0.5

# Reward clipping and normalization
REWARD_CLIP_MIN = -20.0
REWARD_CLIP_MAX = 20.0
REWARD_NORMALIZE_EPSILON = 1e-8

# ===============================================
# ==  OBSERVATION CONFIGURATION  ==
# ===============================================

# State dimensions
PHOENIX_STATE_DIM = 17
TARGET_POS_DIM = 2
LOCAL_GRID_DIM = 25
TOTAL_OBS_DIM = PHOENIX_STATE_DIM + TARGET_POS_DIM + LOCAL_GRID_DIM  ## 4 works

# Action dimensions
ACTION_DIM = 2  # [vx, vy] horizontal velocity commands

# Min-max normalization lists for simple scaling (matching training script exactly)
OBS_MIN_VALUES = np.array([
    -2, -2, -2, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, 0, -3, -3, -3, -2, -2,      # target_pos_x, target_pos_y (relative to drone)
    0, 0, 0, 0, 0,  # local_grid (5x5 = 25 values)
    0, 0, 0, 0, 0,  # local_grid (5x5 = 25 values)
    0, 0, 0, 0, 0,  # local_grid (5x5 = 25 values)
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0
], dtype=np.float32)

# OBS_MIN_VALUES = np.array([
#     0, 0, 0, 0, 0, 0, 0, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0,      # target_pos_x, target_pos_y (relative to drone)
#     0, 0, 0, 0, 0,  # local_grid (5x5 = 25 values)
#     0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0
# ], dtype=np.float32)

OBS_MAX_VALUES = np.array([
    2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 60000, 3, 3, 3, 2, 2 ,          # target_pos_x, target_pos_y (relative to drone)
    2, 2, 2, 2, 2,  # local_grid (5x5 = 25 values)
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2
], dtype=np.float32)


## Works

# OBS_MIN_VALUES = np.array([
#     0,0,-2,-2
# ], dtype=np.float32)

# OBS_MAX_VALUES = np.array([
#     2,2,2,2
# ], dtype=np.float32)


# ===============================================
# ==  LOCAL GRID CONFIGURATION  ==
# ===============================================

LOCAL_GRID_SIZE = 5  # 5x5 local grid around drone
# GRID_RESOLUTION = 0.04 # 0.2  # meters per pixel
GRID_RESOLUTION = 0.2  # meters per pixel

# ===============================================
# ==  SMPC CONFIGURATION  ==
# ===============================================

# SMPC parameters
SMPC_ALPHA = 0.95  # Safety probability
SMPC_CONSTRAINT_EDGE_COORDINATES = np.array([[0, 0], [10, 10]]) 