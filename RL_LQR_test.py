#!/usr/bin/env python3
"""
Test script for trained RL+SMPC model with World Map

This script loads a trained model and runs test episodes to evaluate performance.
It tests the exact same problem as the training script and plots the trajectory for 15 RL steps.

Features:
✅ Loads trained model from specified path
✅ Uses same world map and configuration as training
✅ Runs 15 RL steps per episode (matching training)
✅ Plots trajectory with start, target, and drone path
✅ Shows step-by-step performance metrics
✅ Supports both basic (19D) and map (44D) model types

Usage:
    python RL_smpc_test.py --model-path /path/to/model.pth
    python RL_smpc_test.py --model-path /path/to/model.pth --num-episodes 5
    python RL_smpc_test.py --model-path /path/to/model.pth --render --save-plots

DEBUG MODE:
    Set MODEL_PATH_DEBUG and DEBUG_MODE = True at the top of this file
    Then run: python RL_smpc_test.py
    No command line arguments needed!
"""

import sys
import os
import argparse
import numpy as np
import torch
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display warnings
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import json
import math
from datetime import datetime
try:
    import pandas as pd
except Exception:
    pd = None

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules
from phoenix_drone_simulation.algs.core import ActorCritic
from phoenix_drone_simulation.envs.control import AttitudeRate
from phoenix_drone_simulation.envs.agents import CrazyFlieAgent
import pybullet as p
import pybullet_data

# Import our modularized components
from RL_smpc_config import *
from RL_smpc_utils import setup_device
from RL_smpc_world_map import WorldMap
from RL_smpc_simulation import PhoenixSimulator, StateConverter, TrajectoryGenerator
from RL_LQR_agent import RLAgent, simple_minmax_normalize  # Import the CMDP-enabled RL agent
from RL_LQR_controller import LQRController
from RL_smpc_config import ACTION_SCALE

# Import SMPC modules
try:
    from parameters import load_parameters
    from smpc import shrinking_horizon_SMPC, DAQP_fast, QP
    from quadcopter_dynamics import quadcopter_dynamics_single_step_linear
    SMPC_AVAILABLE = True
except ImportError:
    print("⚠️  SMPC modules not available, using simplified control")
    SMPC_AVAILABLE = False

# ============================================================================
# DEBUG MODE: Set this path to test without command line arguments
# ============================================================================
# MODEL_PATH_DEBUG = "/home/jaeyoun-choi/phoenix-drone-simulation/Stochastic_Hierarchies_Code/trained_model_20250830_171802/trained_model_rl_agent_iter_4000.pth"  # Set your model path here
# MODEL_PATH_DEBUG = "/home/jaeyoun-choi/phoenix-drone-simulation/trained_model_20250901_223755_RL_MPC_withoutdisturbance/model_iter_600.pth"  # Set your model path here
# MODEL_PATH_DEBUG = "/home/jaeyoun-choi/phoenix-drone-simulation/trained_model_20250910_113052_RLMPC_3e-2/model_iter_105.pth"  # Set your model path here
MODEL_PATH_DEBUG = "/home/jaeyoun-choi/phoenix-drone-simulation/Stochastic_Hierarchies_Code/trained_model_20250911_183952_RLMPC_Final_1e-4/model_iter_250.pth"  # Set your model path here
# MODEL_PATH_DEBUG = "/home/jaeyoun-choi/phoenix-drone-simulation/trained_model_20250912_110027_RLMPC_final_Nodisturb/model_iter_170.pth"  # Set your model path here

# MODEL_PATH_DEBUG = "/home/jaeyoun-choi/phoenix-drone-simulation/Stochastic_Hierarchies_Code/trained_model_20250913_122925_RLMPC_1e-4_Final_Easy/model_iter_195.pth"  # Set your model path here


# MODEL_PATH_DEBUG = "/home/jaeyoun-choi/phoenix-drone-simulation/trained_model_20250903_214346_RL_MPC_withdisturbance5e-3/model_iter_600.pth"  # Set your model path here

DEBUG_MODE = True  # Set to True to use debug path, False to use command line args
# ============================================================================


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test trained RL+SMPC model with World Map')
    
    # Required arguments (only required when not in debug mode)
    if not DEBUG_MODE:
        parser.add_argument('--model-path', type=str, required=True,
                           help='Path to the trained model file (.pth)')
    else:
        parser.add_argument('--model-path', type=str, default=MODEL_PATH_DEBUG,
                           help='Path to the trained model file (.pth) (default: debug path)')
    
    # Optional arguments
    parser.add_argument('--num-episodes', type=int, default=1,
                       help='Number of test episodes to run (default: 1)')
    parser.add_argument('--render', action='store_true', default=False,
                       help='Enable PyBullet GUI rendering (default: False)')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save trajectory plots to files (default: False)')
    parser.add_argument('--output-dir', type=str, default='test_results',
                       help='Output directory for results (default: test_results)')
    parser.add_argument('--qr-model-path', type=str, default=None,
                       help='Optional path to QR agent model (.pth) to override Q/R (default: auto-detect with _qr suffix)')
    
    return parser.parse_args()


class TestRLAgent:
    """Test version of RL agent that loads a trained model using existing RLAgent"""
    
    def __init__(self, model_path: str, model_type: str = 'auto'):
        self.model_path = model_path
        self.model_type = model_type
        
        # Infer obs_dim from checkpoint so we support both 44D and 47D (belief-augmented) policies.
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        inferred_obs_dim = self._infer_obs_dim_from_state_dict(state_dict)
        self.obs_dim = inferred_obs_dim

        self.agent = RLAgent(state_dim=inferred_obs_dim, action_dim=2, obs_dim=inferred_obs_dim)
        self.agent.actor_critic.load_state_dict(state_dict)
        
        # Set to evaluation mode
        self.agent.actor_critic.eval()
         
        print(f"✅ Model loaded successfully from: {model_path}")
        print(f"✅ Using existing RLAgent with trained weights")
        print(f"   - Inferred obs_dim: {self.obs_dim}")
    
    @staticmethod
    def _infer_obs_dim_from_state_dict(state_dict: Dict[str, Any]) -> int:
        # Heuristic: look for Linear weights whose input dim matches known obs sizes used in training.
        candidates = [44, 46]
        counts = {c: 0 for c in candidates}
        for _, v in state_dict.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                in_dim = int(v.shape[1])
                if in_dim in counts:
                    counts[in_dim] += 1
        best = max(counts.items(), key=lambda kv: kv[1])[0]
        return best if counts[best] > 0 else 44

    def load_model(self, model_path: str):
        """Load the trained model weights into the existing RLAgent"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the trained weights into the existing actor_critic
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        self.agent.actor_critic.load_state_dict(state_dict)
        
        print(f"🎯 Model architecture:")
        print(f"   - Policy network: {self.agent.actor_critic.pi}")
        print(f"   - Value network: {self.agent.actor_critic.v}")
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action for a single state using the existing RLAgent"""
        # Use the existing agent's get_action method
        action = self.agent.get_action(state)
        
        # Ensure action is numpy array and has correct shape
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        # Validate action shape - must be exactly 2D
        if action.ndim != 1 or len(action) != 2:
            raise ValueError(f"Expected action to be 2D, got shape {action.shape}. Action: {action}")
        
        # Clip to action space
        # action = np.clip(action, -1.0, 1.0)
        
        return action


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


def run_test_episode(agent: RLAgent, world_map: WorldMap, target_position: np.ndarray, 
                     render: bool = False, episode_num: int = 1,
                     qr_agent: RLAgent | None = None) -> Dict[str, Any]:
    """Run a single test episode"""
    print(f"\n🎯 Episode {episode_num}: Starting test")
    print("-" * 50)
    
    # Initialize simulator (render parameter not supported in PhoenixSimulator)
    # Pass wind params explicitly from config so edits in `RL_smpc_config.py` always apply.
    sim = PhoenixSimulator(
        control_mode='attitude_rate',
        disturbance_enabled=ENABLE_POSITION_DISTURBANCE,
        disturbance_strength=POSITION_DISTURBANCE_STRENGTH,
        wind_enabled=ENABLE_WIND_DISTURBANCE,
        wind_verbose=WIND_VERBOSE,
        wind_mean_frac_min=WIND_MEAN_FRAC_MIN,
        wind_mean_frac_max=WIND_MEAN_FRAC_MAX,
        wind_sigma_frac_min=WIND_SIGMA_FRAC_MIN,
        wind_sigma_frac_max=WIND_SIGMA_FRAC_MAX,
    )
    sim.reset(START_POSITION)
    
    # Initialize controllers using existing implementations
    ctrl = LQRController(phoenix_sim=sim)
    conv = StateConverter()
    trajgen = TrajectoryGenerator(dt=MPC_PLANNING_INTERVAL)

    policy_obs_dim = int(getattr(agent, "obs_dim", 44))
    use_belief_in_policy = policy_obs_dim > 44
    belief_filter = None
    belief_vec = None
    if use_belief_in_policy and policy_obs_dim == 46:
        try:
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
        except Exception:
            belief_filter = None
            belief_vec = None
    
    # Episode data collection
    episode_data = {
        'episode_num': episode_num,
        'positions': [START_POSITION.copy()],
        'velocities': [],
        'actions': [],
        'costs': [],
        'qr_Q_diag': [],
        'qr_R_scalar': [],
        'rewards': [],
        'distances': [],
        'total_steps': 0,
        'success': False,
        'final_distance': 0.0,
        'total_reward': 0.0,
        'smpc_failures': 0,
        'reference_traj': [],
        'beliefs': [],
        'residuals': [],
        'belief_positions': []
    }
    if belief_filter is not None and belief_vec is not None:
        episode_data['beliefs'].append(np.asarray(belief_vec, dtype=np.float64).tolist())
        episode_data['belief_positions'].append(START_POSITION.copy())
    
    # Get initial state
    state17 = sim.get_state()
    current_distance = np.linalg.norm(state17[:3] - target_position)
    episode_data['distances'].append(current_distance)
    
    print(f"   Start position: {START_POSITION[:2]}")
    print(f"   Target position: {target_position[:2]}")
    print(f"   Initial distance: {current_distance:.3f}m")
    print(f"   Episode length: {EPISODE_LENGTH} RL steps")
    print("")
    total_reward = 0.0
    step_count = 0
    # Run episode for exactly EPISODE_LENGTH RL steps
    for step in range(EPISODE_LENGTH):
        # Encode local grid around drone position for world map awareness
        # Physical position disturbance is already applied inside the physics layer (PyBulletPhysics/SimplePhysics),
        # so we use the true simulator state here without adding extra noise.
        state17_noised = state17.copy()
        base_obs = state_to_input(state17, target_position, world_map)
        if use_belief_in_policy and belief_filter is not None and policy_obs_dim == 46:
            try:
                belief_filter.predict()
                belief_vec = belief_filter.get_mu()
            except Exception:
                belief_vec = None
            if belief_vec is not None:
                augmented_obs = np.concatenate([base_obs, belief_vec], axis=0).astype(np.float32)
            else:
                augmented_obs = base_obs
        else:
            augmented_obs = base_obs
            
        # Get deterministic action (no exploration) by accessing policy network directly
        with torch.no_grad():
            normalized_state = simple_minmax_normalize(augmented_obs)
            obs_tensor = torch.FloatTensor(normalized_state).unsqueeze(0)
            # Get the policy distribution and use the mean (most probable action)
            dist = agent.actor_critic.pi.dist(obs_tensor)
            action = dist.mean.cpu().numpy().squeeze().astype(np.float32)
            episode_data['actions'].append(action.copy())

        
        # Apply action and get next state
        smpc_state = conv.phoenix_to_smpc_state(state17)
        smpc_state_noised = conv.phoenix_to_smpc_state(state17_noised)  # Convert to 9D SMPC state
        mpc_horizon = int(RL_DECISION_INTERVAL / MPC_PLANNING_INTERVAL)
        reference_traj = trajgen.generate_trajectory(smpc_state_noised, action, mpc_horizon=mpc_horizon)
        # Optionally override Q/R using a second agent
        Q_override = None
        R_override = None
        if qr_agent is not None:
            with torch.no_grad():
                dist_qr = qr_agent.actor_critic.pi.dist(obs_tensor)
                qr_out = dist_qr.mean.cpu().numpy().squeeze().astype(np.float32)
            # Map normalized outputs (-1..1) to requested uniform ranges using exponential interpolation
            #   Q_i in [0.1, 100], R in [1e-5, 1]
            q_min = np.full(9, 0.1, dtype=np.float32)
            q_max = np.full(9, 100.0, dtype=np.float32)
            r_min, r_max = 1e-5, 1.0
            q_raw = qr_out[:9]
            r_raw = qr_out[9]
            s_q = np.clip((q_raw + 1.0) * 0.5, 0.0, 1.0)
            s_r = float(np.clip((r_raw + 1.0) * 0.5, 0.0, 1.0))
            q_ratio = np.maximum(q_max / np.maximum(q_min, 1e-9), 1.0)
            q_diag = q_min * (q_ratio ** s_q)
            r_scalar = float(r_min * ((r_max / r_min) ** s_r))
            Q_override = np.diag(q_diag.astype(np.float64))
            R_override = r_scalar * np.eye(4, dtype=np.float64)
            # Log QR used this step
            episode_data['qr_Q_diag'].append(q_diag.tolist())
            episode_data['qr_R_scalar'].append(float(r_scalar))
            print(f"   QR agent: R={r_scalar:.4g}, Qdiag={[round(x,3) for x in q_diag.tolist()]}")
        control_actions, terminal_state, state_trajectory, control_outputs, terminal_state_phoenix = ctrl.compute_control(
            smpc_state, smpc_state_noised, reference_traj
        )

        # Disturbance belief update using residual_xy from LQRController
        if belief_filter is not None and use_belief_in_policy and policy_obs_dim == 46:
            try:
                residual_xy = getattr(ctrl, "last_residual_xy", None)
                if residual_xy is not None:
                    r_k = np.asarray(residual_xy, dtype=np.float64).reshape(2)
                    if np.all(np.isfinite(r_k)):
                        episode_data['residuals'].append(r_k.tolist())
                        belief_filter.update(r_k)
                        belief_vec = belief_filter.get_mu()
                        episode_data['beliefs'].append(
                            np.asarray(belief_vec, dtype=np.float64).tolist()
                        )
                        episode_data['belief_positions'].append(
                            np.asarray(terminal_state_phoenix[0:3], dtype=np.float64).tolist()
                        )
                    else:
                        print("⚠️  Test: Non-finite residual_xy, skipping belief update.")
            except Exception:
                belief_filter = None
                belief_vec = None

        # Compute per-step cost from executed LQR control (normalized by U_MAX, with weights)
        try:
            U_MAX = np.array([65535.0, 2.0, 2.0, 2.0], dtype=np.float64)
            wT, wR, wY = 1.0, 0.2, 0.1
            step_cost = 0.0
            if isinstance(control_outputs, list) and len(control_outputs) > 0:
                costs_list = []
                for rec in control_outputs:
                    phoenix_action = np.asarray(rec.get('control_action', control_actions), dtype=np.float64)
                    u_norm = phoenix_action / U_MAX
                    thrust_cmd = u_norm[0]
                    roll_rate  = u_norm[1]
                    pitch_rate = u_norm[2]
                    yaw_rate   = u_norm[3]
                    cost = float(wT*(thrust_cmd**2) + wR*((roll_rate**2) + (pitch_rate**2)) + wY*(yaw_rate**2))
                    costs_list.append(cost)
                if len(costs_list) > 0:
                    step_cost = float(np.mean(costs_list))
            else:
                phoenix_action = np.asarray(control_actions, dtype=np.float64)
                u_norm = phoenix_action / U_MAX
                thrust_cmd = u_norm[0]
                roll_rate  = u_norm[1]
                pitch_rate = u_norm[2]
                yaw_rate   = u_norm[3]
                step_cost = float(wT*(thrust_cmd**2) + wR*((roll_rate**2) + (pitch_rate**2)) + wY*(yaw_rate**2))
            episode_data['costs'].append(step_cost)
        except Exception:
            episode_data['costs'].append(0.0)
        
        next_state17 = terminal_state_phoenix
        pos_cur, pos_next = state17[0:3], next_state17[0:3]
        
        # Calculate reward
        reward, done = world_map.get_position_status_with_cur(pos_next, pos_cur, action, step,state_trajectory)

        print("pos_cur", pos_cur,"pos_next", pos_next)
        total_reward += reward
        step_count += 1
                
        episode_data['rewards'].append(reward)
        episode_data['total_steps'] += 1
        
        # Store position and calculate distance
        episode_data['positions'].append(state17[:3].copy())
        for i in range(len(state_trajectory)):
            episode_data['positions'].append(state_trajectory[i][:3].copy())
        current_distance = np.linalg.norm(state17[:3] - target_position)
        episode_data['distances'].append(current_distance)
        episode_data['reference_traj'].append(reference_traj[0:3,0])
        episode_data['reference_traj'].append(reference_traj[0:3,-1])
        # Check termination conditions
        if done:
            print("pos_cur",pos_cur)  # Failure threshold
            print("pos_next",pos_next)  # Failure threshold
            episode_data['positions'].append(next_state17[:3].copy())
            if(reward == 10):
                episode_data['success'] = True
            break
        
        state17 = next_state17
    # Final calculations
    episode_data['final_distance'] = current_distance
    episode_data['total_reward'] = sum(episode_data['rewards'])
    episode_data['final_position'] = next_state17[:3].copy()
    try:
        episode_data['total_cost'] = float(sum(episode_data['costs'])) if len(episode_data['costs']) > 0 else 0.0
        episode_data['mean_step_cost'] = float(np.mean(episode_data['costs'])) if len(episode_data['costs']) > 0 else 0.0
        episode_data['max_step_cost'] = float(np.max(episode_data['costs'])) if len(episode_data['costs']) > 0 else 0.0
    except Exception:
        episode_data['total_cost'] = 0.0
        episode_data['mean_step_cost'] = 0.0
        episode_data['max_step_cost'] = 0.0
    
    
    print(f"   Final position: {episode_data['final_position']}")
    print(f"   Final distance: {episode_data['final_distance']:.3f}m")
    print(f"   Total reward: {episode_data['total_reward']:.3f}")
    print(f"   Success: {'✅ YES' if episode_data['success'] else '❌ NO'}")
    if len(episode_data['qr_R_scalar']) > 0 and len(episode_data['qr_Q_diag']) > 0:
        final_r = episode_data['qr_R_scalar'][-1]
        final_q = episode_data['qr_Q_diag'][-1]
        print(f"   Final QR weights: R={final_r:.4g}, Qdiag={[round(x,3) for x in final_q]}")
    info = getattr(sim.physics, "get_disturbance_info", None)
    if callable(info):
        disturb_info = sim.physics.get_disturbance_info()
        wind_cfg = disturb_info.get("wind_config", None)
    return episode_data, total_reward, wind_cfg


def plot_episode_trajectory(episode_data: Dict[str, Any], world_map: WorldMap, 
                           target_position: np.ndarray, episode_num: int, 
                           save_plot: bool = False, output_dir: str = 'test_results', is_test: bool = False, wind_cfg: Dict[str, Any] | None = None):
    """Plot the trajectory for a single episode with world map visualization"""
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data
    positions = np.array(episode_data['positions'])
    actions = np.array(episode_data['actions'])
    distances = np.array(episode_data['distances'])
    rewards = np.array(episode_data['rewards'])
    reference_traj = np.array(episode_data.get('reference_traj', []))
    
    # Plot 1: Trajectory in world coordinates with world map overlay
    # First, create a grid to visualize the world map
    x_grid = np.linspace(0, 2.0, 51)  # 0.0 to 2.0 with 0.1m resolution
    y_grid = np.linspace(0, 2.0, 51)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Create world map visualization arrays
    safe_mask = np.zeros_like(X, dtype=bool)
    avoid_mask = np.zeros_like(X, dtype=bool)
    target_mask = np.zeros_like(X, dtype=bool)
    
    # Fill the masks based on world map data
    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            # Convert world coordinates to image coordinates
            status= world_map.check_position_status([x,y])
            if (status["status"] == "safe"):
                safe_mask[j, i] = True
            elif (status["status"] == "target"):
                target_mask[j, i] = True
            elif (status["status"] == "avoid"):
                avoid_mask[j, i] = True
                
            # img_x = x / world_map.grid_resolution
            # img_y = y / world_map.grid_resolution
            
            # # Check bounds
            # if 0 <= img_x < world_map.img_width and 0 <= img_y < world_map.img_height:
            #     img_x_int = int(img_x)
            #     img_y_int = int(img_y)
                
                # Get values from world map
                
                
                # safe_val = world_map.safe_set[img_y_int, img_x_int]
                # target_val = world_map.target_set[img_y_int, img_x_int]
                # avoid_val = world_map.avoid_set[img_y_int, img_x_int]
                
                # # Set masks
                # if safe_val > 0.5:
                #     safe_mask[j, i] = True
                # elif target_val > 0.5:
                #     target_mask[j, i] = True
                # elif avoid_val > 0.5:
                #     avoid_mask[j, i] = True
    
    # Plot world map regions with different colors
    ax1.scatter(X[safe_mask], Y[safe_mask], c='lightgreen', s=20, alpha=0.6, label='Safe Area', marker='s')
    ax1.scatter(X[target_mask], Y[target_mask], c='lightblue', s=20, alpha=0.8, label='Target Area', marker='s')
    ax1.scatter(X[avoid_mask], Y[avoid_mask], c='lightcoral', s=20, alpha=0.6, label='Avoid Area', marker='s')
    
    # Plot actual drone trajectory (unchanged style)
    ax1.plot(
        positions[:, 0],
        positions[:, 1],
        'b-',
        linewidth=3,
        label='Drone Path',
        zorder=10,
        alpha=0.6,
    )
    ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=12, label='Start', zorder=11)
    ax1.plot(target_position[0], target_position[1], 'r*', markersize=20, label='Target', zorder=11)

    # Plot reference trajectory determined by RL action (if available)
    if reference_traj.size > 0:
        try:
            if reference_traj.ndim == 1:
                reference_traj = reference_traj.reshape(-1, 3)
            ax1.plot(
                reference_traj[:, 0],
                reference_traj[:, 1],
                color='red',
                # linestyle='--',
                linewidth=2.0,
                label='Reference Trajectory',
                zorder=8,
                alpha=0.9,
            )
            ax1.scatter(
                reference_traj[:, 0],
                reference_traj[:, 1],
                c='orange',
                s=15,
                alpha=1.0,
                marker='o',
                edgecolors='k',
                linewidths=0.3,
                label='Reference Step',
                zorder=9,
            )
        except Exception:
            pass

    # Overlay belief arrows along the trajectory (requested: use (belief[0], belief[1]) and ignore belief[2])
    belief_positions = episode_data.get('belief_positions', None)
    beliefs = episode_data.get('beliefs', None)
    if belief_positions is not None and beliefs is not None and len(belief_positions) >= 2 and len(beliefs) >= 2:
        try:
            BP = np.asarray(belief_positions, dtype=np.float64)
            B = np.asarray(beliefs, dtype=np.float64)
            n = min(BP.shape[0], B.shape[0])
            BP = BP[:n]
            B = B[:n]
            # Use components 0 and 1 only
            bx = B[:, 0]
            by = B[:, 1]
            # Scale for visualization (units are "probability", not meters)
            belief_arrow_scale = 10.0
            ax1.quiver(
                BP[:, 0],
                BP[:, 1],
                belief_arrow_scale * bx,
                belief_arrow_scale * by,
                angles='xy',
                scale_units='xy',
                scale=1.0,
                color='magenta',
                alpha=0.8,
                width=0.003,
                zorder=12,
                label='Belief arrow (b0,b1)',
            )
        except Exception:
            pass
    
    # Add world map boundaries
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=2.0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axvline(x=2.0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add grid
    ax1.grid(True, alpha=0.2)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title(f'Episode {episode_num}: Drone Trajectory with World Map')
    ax1.legend(loc='upper right')
    ax1.set_aspect('equal')
    
    # Set axis limits to world map bounds
    ax1.set_xlim(-0.1, 2.1)
    ax1.set_ylim(-0.1, 2.1)
    
    # Plot 2: Performance metrics over time
    steps = range(len(episode_data['actions']))
    ax2_twin = ax2.twinx()
    
    # Plot distance and reward
    line1 = ax2.plot(steps, distances[1:], 'b-', linewidth=2, label='Distance to Target')
    line2 = ax2_twin.plot(steps, rewards, 'r-', linewidth=2, label='Reward')
    
    # Plot actions
    if len(actions) > 0:
        ax2_twin.plot(steps, actions[:, 0], 'g--', alpha=0.7, label='Action vx')
        ax2_twin.plot(steps, actions[:, 1], 'm--', alpha=0.7, label='Action vy')
    
    ax2.set_xlabel('RL Step')
    ax2.set_ylabel('Distance (m)', color='b')
    ax2_twin.set_ylabel('Reward / Action', color='r')
    ax2.set_title(f'Episode {episode_num}: Performance Metrics')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')
    
    # Add success/failure annotation
    success_text = "SUCCESS" if episode_data['success'] else "FAILURE"
    ax1.text(0.02, 0.98, success_text, transform=ax1.transAxes, 
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if episode_data['success'] else 'lightcoral'))
    
    # Add world map legend box
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', alpha=0.6, label='Safe Area'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.8, label='Target Area'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightcoral', alpha=0.6, label='Avoid Area')
    ]
    
    # Add legend box in upper left
    ax1.legend(handles=legend_elements, loc='upper left', title='World Map Regions', 
              title_fontsize=10, fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_plot:

        os.makedirs(output_dir, exist_ok=True)
        
        # Handle duplicate filenames by incrementing episode number
        base_episode_num = episode_num
        counter = 0
        if is_test==False:
            while True: 
                if counter == 0:
                    plot_filename = os.path.join(output_dir, f'trajectory_test_{base_episode_num}.png')
                else:
                    plot_filename = os.path.join(output_dir, f'trajectory_test_{base_episode_num+counter}.png')
                
                if not os.path.exists(plot_filename):
                    break
                counter += 1
        else:
            plot_filename = os.path.join(output_dir, f'trajectory_test.png')
        
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"   📊 Trajectory plot saved: {plot_filename}")
        plt.close()  # Close the figure to free memory
    else:
        # Close the figure since we're using non-interactive backend
        plt.close()


def plot_belief_history(episode_data: Dict[str, Any], episode_num: int,
                        save_plot: bool = False, output_dir: str = 'test_results', is_test: bool = False):
    """Plot Bayes belief b_k(z) over time (debug visualization)."""
    beliefs = episode_data.get('beliefs', None)
    if not save_plot or beliefs is None or len(beliefs) < 2:
        return

    B = np.asarray(beliefs, dtype=np.float64)
    if B.ndim != 2:
        return

    residuals = episode_data.get('residuals', None)
    R = None
    if residuals is not None and len(residuals) > 0:
        R = np.asarray(residuals, dtype=np.float64)
        if R.ndim == 2 and R.shape[1] >= 2:
            R = np.linalg.norm(R[:, 0:2], axis=1)
        else:
            R = None

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    steps = np.arange(B.shape[0])
    for k in range(B.shape[1]):
        ax.plot(steps, B[:, k], linewidth=2, label=f"b(z={k})")

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Step k")
    ax.set_ylabel("Belief b_k(z)")
    ax.set_title(f"Belief update over time (episode {episode_num})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=min(3, B.shape[1]))

    if R is not None:
        ax2 = ax.twinx()
        # residual r_k corresponds to transition k -> k+1, so align at steps 1..T
        ax2.plot(np.arange(1, 1 + len(R)), R, color="black", alpha=0.35, linewidth=1)
        ax2.set_ylabel("Residual norm ||r_k||")

    os.makedirs(output_dir, exist_ok=True)
    if is_test:
        filepath = os.path.join(output_dir, "belief_test.png")
    else:
        base = os.path.join(output_dir, f"belief_{episode_num}.png")
        filepath = base
        counter = 1
        while os.path.exists(filepath):
            filepath = os.path.join(output_dir, f"belief_{episode_num}_{counter}.png")
            counter += 1

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"   📊 Belief plot saved: {filepath}")
    plt.close(fig)

def state_to_input(state17, target_position, world_map):
    drone_pos_xy = state17[:2]
    local_grid = world_map.encode_local_grid(drone_pos_xy)
    augmented_obs = np.concatenate([state17, target_position[:2]-state17[0:2], local_grid], axis=0).astype(np.float32)
    return augmented_obs

def plot_action_field_together(agent: TestRLAgent, world_map: WorldMap, 
                           target_position: np.ndarray, episode_num: int, 
                           save_plot: bool = False, output_dir: str = 'test_results', sim: PhoenixSimulator = None, is_test: bool = False,
                           episode_data: Dict[str, Any] | None = None, wind_cfg: Dict[str, Any] | None = None):
    """Plot the action field and state values combined in a single plot with world map visualization"""
    # Create figure with single subplot - smaller figure size to reduce empty space
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # First, create a grid to visualize the world map
    x_grid = np.linspace(0, 2.0, 11)  # 0.0 to 2.0 with 0.1m resolution
    y_grid = np.linspace(0, 2.0, 11)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Create world map visualization arrays
    safe_mask = np.zeros_like(X, dtype=bool)
    avoid_mask = np.zeros_like(X, dtype=bool)
    target_mask = np.zeros_like(X, dtype=bool)
    
    # Arrays to store actions, values, and mean disturbance for plotting
    actions_x = np.zeros_like(X)
    actions_y = np.zeros_like(X)
    state_values = np.zeros_like(X)
    noise_mean_x = np.zeros_like(X)
    noise_mean_y = np.zeros_like(X)
    
    # Fill the masks based on world map data and collect actions/values
    policy_obs_dim = int(getattr(agent, "obs_dim", 44))
    belief_for_field = None
    if policy_obs_dim > 44:
        # Use episode belief if available; otherwise default prior.
        last_belief = None
        if episode_data is not None and isinstance(episode_data.get("beliefs", None), list) and len(episode_data["beliefs"]) > 0:
            last_belief = np.asarray(episode_data["beliefs"][-1], dtype=np.float32).reshape(-1)
        else:
            if policy_obs_dim == 47:
                last_belief = np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
            elif policy_obs_dim == 46:
                last_belief = np.zeros(2, dtype=np.float32)
        if last_belief is not None:
            if policy_obs_dim == 47:
                belief_for_field = last_belief[:3]
            elif policy_obs_dim == 46:
                belief_for_field = last_belief[:2]

    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            # Convert world coordinates to image coordinates
            # Create a mock 17D state for plotting (only position matters for action field)
            
            sim.reset(np.array([x, y, 1.0]))
            state = state_to_input(sim.get_state(), target_position, world_map)
            if belief_for_field is not None:
                state = np.concatenate([state, belief_for_field], axis=0).astype(np.float32)
            status= world_map.check_position_status([x,y])
            if (status["status"] == "safe"):
                safe_mask[j, i] = True
            elif (status["status"] == "target"):
                target_mask[j, i] = True
            elif (status["status"] == "avoid"):
                avoid_mask[j, i] = True
            with torch.no_grad():
                normalized_state = simple_minmax_normalize(state)
                obs_tensor = torch.FloatTensor(normalized_state).unsqueeze(0)
                # Get the policy distribution and use the mean (most probable action)
                dist = agent.actor_critic.pi.dist(obs_tensor)
                state_value = agent.actor_critic.v(obs_tensor).cpu().numpy().squeeze()
                # state_value = agent.actor_critic.entropy(obs_tensor).cpu().numpy().squeeze()
                # policy_entropy = dist.entropy().sum(dim=-1).cpu().numpy().squeeze()
                action = dist.mean.cpu().numpy().squeeze().astype(np.float32)
                
                # Store actions and values
                actions_x[j, i] = action[0]
                actions_y[j, i] = action[1]
                state_values[j, i] = state_value
                # If the underlying physics has a wind configuration, reuse its mean wind force for plotting
                
                        
                            # Same region split as in physics: X<0.5, 0.5<=X<1.5, X>=1.5
                if x < 0.5:
                    region_idx = 0
                elif x < 1.5:
                    region_idx = 1
                else:
                    region_idx = 2
                mean_force_xy = np.asarray(wind_cfg[region_idx].get("mean_force_xy", [0.0, 0.0]), dtype=np.float64)
                noise_mean_x[j, i] = mean_force_xy[0]
                noise_mean_y[j, i] = mean_force_xy[1]
                # print(f"~~~~~~~~~~Mean force xy~~~~~~~~~~~: {mean_force_xy}")

    # Plot 1: Value function as background contour plot
    masked_values = state_values
    contour = ax.contourf(X, Y, masked_values, levels=20, cmap='viridis', alpha=0.8)
    
    # Add colorbar for value function (positioned to the right with more padding)
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8, pad=0.15)
    cbar.set_label('State Value', rotation=270, labelpad=25, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Plot 2: World map regions with different colors (semi-transparent overlay)
    ax.scatter(X[safe_mask], Y[safe_mask], c='lightgreen', s=40, alpha=0.5, label='Safe Area', marker='s', edgecolors='darkgreen', linewidth=0.5)
    ax.scatter(X[target_mask], Y[target_mask], c='lightblue', s=40, alpha=0.7, label='Target Area', marker='s', edgecolors='darkblue', linewidth=0.5)
    ax.scatter(X[avoid_mask], Y[avoid_mask], c='lightcoral', s=40, alpha=0.5, label='Avoid Area', marker='s', edgecolors='darkred', linewidth=0.5)
    
    # Plot 3: Action field arrows and mean disturbance arrows overlaid on top
    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            if safe_mask[j, i]:  # Only draw arrows in safe areas
                action_x = actions_x[j, i]
                action_y = actions_y[j, i]
                # Scale arrow length for better visibility
                arrow_scale = 0.4
                ax.arrow(
                    x,
                    y,
                    arrow_scale * action_x * ACTION_SCALE,
                    arrow_scale * action_y * ACTION_SCALE,
                    head_width=0.04,
                    head_length=0.04,
                    fc='red',
                    ec='darkred',
                    alpha=0.9,
                    linewidth=2.0,
                )
                # Draw mean disturbance arrow if available from physics configuration
                mean_x = noise_mean_x[j, i]
                mean_y = noise_mean_y[j, i]
                if not (mean_x == 0.0 and mean_y == 0.0):
                    noise_scale = 1
                    ax.arrow(
                        x,
                        y,
                        noise_scale * mean_x,
                        noise_scale * mean_y,
                        head_width=0.04,
                        head_length=0.04,
                        fc='cyan',
                        ec='darkcyan',
                        alpha=0.9,
                        linewidth=2.0,
                    )
    
    # Add world map boundaries (cleaner style)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    ax.axhline(y=2.0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    ax.axvline(x=2.0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    
    # Remove grid lines
    ax.grid(False)
    
    # Styling
    ax.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
    ax.set_title(f'Episode {episode_num}: Action Field + Value Function', fontsize=16, fontweight='bold', pad=20)
    ax.set_aspect('equal')
    
    # Set axis limits to world map bounds - remove extra space
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.1, 2.1)
    
    # Set custom tick spacing to 0.2m
    ax.set_xticks(np.arange(0, 2.1, 0.2))
    ax.set_yticks(np.arange(0, 2.1, 0.2))
    
    # Style the axes
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    
    # Create legend elements
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', alpha=0.6, edgecolor='darkgreen', linewidth=1, label='Safe Area'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.8, edgecolor='darkblue', linewidth=1, label='Target Area'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightcoral', alpha=0.6, edgecolor='darkred', linewidth=1, label='Avoid Area'),
        plt.Line2D([0], [0], color='red', linewidth=3, label='Action Field'),
        plt.Line2D([0], [0], color='cyan', linewidth=3, label='Mean Disturbance'),
    ]
    
    # Position legend OUTSIDE the plot area (to the right)
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), 
              title='Legend', title_fontsize=12, fontsize=11, framealpha=0.95, 
              fancybox=True, shadow=True)
    
    # Adjust layout to make room for external legend and colorbar
    plt.tight_layout()
    
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle duplicate filenames by incrementing episode number
        base_episode_num = episode_num
        counter = 0
        if is_test==False:
            while True:
                if counter == 0:
                    png_filename = os.path.join(output_dir, f'action_field_combined_{base_episode_num}.png')
                    pdf_filename = os.path.join(output_dir, f'action_field_combined_{base_episode_num}.pdf')
                else:
                    png_filename = os.path.join(output_dir, f'action_field_combined_{base_episode_num+counter}.png')
                    pdf_filename = os.path.join(output_dir, f'action_field_combined_{base_episode_num+counter}.pdf')
                
                if not os.path.exists(png_filename) and not os.path.exists(pdf_filename):
                    break
                counter += 1
        else:
            png_filename = os.path.join(output_dir, f'action_field_combined_test.png')
            pdf_filename = os.path.join(output_dir, f'action_field_combined_test.pdf')
        
        # Save as PNG
        plt.savefig(png_filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"   📊 Combined action field + value function plot saved: {png_filename}")
        
        # Save as PDF (vector format)
        plt.savefig(pdf_filename, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"   📊 Combined action field + value function plot saved: {pdf_filename}")
        
        plt.close()  # Close the figure to free memory
    else:
        # Close the figure since we're using non-interactive backend
        plt.close()

def plot_action_field(agent: TestRLAgent, world_map: WorldMap, 
                           target_position: np.ndarray, episode_num: int, 
                           save_plot: bool = False, output_dir: str = 'test_results', sim: PhoenixSimulator = None, is_test: bool = False,
                           episode_data: Dict[str, Any] | None = None, wind_cfg: Dict[str, Any] | None = None):
    """Plot the action field and state values with world map visualization"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Action field in world coordinates with world map overlay
    # First, create a grid to visualize the world map
    x_grid = np.linspace(0, 2.0, 10)  # 0.0 to 2.0 with 0.1m resolution
    y_grid = np.linspace(0, 2.0, 10)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Create world map visualization arrays
    safe_mask = np.zeros_like(X, dtype=bool)
    avoid_mask = np.zeros_like(X, dtype=bool)
    target_mask = np.zeros_like(X, dtype=bool)
    
    # Arrays to store actions, values, and mean disturbance for plotting
    actions_x = np.zeros_like(X)
    actions_y = np.zeros_like(X)
    state_values = np.zeros_like(X)
    noise_mean_x = np.zeros_like(X)
    noise_mean_y = np.zeros_like(X)
    
    # Fill the masks based on world map data and collect actions/values
    policy_obs_dim = int(getattr(agent, "obs_dim", 44))
    belief_for_field = None
    if policy_obs_dim > 44:
        # Use episode belief if available; otherwise default prior.
        last_belief = None
        last_belief = np.zeros(2, dtype=np.float32)
        belief_for_field = last_belief[:2]
        # if episode_data is not None and isinstance(episode_data.get("beliefs", None), list) and len(episode_data["beliefs"]) > 0:
        #     last_belief = np.asarray(episode_data["beliefs"][-1], dtype=np.float32).reshape(-1)
        # else:
        #     if policy_obs_dim == 46:
        #         last_belief = np.zeros(2, dtype=np.float32)
        # if last_belief is not None:
        #     if policy_obs_dim == 46:
        #         belief_for_field = last_belief[:2]
       
    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            # Convert world coordinates to image coordinates
            # Create a mock 17D state for plotting (only position matters for action field)
            
            sim.reset(np.array([x, y, 1.0]))
            state = state_to_input(sim.get_state(), target_position, world_map)
            if belief_for_field is not None:
                state = np.concatenate([state, belief_for_field], axis=0).astype(np.float32)

            with torch.no_grad():
                normalized_state = simple_minmax_normalize(state)
                obs_tensor = torch.FloatTensor(normalized_state).unsqueeze(0)
                # Get the policy distribution and use the mean (most probable action)
                dist = agent.actor_critic.pi.dist(obs_tensor)
                state_value = agent.actor_critic.v(obs_tensor).cpu().numpy().squeeze()
                action = dist.mean.cpu().numpy().squeeze().astype(np.float32)
                
                # Store actions and values
                actions_x[j, i] = action[0]
                actions_y[j, i] = action[1]
                state_values[j, i] = state_value
                # If the underlying physics has a wind configuration, reuse its mean wind force for plotting
                if wind_cfg is not None:
                    # Same region split as in physics: X<0.5, 0.5<=X<1.5, X>=1.5
                    if x < 0.5:
                        region_idx = 0
                    elif x < 1.5:
                        region_idx = 1
                    else:
                        region_idx = 2
                    mean_force_xy = np.asarray(wind_cfg[region_idx].get("mean_force_xy", [0.0, 0.0]), dtype=np.float64)
                    # print(f"~~~~~~~~~~LQR TEST Mean Wind force xy~~~~~~~~~~~: {mean_force_xy}")
                    noise_mean_x[j, i] = mean_force_xy[0]
                    noise_mean_y[j, i] = mean_force_xy[1]    
           
            img_x = x / world_map.grid_resolution
            img_y = y / world_map.grid_resolution
            # Check bounds
            if 0 <= img_x < world_map.img_width and 0 <= img_y < world_map.img_height:
                img_x_int = int(img_x)
                img_y_int = int(img_y)
                
                # Get values from world map
                safe_val = world_map.safe_set[img_y_int, img_x_int]
                target_val = world_map.target_set[img_y_int, img_x_int]
                avoid_val = world_map.avoid_set[img_y_int, img_x_int]
                
                # Set masks
                if safe_val > 0.5:
                    safe_mask[j, i] = True
                elif target_val > 0.5:
                    target_mask[j, i] = True
                elif avoid_val > 0.5:
                    avoid_mask[j, i] = True
             # Draw arrow from (x,y) to (x+0.5*action[0], y+0.5*action[1])
            if safe_mask[j, i]:
                ax1.arrow(
                    x,
                    y,
                    0.5 * action[0] * ACTION_SCALE,
                    0.5 * action[1] * ACTION_SCALE,
                    head_width=0.02,
                    head_length=0.02,
                    fc='red',
                    ec='red',
                    alpha=0.7,
                )
                # Draw mean disturbance arrow if available from physics configuration
                mean_x = noise_mean_x[j, i]
                mean_y = noise_mean_y[j, i]
                if not (mean_x == 0.0 and mean_y == 0.0):
                    noise_scale = 1
                    ax1.arrow(
                        x,
                        y,
                        noise_scale * mean_x,
                        noise_scale * mean_y,
                        head_width=0.02,
                        head_length=0.02,
                        fc='cyan',
                        ec='cyan',
                        alpha=0.9,
                    )

    # Overlay belief arrow(s) from episode on action-field plot (requested: (belief[0], belief[1]) only)
    if episode_data is not None:
        belief_positions = episode_data.get('belief_positions', None)
        beliefs = episode_data.get('beliefs', None)
        if belief_positions is not None and beliefs is not None and len(belief_positions) >= 2 and len(beliefs) >= 2:
            try:
                BP = np.asarray(belief_positions, dtype=np.float64)
                B = np.asarray(beliefs, dtype=np.float64)
                n = min(BP.shape[0], B.shape[0])
                BP = BP[:n]
                B = B[:n]
                bx = B[:, 0]
                by = B[:, 1]
                belief_arrow_scale = 10.0
                ax1.quiver(
                    BP[:, 0],
                    BP[:, 1],
                    belief_arrow_scale * bx,
                    belief_arrow_scale * by,
                    angles='xy',
                    scale_units='xy',
                    scale=1.0,
                    color='magenta',
                    alpha=0.8,
                    width=0.003,
                    zorder=20,
                )
            except Exception:
                pass
    
    # Plot 1: Action field
    # Plot world map regions with different colors
    ax1.scatter(X[safe_mask], Y[safe_mask], c='lightgreen', s=20, alpha=0.6, label='Safe Area', marker='s')
    ax1.scatter(X[target_mask], Y[target_mask], c='lightblue', s=20, alpha=0.8, label='Target Area', marker='s')
    ax1.scatter(X[avoid_mask], Y[avoid_mask], c='lightcoral', s=20, alpha=0.6, label='Avoid Area', marker='s')
    
    # Add world map boundaries
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=2.0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axvline(x=2.0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add grid
    ax1.grid(True, alpha=0.2)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title(f'Episode {episode_num}: Action Field')
    ax1.legend(loc='upper right')
    ax1.set_aspect('equal')
    
    # Set axis limits to world map bounds
    ax1.set_xlim(-0.1, 2.1)
    ax1.set_ylim(-0.1, 2.1)
    
    # Plot 2: State values with color coding
    # Create a masked array to only show values in safe areas
    # masked_values = np.ma.masked_where(~safe_mask, state_values)
    masked_values = state_values
    
    # Create contour plot for state values
    contour = ax2.contourf(X, Y, masked_values, levels=20, cmap='viridis', alpha=0.8)
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax2)
    cbar.set_label('State Value', rotation=270, labelpad=15)
    
    # Plot world map regions with different colors (transparent overlay)
    ax2.scatter(X[safe_mask], Y[safe_mask], c='lightgreen', s=20, alpha=0.3, label='Safe Area', marker='s')
    ax2.scatter(X[target_mask], Y[target_mask], c='lightblue', s=20, alpha=0.5, label='Target Area', marker='s')
    ax2.scatter(X[avoid_mask], Y[avoid_mask], c='lightcoral', s=20, alpha=0.3, label='Avoid Area', marker='s')
    
    # Add world map boundaries
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=2.0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axvline(x=2.0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add grid
    ax2.grid(True, alpha=0.2)
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title(f'Episode {episode_num}: State Values')
    ax2.legend(loc='upper right')
    ax2.set_aspect('equal')
    
    # Set axis limits to world map bounds
    ax2.set_xlim(-0.1, 2.1)
    ax2.set_ylim(-0.1, 2.1)
    
    # Add world map legend box to the first subplot
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', alpha=0.6, label='Safe Area'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.8, label='Target Area'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightcoral', alpha=0.6, label='Avoid Area'),
        plt.Line2D([0], [0], color='red', linewidth=3, label='Action Field'),
        plt.Line2D([0], [0], color='cyan', linewidth=3, label='Mean Disturbance'),
    ]
    
    # Add legend box in upper left of first subplot
    ax1.legend(handles=legend_elements, loc='upper left', title='World Map Regions', 
              title_fontsize=10, fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle duplicate filenames by incrementing episode number
        base_episode_num = episode_num
        counter = 0
        if is_test==False:
            while True:
                if counter == 0:
                    plot_filename = os.path.join(output_dir, f'action_field_{base_episode_num}.png')
                else:
                    plot_filename = os.path.join(output_dir, f'action_field_{base_episode_num+counter}.png')
                
                if not os.path.exists(plot_filename):
                    break
                counter += 1
        else:
            plot_filename = os.path.join(output_dir, f'action_field_test.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"   📊 Action field plot saved: {plot_filename}")
        plt.close()  # Close the figure to free memory
    else:
        # Close the figure since we're using non-interactive backend
        plt.close()


def run_test_suite(agent: RLAgent, model_path: str, num_episodes: int, render: bool, 
                   save_plots: bool, output_dir: str, world_map: WorldMap, sim: PhoenixSimulator, is_test: bool = False,
                   qr_agent: RLAgent | None = None, qr_model_path: str | None = None):
    """Run a complete test suite"""
    # print("🧪 Starting RL+SMPC Model Test Suite")
    # print("=" * 60)
    
    # # Print configuration information
    # print(f"🎯 Configuration:")
    # print(f"   - Start position: {START_POSITION}")
    # print(f"   - Target position: [0.0, 2.0, 1.0] (goal center)")
    # print(f"   - Episode length: {EPISODE_LENGTH} steps")
    # print(f"   - Model type: 44D (Map version) - using existing RLAgent")
    # print(f"   - World map: {WORLD_MAP_NAME}")
    # print("=" * 60)
    
    # Load the trained agent
    
    
    # Initialize world map and get target position
    
    target_position = world_map.goal_center.copy()
    
    # print(f"🗺️  World map loaded: {world_map.world_name}")
    # print(f"   Target position: {target_position}")
    # print("=" * 60)
    
    # Run test episodes
    all_episodes = []
    success_count = 0
    
    for episode in range(num_episodes):
        episode_data, total_reward, wind_cfg = run_test_episode(
            agent,
            world_map,
            target_position,
            render,
            episode + 1,
            qr_agent=qr_agent,
        )
        all_episodes.append(episode_data)
        
        if episode_data['success']:
            success_count += 1
        
        # Plot trajectory for this episode
        plot_episode_trajectory(episode_data, world_map, target_position, episode + 1, save_plots, output_dir, is_test,wind_cfg)
        plot_action_field(
            agent,
            world_map,
            target_position,
            episode + 1,
            save_plots,
            output_dir,
            sim,
            is_test,
            episode_data=episode_data,
            wind_cfg=wind_cfg,
        )
    
    # Calculate overall statistics
    final_distances = [ep['final_distance'] for ep in all_episodes]
    total_rewards = [ep['total_reward'] for ep in all_episodes]
    steps_completed = [ep['total_steps'] for ep in all_episodes]
    mean_step_costs = [float(np.mean(ep['costs'])) if ('costs' in ep and len(ep['costs']) > 0) else 0.0 for ep in all_episodes]
    total_costs = [float(sum(ep['costs'])) if ('costs' in ep and len(ep['costs']) > 0) else 0.0 for ep in all_episodes]
    max_step_costs = [float(np.max(ep['costs'])) if ('costs' in ep and len(ep['costs']) > 0) else 0.0 for ep in all_episodes]
    
    success_rate = (success_count / num_episodes) * 100
    avg_final_distance = np.mean(final_distances)
    avg_total_reward = np.mean(total_rewards)
    avg_steps_completed = np.mean(steps_completed)
    avg_mean_step_cost = float(np.mean(mean_step_costs)) if len(mean_step_costs) > 0 else 0.0
    avg_total_cost = float(np.mean(total_costs)) if len(total_costs) > 0 else 0.0
    avg_max_step_cost = float(np.mean(max_step_costs)) if len(max_step_costs) > 0 else 0.0
    
    # Print overall results
    print("\n" + "=" * 60)
    print("📈 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model: {model_path}")
    if qr_model_path:
        print(f"Model (QR): {qr_model_path}")
    print(f"Model type: 44D (Map version) - using existing RLAgent")
    print(f"Episodes tested: {num_episodes}")
    print(f"Success rate: {success_rate:.1f}% ({success_count}/{num_episodes})")
    print(f"Average final distance: {avg_final_distance:.3f}m")
    print(f"Average total reward: {avg_total_reward:.3f}")
    print(f"Average steps completed: {avg_steps_completed:.1f}")
    print("=" * 60)
    
    # Save results to file
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        results_filename = os.path.join(output_dir, f'test_results_summary.json')
        
        results_summary = {
            'model_path': model_path,
            'model_type': '44D (Map version) - using existing RLAgent',
            'num_episodes': num_episodes,
            'success_rate': success_rate,
            'avg_final_distance': avg_final_distance,
            'avg_total_reward': avg_total_reward,
            'avg_steps_completed': avg_steps_completed,
            'avg_mean_step_cost': avg_mean_step_cost,
            'avg_total_cost': avg_total_cost,
            'avg_max_step_cost': avg_max_step_cost,
            'episodes': all_episodes
        }
        
        with open(results_filename, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print(f"📁 Results saved to: {results_filename}")
    
    return all_episodes


def run_test_for_ICRA(agent: RLAgent, model_path: str, num_episodes: int, render: bool, 
                   save_plots: bool, output_dir: str, world_map: WorldMap, sim: PhoenixSimulator, is_test: bool = False,
                   qr_agent: RLAgent | None = None, qr_model_path: str | None = None):
    """Run a complete test suite"""
    # print("🧪 Starting RL+SMPC Model Test Suite")
    # print("=" * 60)
    
    # # Print configuration information
    # print(f"🎯 Configuration:")
    # print(f"   - Start position: {START_POSITION}")
    # print(f"   - Target position: [0.0, 2.0, 1.0] (goal center)")
    # print(f"   - Episode length: {EPISODE_LENGTH} steps")
    # print(f"   - Model type: 44D (Map version) - using existing RLAgent")
    # print(f"   - World map: {WORLD_MAP_NAME}")
    # print("=" * 60)
    
    # Load the trained agent
    
    
    # Initialize world map and get target position
    
    target_position = world_map.goal_center.copy()
    
    # print(f"🗺️  World map loaded: {world_map.world_name}")
    # print(f"   Target position: {target_position}")
    # print("=" * 60)
    
    # Run test episodes
    all_episodes = []
    success_count = 0
    position_datasets = []
    reference_traj_datasets = []
    for episode in range(num_episodes):
        episode_data, total_reward = run_test_episode(
            agent,
            world_map,
            target_position,
            render,
            episode + 1,
            qr_agent=qr_agent,
        )
        all_episodes.append(episode_data)
        
        if episode_data['success']:
            success_count += 1
        position_data=np.array(episode_data['positions'])
        position_datasets.append(position_data)
        reference_traj=np.array(episode_data['reference_traj'])
        reference_traj_datasets.append(reference_traj)
        # Plot trajectory for this episode
        plot_episode_trajectory(episode_data, world_map, target_position, episode + 1, save_plots, output_dir, is_test)
        plot_action_field_together(
            agent,
            world_map,
            target_position,
            episode + 1,
            save_plots,
            output_dir,
            sim,
            is_test,
            episode_data=episode_data,
        )
    
    # Calculate overall statistics
    final_distances = [ep['final_distance'] for ep in all_episodes]
    total_rewards = [ep['total_reward'] for ep in all_episodes]
    steps_completed = [ep['total_steps'] for ep in all_episodes]
    
    success_rate = (success_count / num_episodes) * 100
    avg_final_distance = np.mean(final_distances)
    avg_total_reward = np.mean(total_rewards)
    avg_steps_completed = np.mean(steps_completed)
    
    # Print overall results
    print("\n" + "=" * 60)
    print("📈 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model: {model_path}")
    if qr_model_path:
        print(f"Model (QR): {qr_model_path}")
    print(f"Model type: 44D (Map version) - using existing RLAgent")
    print(f"Episodes tested: {num_episodes}")
    print(f"Success rate: {success_rate:.1f}% ({success_count}/{num_episodes})")
    print(f"Average final distance: {avg_final_distance:.3f}m")
    print(f"Average total reward: {avg_total_reward:.3f}")
    print(f"Average steps completed: {avg_steps_completed:.1f}")
    print("=" * 60)
    
    # Save results to file
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        results_filename = os.path.join(output_dir, f'test_results_summary.json')
        
        results_summary = {
            'model_path': model_path,
            'model_type': '44D (Map version) - using existing RLAgent',
            'num_episodes': num_episodes,
            'success_rate': success_rate,
            'avg_final_distance': avg_final_distance,
            'avg_total_reward': avg_total_reward,
            'avg_steps_completed': avg_steps_completed,
            'episodes': all_episodes
        }
        
        with open(results_filename, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print(f"📁 Results saved to: {results_filename}")
        
        # Save per-episode position data with a unique datetime_RLMPC name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f'position_data_{timestamp}_RLMPC'
        os.makedirs(output_dir, exist_ok=True)
        for idx, data in enumerate(position_datasets, start=1):
            ref_data = reference_traj_datasets[idx-1] if idx-1 < len(reference_traj_datasets) else np.empty((0, 3), dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(-1, 3)
            if ref_data.ndim == 1:
                ref_data = ref_data.reshape(-1, 3)
            max_len = max(len(data), len(ref_data))
            pad_pos = np.full((max_len, 3), np.nan, dtype=np.float32)
            pad_ref = np.full((max_len, 3), np.nan, dtype=np.float32)
            if len(data) > 0:
                pad_pos[:len(data), :] = data
            if ref_data.size > 0:
                pad_ref[:len(ref_data), :] = ref_data
            combined = np.concatenate((pad_pos, pad_ref), axis=1)
            header = 'x,y,z,ref_x,ref_y,ref_z'
            csv_path = os.path.join(output_dir, f'{base_name}_episode_{idx}.csv')
            np.savetxt(csv_path, combined, delimiter=',', header=header, comments='')
        print(f"📁 Position+reference data saved as CSV files with base: {base_name}_episode_*.csv")
    
    return all_episodes


def main():
    """Main entry point"""
    # Show debug mode status
    if DEBUG_MODE:
        print(f"🔧 DEBUG MODE: Using hardcoded model path: {MODEL_PATH_DEBUG}")
        print("   To use command line arguments, set DEBUG_MODE = False")
        print("")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found: {args.model_path}")
        if DEBUG_MODE:
            print(f"   Debug path: {MODEL_PATH_DEBUG}")
            print(f"   Current working directory: {os.getcwd()}")
            print(f"   Available files in current directory:")
            for f in os.listdir('.'):
                if f.endswith('.pth') or 'trained_model' in f:
                    print(f"     - {f}")
        sys.exit(1)
    
    # Create output directory if saving plots
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
    agent = TestRLAgent(args.model_path)
    world_map = WorldMap(world_name=WORLD_MAP_NAME)
    sim = PhoenixSimulator(control_mode='attitude_rate', disturbance_enabled=ENABLE_POSITION_DISTURBANCE, disturbance_strength=POSITION_DISTURBANCE_STRENGTH)
    # Try to load companion QR model for testing (optional)
    qr_agent = RLAgent(state_dim=44, action_dim=10)
    qr_model_path = None
    try:
        if args.qr_model_path is not None:
            # If user provided a QR path, prefer it
            if os.path.exists(args.qr_model_path):
                qr_model_path = args.qr_model_path
            else:
                print(f"⚠️  Provided --qr-model-path not found: {args.qr_model_path}. Falling back to auto-detect.")
        if qr_model_path is None:
            base, ext = os.path.splitext(args.model_path)
            candidate = f"{base}_qr{ext}"
            if os.path.exists(candidate):
                qr_model_path = candidate
            else:
                dirname = os.path.dirname(args.model_path)
                for f in os.listdir(dirname):
                    if f.endswith(ext) and f"_qr" in f and os.path.basename(base) in f:
                        qr_model_path = os.path.join(dirname, f)
                        break
        if qr_model_path is not None:
            qr_state = torch.load(qr_model_path, map_location='cpu', weights_only=False)
            qr_agent.actor_critic.load_state_dict(qr_state)
        else:
            qr_agent = None
    except Exception:
        qr_agent = None

    # Run the test suite
    results = run_test_for_ICRA(
        agent=agent.agent,
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        render=args.render,
        save_plots=args.save_plots,
        output_dir=args.output_dir,
        world_map=world_map,
        sim=sim,
        is_test=True,
        qr_agent=qr_agent,
        qr_model_path=qr_model_path
    )
    
    print("\n✅ Test suite completed successfully!")


if __name__ == "__main__":
    main() 