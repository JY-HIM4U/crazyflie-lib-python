#!/usr/bin/env python3
"""
Simplified Parallel Training for RL+SMPC Training Pipeline

This version uses an improved workflow to handle parallel data collection
by having workers return data directly to the main process, eliminating
the need for file-based I/O.
"""

import os
import time
import numpy as np
import torch
import torch.multiprocessing as mp
import pickle
from typing import List, Dict, Any

from RL_smpc_config import (
    EPISODE_LENGTH, MAX_ITERATIONS, CKPT_EVERY, DEFAULT_NUM_WORKERS, 
    DEFAULT_SAVE_INTERVAL, START_POSITION, MODEL_SAVE_DIR, 
    WORLD_MAP_NAME, RL_DECISION_INTERVAL, MPC_PLANNING_INTERVAL,
    ENABLE_POSITION_DISTURBANCE, POSITION_DISTURBANCE_STRENGTH,
    ENABLE_WIND_DISTURBANCE, WIND_VERBOSE,
    WIND_MEAN_FRAC_MIN, WIND_MEAN_FRAC_MAX,
    WIND_SIGMA_FRAC_MIN, WIND_SIGMA_FRAC_MAX,
)
from RL_smpc_world_map import WorldMap
from RL_smpc_simulation import PhoenixSimulator, StateConverter, TrajectoryGenerator
from RL_smpc_smpc_controller import SMPCController
from RL_LQR_controller import LQRController
from RL_LQR_agent import RLAgent
from RL_LQR_test import run_test_suite


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


def state_to_input(state17, target_position, world_map):
    drone_pos_xy = state17[:2]
    local_grid = world_map.encode_local_grid(drone_pos_xy)
    augmented_obs = np.concatenate([state17, target_position[:2]-state17[0:2], local_grid], axis=0).astype(np.float32)
    return augmented_obs

def worker_collect_episodes(
    worker_id: int,
    num_episodes: int,
    world_map,
    target_position,
    policy_state_dict_action,
    disturbance_enabled: bool,
    disturbance_strength: float,
    wind_enabled: bool,
    wind_verbose: bool,
    wind_mean_frac_min: float,
    wind_mean_frac_max: float,
    wind_sigma_frac_min: float,
    wind_sigma_frac_max: float,
    belief_enabled: bool,
) -> List[List[Dict[str, Any]]]:
    """
    Worker process: runs episodes independently and returns a list of episodes.
    
    Returns:
        List[List[Dict[str, Any]]]: A list of episodes, where each episode is a list of transitions.
    """
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        # 🔑 FIXED: Use time-based seeds for true randomness (within PyTorch limits)
        worker_seed = (int(time.time() * 1000) + worker_id * 1000) % (2**32 - 1)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(worker_seed)
            torch.cuda.manual_seed_all(worker_seed)
        
        # print(f"🔑 Worker {worker_id}: Set random seed to {worker_seed}")
        
        # 🔑 CRITICAL FIX: Create simulation instance in each worker
        # PyBullet objects can't be pickled and sent between processes
        
        sim = PhoenixSimulator(
            control_mode='attitude_rate',
            disturbance_enabled=disturbance_enabled,
            disturbance_strength=disturbance_strength,
            wind_enabled=wind_enabled,
            wind_verbose=wind_verbose,
            wind_mean_frac_min=wind_mean_frac_min,
            wind_mean_frac_max=wind_mean_frac_max,
            wind_sigma_frac_min=wind_sigma_frac_min,
            wind_sigma_frac_max=wind_sigma_frac_max,
        )
        # ctrl = SMPCController(phoenix_sim=sim)
        ctrl = LQRController(phoenix_sim=sim)
        conv = StateConverter()
        trajgen = TrajectoryGenerator(dt=MPC_PLANNING_INTERVAL)  # MPC_PLANNING_INTERVAL
        # We don't need a full SMPC controller here, just a trajectory generator
        # from RL_smpc_simulation import TrajectoryGenerator
        # trajgen = TrajectoryGenerator(dt=MPC_PLANNING_INTERVAL)
        
        obs_dim = 46 if belief_enabled else 44
        local_agent = RLAgent(state_dim=obs_dim, action_dim=2, obs_dim=obs_dim)
        local_agent.actor_critic.load_state_dict(policy_state_dict_action)

        # Second agent outputs 10 dims: 9 for diag(Q), 1 for scalar R
        # agent_QR = RLAgent(state_dim=44, action_dim=10)
        # agent_QR.actor_critic.load_state_dict(policy_state_dict_qr)
        
        # Local list to hold all collected episodes
        all_collected_episodes = []
        
        for episode in range(num_episodes):
            # 🔑 FIXED: Use truly random episode seeds (within PyTorch limits)
            episode_seed = (worker_seed + episode * 100 + int(time.time() * 1000)) % (2**32 - 1)
            np.random.seed(episode_seed)
            
            while(True):
                x_random = np.random.uniform(0.1, 1.9)
                y_random = np.random.uniform(0.1, 1.9)
                initial_pos = np.array([x_random, y_random, 1.0])
                # print(f" Worker {worker_id} Episode {episode}: Initial position: {initial_pos}")
                if world_map.check_position_status(initial_pos)['status'] == 'safe':
                    break
            
            sim.reset(initial_pos)

            state17 = sim.get_state()
            if state17 is None or len(state17) != 17:
                print(f"❌ Worker {worker_id}: Invalid initial state in episode {episode}")
                continue
            
            belief_filter = None
            belief_vec = None
            if belief_enabled:
                try:
                    # Initialize 2D disturbance Kalman filter d ~ N(mu, Sigma)
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

            episode_transitions = []
            initial_pos = state17[:3]
            initial_height = initial_pos[2]
            for step in range(EPISODE_LENGTH):
                # Kalman predict step for disturbance belief (prior for this decision)
                if belief_filter is not None:
                    try:
                        belief_filter.predict()
                        belief_vec = belief_filter.get_mu()
                    except Exception:
                        belief_filter = None
                        belief_vec = None

                # Get observation from true simulator state (all physical disturbance is applied in the physics layer)
                state17_noised = state17.copy()
                base_obs = state_to_input(state17, target_position, world_map)
                if belief_enabled and belief_vec is not None:
                    augmented_obs = np.concatenate([base_obs, belief_vec], axis=0).astype(np.float32)
                else:
                    augmented_obs = base_obs
                    
                # Validate observation
                if augmented_obs.shape != (obs_dim,):
                    print(f"❌ Worker {worker_id}: Invalid observation shape {augmented_obs.shape} in episode {episode} step {step}")
                    continue
                
                # Get action from local policy
                
                action, value, log_prob = local_agent.get_action(augmented_obs)
                # get_action returns numpy array, no need for .cpu()
                action = action.astype(np.float32)
                log_prob = log_prob.astype(np.float32)
                value = value.astype(np.float32)
                # Debug: print action details
                # if step == 0:  # Only print for first step to avoid spam
                    # print(f"🔍 Worker {worker_id}: Action debug - shape: {action.shape}, type: {type(action)}, value: {action}")
                
                # SIMPLIFIED DYNAMICS: Apply RL action directly for one MPC interval
                # In this simplified version, we skip the SMPC step for speed.
                # This is a key trade-off for a faster training loop.

                smpc_state = conv.phoenix_to_smpc_state(state17)  # Convert to 9D SMPC state
                smpc_state_noised = conv.phoenix_to_smpc_state(state17_noised)  # Convert to 9D SMPC state
                mpc_horizon = int(RL_DECISION_INTERVAL / MPC_PLANNING_INTERVAL)  # RL_DECISION_INTERVAL / MPC_PLANNING_INTERVAL
                reference_traj = trajgen.generate_trajectory(smpc_state_noised, action, mpc_horizon=mpc_horizon)
                # Get Q/R weights from second agent
                # qr_action, qr_value, qr_log_prob = agent_QR.get_action(augmented_obs)
                # qr_action = qr_action.astype(np.float32)
                # qr_log_prob = qr_log_prob.astype(np.float32)
                # qr_value = qr_value.astype(np.float32)
                # Map normalized agent outputs (-1..1) to requested ranges using exponential interpolation
                # Uniform ranges across all Q-diagonal entries and R scalar:
                #   Q_i in [0.1, 100], R in [1e-5, 1]
                # q_min = np.full(9, 0.1, dtype=np.float32)
                # q_max = np.full(9, 100.0, dtype=np.float32)
                # r_min, r_max = 1e-5, 1.0
                # q_raw = qr_action[:9]
                # r_raw = qr_action[9]
                # # squash to [0,1]
                # s_q = np.clip((q_raw + 1.0) * 0.5, 0.0, 1.0)
                # s_r = float(np.clip((r_raw + 1.0) * 0.5, 0.0, 1.0))
                # # exponential interpolation
                # q_ratio = np.maximum(q_max / np.maximum(q_min, 1e-9), 1.0)
                # q_diag = q_min * (q_ratio ** s_q)
                # r_scalar = float(r_min * ((r_max / r_min) ** s_r))
                # Q_override = np.diag(q_diag.astype(np.float64))
                # R_override = r_scalar * np.eye(4, dtype=np.float64)

                control_actions, terminal_state, state_trajectory, control_outputs, terminal_state_phoenix = ctrl.compute_control(
                    smpc_state, smpc_state_noised, reference_traj
                )

                # terminal_state_phoenix = np.array(state17, copy=True)
                # terminal_state_phoenix[0:3] = reference_traj[0:3,-1]
                # terminal_state_phoenix[7:10] = reference_traj[3:6,-1]
                # print("control_actions", control_actions)
                # print("terminal_state_phoenix", terminal_state_phoenix)
                next_state17 = terminal_state_phoenix
                # Compute CMDP step cost from executed LQR controls in this RL step
                # Weighted quadratic cost on normalized controls (by U_MAX) emphasizing thrust effect
                U_MAX = np.array([65535.0, 2.0, 2.0, 2.0], dtype=np.float64)  # [thrust_cmd, roll_rate, pitch_rate, yaw_rate]
                wT, wR, wY = 0.5, 0.1, 0.05
                actuated_cost = 0.0
                try:
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
                            actuated_cost = float(np.mean(costs_list))
                    else:
                        phoenix_action = np.asarray(control_actions, dtype=np.float64)
                        u_norm = phoenix_action / U_MAX
                        thrust_cmd = u_norm[0]
                        roll_rate  = u_norm[1]
                        pitch_rate = u_norm[2]
                        yaw_rate   = u_norm[3]
                        actuated_cost = float(wT*(thrust_cmd**2) + wR*((roll_rate**2) + (pitch_rate**2)) + wY*(yaw_rate**2))
                except Exception:
                    # Fallback to zero cost on any numerical issue
                    actuated_cost = 0.0
                pos_cur, pos_next = state17[0:3], next_state17[0:3]
                safety_violation = False
                reward, done = world_map.get_position_status_with_cur(pos_next, pos_cur, action, step,state_trajectory)
                step_cost = 0.0
                if(reward==-10.0):
                    step_cost += 10.0
                    # reward = -5.0
                # reward -= actuated_cost
                    
                # Disturbance belief update (for next step) using residual from LQR rollout.
                # Residual is cached on the controller to keep compute_control's public
                # return signature backward-compatible (5 values).
                residual_xy = getattr(ctrl, "last_residual_xy", None)
                # print("!!!!!!!!!!!!!!!!residual_xy!!!!!!!!!!!!!!", residual_xy)
                if belief_filter is not None and residual_xy is not None:
                    try:
                        r_k = np.asarray(residual_xy, dtype=np.float64).reshape(2)
                        if np.all(np.isfinite(r_k)):
                            belief_filter.update(r_k)
                            belief_vec = belief_filter.get_mu()
                        else:
                            print(f"⚠️ Worker {worker_id}: Non-finite residual_xy {r_k}, skipping belief update")
                    except Exception:
                        belief_filter = None
                        belief_vec = None

                next_base_obs = state_to_input(next_state17, target_position, world_map)
                if belief_enabled and belief_vec is not None:
                    next_augmented_obs = np.concatenate([next_base_obs, belief_vec], axis=0).astype(np.float32)
                else:
                    next_augmented_obs = next_base_obs
                # next_augmented_obs = np.concatenate([next_state17[0:2], target_position[:2]-next_state17[0:2]], axis=0).astype(np.float32)

                    
                transition = {
                    'state': augmented_obs,
                    'action': action,
                    'log_prob': log_prob,
                    'value': value,
                    'reward': reward,
                    'cost': step_cost,
                    'next_state': next_augmented_obs,
                    'done': done,
                }
                episode_transitions.append(transition)
                # if(reward == 10.0):
                    # print("Reached target", pos_cur,pos_next )
                    # print("Episode",episode_transitions, "Length", len(episode_transitions))
                    # print("Initial State", episode_transitions[0]['state'], "Length", len(episode_transitions))
                if done:
                    break
                
                state17 = next_state17
            
            if episode_transitions:
                all_collected_episodes.append(episode_transitions)

        
        print(f"✅ Worker {worker_id}: Completed {len(all_collected_episodes)} episodes")
        return all_collected_episodes
        
    except Exception as e:
        print(f"❌ Worker {worker_id}: Critical error: {e}")
        return []


def run_parallel_training(
    num_workers: int = None,
    episodes_per_worker: int = 50,
    save_interval: int = 100,
    world_map=None,
    target_position=None,
    model_path=None,
    disturbance_enabled: bool | None = None,
    disturbance_strength: float | None = None,
    wind_enabled: bool | None = None,
    wind_verbose: bool | None = None,
    wind_mean_frac_min: float | None = None,
    wind_mean_frac_max: float | None = None,
    wind_sigma_frac_min: float | None = None,
    wind_sigma_frac_max: float | None = None,
    belief_enabled: bool = True,
):
    """
    Simplified parallel training: workers collect episodes independently, then batch train.
    """
    if num_workers is None:
        num_workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    
    obs_dim = 46 if belief_enabled else 44
    agent = RLAgent(state_dim=obs_dim, action_dim=2, obs_dim=obs_dim)
    # qr_agent = RLAgent(state_dim=44, action_dim=10)
    if (model_path is not None):
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        agent.actor_critic.load_state_dict(state_dict)
        # Try load matching QR model alongside, if exists (suffix _qr)
        # qr_path = None
        # try:
        #     base, ext = os.path.splitext(model_path)
        #     candidate = f"{base}_qr{ext}"
        #     if os.path.exists(candidate):
        #         qr_path = candidate
        #     else:
        #         # fallback: same dir, filename contains '_qr'
        #         dirname = os.path.dirname(model_path)
        #         for f in os.listdir(dirname):
        #             if f.endswith(ext) and f"_qr" in f and os.path.basename(base) in f:
        #                 qr_path = os.path.join(dirname, f)
        #                 break
        #     # if qr_path is not None:
        #     #     qr_state = torch.load(qr_path, map_location='cpu', weights_only=False)
        #     #     qr_agent.actor_critic.load_state_dict(qr_state)
        # except Exception:
        #     pass
    
    if disturbance_enabled is None:
        disturbance_enabled = ENABLE_POSITION_DISTURBANCE
    if disturbance_strength is None:
        disturbance_strength = POSITION_DISTURBANCE_STRENGTH
    if wind_enabled is None:
        wind_enabled = ENABLE_WIND_DISTURBANCE
    if wind_verbose is None:
        wind_verbose = WIND_VERBOSE
    if wind_mean_frac_min is None:
        wind_mean_frac_min = WIND_MEAN_FRAC_MIN
    if wind_mean_frac_max is None:
        wind_mean_frac_max = WIND_MEAN_FRAC_MAX
    if wind_sigma_frac_min is None:
        wind_sigma_frac_min = WIND_SIGMA_FRAC_MIN
    if wind_sigma_frac_max is None:
        wind_sigma_frac_max = WIND_SIGMA_FRAC_MAX

    print(f"🚀 Starting simplified parallel training: {num_workers} workers, {episodes_per_worker} episodes each")

    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1}/{MAX_ITERATIONS} ---")
        
        policy_state_dict_action = agent.actor_critic.state_dict()
        # policy_state_dict_qr = qr_agent.actor_critic.state_dict()
        
        # Use a Pool to manage workers and collect results
        print(f"🔄 Starting {num_workers} workers...")
        start_time = time.time()
        sim = PhoenixSimulator(
            control_mode='attitude_rate',
            disturbance_enabled=disturbance_enabled,
            disturbance_strength=disturbance_strength,
            wind_enabled=wind_enabled,
            wind_verbose=wind_verbose,
            wind_mean_frac_min=wind_mean_frac_min,
            wind_mean_frac_max=wind_mean_frac_max,
            wind_sigma_frac_min=wind_sigma_frac_min,
            wind_sigma_frac_max=wind_sigma_frac_max,
        )
        with mp.Pool(num_workers) as pool:
            worker_args = [
                (
                    worker_id,
                    episodes_per_worker,
                    world_map,
                    target_position,
                    policy_state_dict_action,
                    disturbance_enabled,
                    disturbance_strength,
                    wind_enabled,
                    wind_verbose,
                    wind_mean_frac_min,
                    wind_mean_frac_max,
                    wind_sigma_frac_min,
                    wind_sigma_frac_max,
                    belief_enabled,
                )
                for worker_id in range(num_workers)
            ]
            collected_data_from_workers = pool.starmap(worker_collect_episodes, worker_args)
        
        collection_time = time.time() - start_time
        print(f"✅ All workers completed in {collection_time:.1f}s")
        
        # Flatten the list of lists into a single list of episodes
        all_episodes = [episode for worker_episodes in collected_data_from_workers for episode in worker_episodes]
        
        # print(f"📊 Collection summary:")
        # print(f"   Workers completed: {len([w for w in collected_data_from_workers if w])}")
        # print(f"   Total episodes collected: {len(all_episodes)}")
        # print(f"   Expected episodes: {num_workers * episodes_per_worker}")
        
        # if not all_episodes:
        #     print(f"⚠️  No episodes were collected in this iteration, skipping training.")
        #     continue
        
        # Prepare data for training
        # print(f"🔧 Preparing training data: {len(all_episodes)} episodes")
        
        sequential_states, sequential_actions, sequential_rewards, sequential_costs, sequential_next_states, sequential_dones, sequential_log_probs, sequential_values = [], [], [], [], [], [], [], []
        # sequential_qr_actions, sequential_qr_log_probs, sequential_qr_values = [], [], []
        
        for episode_transitions in all_episodes:
            if not episode_transitions:
                continue
                
            for transition in episode_transitions:
                sequential_states.append(transition['state'])
                sequential_actions.append(transition['action'])
                sequential_rewards.append(transition['reward'])
                sequential_costs.append(transition.get('cost', 0.0))
                sequential_log_probs.append(transition['log_prob'])
                sequential_values.append(transition['value'])
                sequential_next_states.append(transition['next_state'])
                sequential_dones.append(transition['done'])
                # Collect QR agent data
                # if 'qr_action' in transition:
                #     sequential_qr_actions.append(transition['qr_action'])
                #     sequential_qr_log_probs.append(transition['qr_log_prob'])
                #     sequential_qr_values.append(transition['qr_value'])
            
            if episode_transitions and not episode_transitions[-1]['done']:
                sequential_dones[-1] = True
        
        # Validate data structure
        total_transitions = len(sequential_states)
        episode_count = sum(1 for done in sequential_dones if done)
        
        # print(f"📊 Data structure validation:")
        # print(f"   Total transitions: {total_transitions}")
        # print(f"   Episode boundaries: {episode_count}")
        # print(f"   Expected episodes: {len(all_episodes)}")
        
        if episode_count < len(all_episodes) // 2:
            print(f"⚠️  Too few episode boundaries ({episode_count}), GAE may be incorrect")
            continue
        
        states = np.array(sequential_states, dtype=np.float32)
        actions = np.array(sequential_actions, dtype=np.float32)
        log_probs = np.array(sequential_log_probs, dtype=np.float32)
        values = np.array(sequential_values, dtype=np.float32)
        rewards = np.array(sequential_rewards, dtype=np.float32)
        costs = np.array(sequential_costs, dtype=np.float32) if len(sequential_costs) == len(sequential_states) else np.zeros_like(rewards)
        next_states = np.array(sequential_next_states, dtype=np.float32)
        dones = np.array(sequential_dones, dtype=np.float32)
        # Train
        print("🎯 Training policy...")
        # Update both agents (reference action agent and QR agent)
        agent.update(states, actions, rewards, costs, next_states, dones, log_probs, values)
        # if len(sequential_qr_actions) == len(states):
        #     qr_actions = np.array(sequential_qr_actions, dtype=np.float32)
        #     qr_log_probs = np.array(sequential_qr_log_probs, dtype=np.float32)
        #     qr_values = np.array(sequential_qr_values, dtype=np.float32)
        #     qr_agent.update(states, qr_actions, rewards, costs, next_states, dones, qr_log_probs, qr_values)
        print(f"✅ Training completed")
        
        # Test the updated model before saving
        # test_reward = test_model_performance(agent, world_map, target_position)
        
        # Log metrics
        mean_reward = float(np.mean(rewards))
        mean_action_norm = float(np.mean([np.linalg.norm(a) for a in actions]))
        # mean_qr_norm = float(np.mean([np.linalg.norm(a) for a in sequential_qr_actions])) if sequential_qr_actions else 0.0
        mean_step_cost = float(np.mean(costs)) if costs.size > 0 else 0.0
        max_step_cost = float(np.max(costs)) if costs.size > 0 else 0.0
        
        print(f"📊 Metrics: Mean reward: {mean_reward:.3f}, Mean action norm: {mean_action_norm:.3f}")
        print(f"   GAE data: {episode_count} episodes, {total_transitions} transitions")
        # print(f"   Test reward: {test_reward:.3f}")
        
        # Log to wandb if available
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "training/iteration": iteration + 1,
                    "training/mean_reward": mean_reward,
                    "training/mean_action_norm": mean_action_norm,
                    # "training/mean_qr_norm": mean_qr_norm,
                    "training/episode_count": episode_count,
                    "training/total_transitions": total_transitions,
                    "training/workers": num_workers,
                    "training/episodes_per_worker": episodes_per_worker,
                    "training/collection_time": collection_time if 'collection_time' in locals() else 0.0,
                    "cost/mean_step_cost": mean_step_cost,
                    "cost/max_step_cost": max_step_cost,
                    # "testing/test_reward": test_reward
                })
                # print(f"📊 Logged training metrics to wandb")
        except ImportError:
            pass
        except Exception as e:
            print(f"⚠️  Failed to log to wandb: {e}")
        
        if (iteration + 1) % save_interval == 0:
            model_path_action = os.path.join(MODEL_SAVE_DIR, f"model_iter_{iteration + 1}.pth")
            model_path_qr = os.path.join(MODEL_SAVE_DIR, f"model_iter_{iteration + 1}_qr.pth")
            torch.save(agent.actor_critic.state_dict(), model_path_action)
            # torch.save(qr_agent.actor_critic.state_dict(), model_path_qr)
            print(f"💾 Saved models: {model_path_action}, {model_path_qr}")
            run_test_suite(agent, model_path_action, num_episodes=1, render=False, save_plots=True, output_dir=MODEL_SAVE_DIR, world_map=world_map, sim=sim)
    
    final_model_path = os.path.join(MODEL_SAVE_DIR, "final_model.pth")
    final_model_qr_path = os.path.join(MODEL_SAVE_DIR, "final_model_qr.pth")
    torch.save(agent.actor_critic.state_dict(), final_model_path)
    # torch.save(qr_agent.actor_critic.state_dict(), final_model_qr_path)
    print(f"💾 Final models saved: {final_model_path}, {final_model_qr_path}")
    
    print("✅ Training completed successfully!") 