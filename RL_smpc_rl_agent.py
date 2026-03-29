#!/usr/bin/env python3
"""
RL Agent for RL+SMPC Training Pipeline
Implements PPO algorithm with Phoenix integration
"""

import numpy as np
import torch
import os
from typing import List, Dict, Any
from gymnasium.spaces import Box

from RL_smpc_config import (
    PHOENIX_STATE_DIM, TOTAL_OBS_DIM, ACTION_DIM,
    CLIP_RATIO, TARGET_KL, TRAIN_PI_ITERATIONS, TRAIN_V_ITERATIONS,
    MAX_GRAD_NORM, GAMMA, LAM,
    PI_LEARNING_RATE, VF_LEARNING_RATE, MIN_LEARNING_RATE,
    PI_SCHEDULER_STEP_SIZE, PI_SCHEDULER_GAMMA,
    VF_SCHEDULER_STEP_SIZE, VF_SCHEDULER_GAMMA,
    REWARD_CLIP_MIN, REWARD_CLIP_MAX,
    REWARD_CLIP_MAX, REWARD_NORMALIZE_EPSILON,
    ACTOR_HIDDEN_SIZES, ACTOR_ACTIVATION, VALUE_HIDDEN_SIZES, VALUE_ACTIVATION,
    OBS_MIN_VALUES, OBS_MAX_VALUES, ENTROPY_CONSTANT
)

def simple_minmax_normalize(x: np.ndarray) -> np.ndarray:
    """Simple min-max normalization using predefined min/max values from config"""
    # Ensure input is numpy array and create a copy to avoid modifying input
    x = np.asarray(x, dtype=np.float32)
    normalized = x.copy()  # Create a copy instead of reference
    # Apply min-max normalization: (x - min) / (max - min)
    normalized[..., 0:44] = (x[..., 0:44] - OBS_MIN_VALUES[0:44]) / (OBS_MAX_VALUES[0:44] - OBS_MIN_VALUES[0:44])
    # print('x',x.shape)
    # normalized[..., 0:4] = (x[..., 0:4] - OBS_MIN_VALUES[0:4]) / (OBS_MAX_VALUES[0:4] - OBS_MIN_VALUES[0:4])
    return normalized

class RLAgent:
    """Simple RL agent using existing PPO implementation from Phoenix"""
    
    def __init__(self, state_dim: int =19, action_dim: int = 2):
        self.state_dim = state_dim  # 17D Phoenix state
        self.action_dim = action_dim  # vx, vy, vz
        
        # The ActorCritic expects observations with map encoding
        # 17D state + 2D target + 25D local grid = 44D total
        self.obs_dim = 46
        
        # Create observation and action spaces
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.obs_dim,),
            dtype=np.float32
        )
        self.action_space = Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(action_dim,),
            dtype=np.float32
        )

        # Use simple min-max normalization instead of OnlineMeanStd
        self.obs_oms = None  # Not needed for min-max normalization
        
        # Initialize ActorCritic with proper parameters (expects 44D observations)
        from phoenix_drone_simulation.algs.core import ActorCritic
        self.actor_critic = ActorCritic(
            actor_type='mlp',
            observation_space=self.observation_space,
            action_space=self.action_space,
            use_standardized_obs=False,  # We handle normalization ourselves
            use_scaled_rewards=False,    # We handle reward scaling ourselves
            use_shared_weights=False,    # No shared weights
            ac_kwargs={
                'pi': {
                    'hidden_sizes': ACTOR_HIDDEN_SIZES,
                    'activation': ACTOR_ACTIVATION
                },
                'val': {
                    'hidden_sizes': VALUE_HIDDEN_SIZES,
                    'activation': VALUE_ACTIVATION
                }
            }
        )
        
        # PPO training parameters - More conservative for stability
        self.clip_ratio = CLIP_RATIO
        self.target_kl = TARGET_KL
        self.train_pi_iterations = TRAIN_PI_ITERATIONS
        self.train_v_iterations = TRAIN_V_ITERATIONS
        self.max_grad_norm = MAX_GRAD_NORM
        self.gamma = GAMMA
        self.lam = LAM  # GAE lambda
        
        # Optimizers - Much lower learning rates for stability
        self.pi_optimizer = torch.optim.Adam(self.actor_critic.pi.parameters(), lr=PI_LEARNING_RATE)
        self.vf_optimizer = torch.optim.Adam(self.actor_critic.v.parameters(), lr=VF_LEARNING_RATE)
        
        # Learning rate schedulers - More gradual decay
        self.pi_scheduler = torch.optim.lr_scheduler.StepLR(self.pi_optimizer, step_size=PI_SCHEDULER_STEP_SIZE, gamma=PI_SCHEDULER_GAMMA)
        self.vf_scheduler = torch.optim.lr_scheduler.StepLR(self.vf_optimizer, step_size=VF_SCHEDULER_STEP_SIZE, gamma=VF_SCHEDULER_GAMMA)
        
        # Learning rate floor to prevent collapse
        self.min_lr = MIN_LEARNING_RATE
        
        # Training state
        self.training_mode = True
        self.episode_count = 0
        self.update_count = 0
        
        # Statistics tracking for normalization
        self.obs_stats_updated = False
        
        # Reward tracking for debugging
        self.reward_history = []
        self.episode_rewards = []
        self.best_reward = float('-inf')
        
        # Track latest training losses for logging
        self.last_policy_loss = None
        self.last_value_loss = None
        
        # # Print debug info
        # print(f"🔧 RLAgent initialized:")
        # print(f"   - State dimension: {self.state_dim}")
        # print(f"   - Action dimension: {self.action_dim}")
        # print(f"   - Observation dimension: {self.obs_dim}")
        # print(f"   - Observation space: {self.observation_space}")
        # print(f"   - Action space: {self.action_space}")
        # print(f"   - ActorCritic created successfully")
        
        # # Debug: Check policy network output
        # print(f"🔍 Policy network debug:")
        # print(f"   - Policy network type: {type(self.actor_critic.pi)}")
        # print(f"   - Policy network parameters: {sum(p.numel() for p in self.actor_critic.pi.parameters())} parameters")
        
        # Test policy network with dummy input
        # try:
        #     with torch.no_grad():
        #         dummy_obs = torch.randn(1, self.obs_dim)
        #         dummy_dist = self.actor_critic.pi.dist(dummy_obs)
        #         print(f"   - Policy output distribution: {type(dummy_dist)}")
        #         if hasattr(dummy_dist, 'mean'):
        #             print(f"   - Policy output mean shape: {dummy_dist.mean.shape}")
        #         if hasattr(dummy_dist, 'stddev'):
        #             print(f"   - Policy output stddev shape: {dummy_dist.stddev.shape}")
        # except Exception as e:
        #     print(f"   - Policy network test failed: {e}")
    
    def _augment_state(self, state: np.ndarray) -> np.ndarray:
        """Augment 17D Phoenix state with 2D target position and local grid to create 44D observation"""
        # Use global TARGET_POSITION and WORLD_MAP to avoid circular imports
        global TARGET_POSITION, WORLD_MAP
        
        # Get drone position for local grid encoding
        drone_pos_xy = state[:2]
        
        # Encode local grid around drone position
        local_grid = WORLD_MAP.encode_local_grid(drone_pos_xy)
        
        # Concatenate: [17D state, 2D target, 25D local grid]
        return np.concatenate([state, TARGET_POSITION[:2]-state[:2], local_grid])
    
    def get_actions_batch(self, states: np.ndarray) -> np.ndarray:
        """Get actions for a batch of states (for distributed training)"""
        # States are already augmented (44D) from the worker, no need to augment again
        # Just convert to tensor directly
        if states.shape[1] != self.obs_dim:
            raise ValueError(f"Expected {self.obs_dim}D states, got {states.shape[1]}D")
        states_normalized = simple_minmax_normalize(states)
        states_tensor = torch.FloatTensor(states_normalized)
        
        # Get actions from actor
        with torch.no_grad():
            actions, values, log_probs = self.actor_critic.step(states_tensor)
            
            # Convert to numpy if needed and ensure correct shape
            if isinstance(actions, torch.Tensor):
                actions = actions.detach().cpu().numpy()
            if isinstance(log_probs, torch.Tensor):
                log_probs = log_probs.detach().cpu().numpy()
            
            actions = np.array(actions)
            log_probs = np.array(log_probs)
                        
            if len(log_probs.shape) == 1:
                # Already correct shape
                pass
            else:
                # Flatten to [batch_size]
                log_probs = log_probs.flatten()
            
            # Clip actions to [-1, 1] range
            # print('states_tensor',states_tensor)
            # print('actions',actions)
            # print('log_probs',log_probs)
            # print('values',values)
            # actions = np.clip(actions, -1.0, 1.0)
            # actions = np.tanh(actions) * 0.3
            
        
            return actions, values, log_probs
        
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action for a single state by calling get_actions_batch"""
        # Ensure state is 2D for batch processing
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        # Get actions using batch method
        actions, values, log_probs = self.get_actions_batch(state)       
        # Return single action - actions has shape [1, 2], so actions[0] gives us [2]
        single_action = actions[0]
        single_log_probs = log_probs[0]
        single_values = values[0]
        
        # Ensure it's a proper numpy array with correct shape for multiprocessing
        single_action = np.asarray(single_action, dtype=np.float32)
        
        # Clip action to [-1, 1] range for safety
        # single_action = np.clip(single_action, -1.0, 1.0)
        
        return single_action, single_values, single_log_probs
    
    def update(self, states: np.ndarray, actions: np.ndarray, 
               rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray, log_probs: np.ndarray, values: np.ndarray):
        """Update RL agent with experience using PPO algorithm from Phoenix"""
        if len(states) == 0:
            return
        
        # Convert to numpy arrays if they aren't already
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        log_probs = np.array(log_probs)
        values = np.array(values)
        
        # Use PPO algorithm from Phoenix
        self._ppo_update(states, actions, rewards, next_states, dones, log_probs, values)
    
    def _ppo_update(self, states, actions, rewards, next_states, dones, log_probs, values):
        """Update using PPO algorithm from Phoenix"""
        # Create a simple buffer for this batch of experience   
        batch_size = len(states)
        
        # Normalize rewards first (important for stable training)
        rewards = np.array(rewards)
        if len(rewards) > 1:
            # Clip rewards to prevent extreme values
            rewards = np.clip(rewards, REWARD_CLIP_MIN, REWARD_CLIP_MAX)
            # Normalize with more robust statistics
            rewards = (rewards - rewards.mean()) / (rewards.std() + REWARD_NORMALIZE_EPSILON)
           
        if states.shape[1] != self.obs_dim:
            raise ValueError(f"Expected {self.obs_dim}D states, got {states.shape[1]}D")
        if next_states.shape[1] != self.obs_dim:
            raise ValueError(f"Expected {self.obs_dim}D next_states, got {next_states.shape[1]}D")
        
        states_normalized = simple_minmax_normalize(states)
        next_states_normalized = simple_minmax_normalize(next_states)
        # Convert to tensors for batched processing
        states_tensor = torch.FloatTensor(states_normalized)
        next_states_tensor = torch.FloatTensor(next_states_normalized)
        
        # Get value estimates for all states in ONE forward pass
        with torch.no_grad():
            _, next_values, _ = self.actor_critic.step(next_states_tensor)
        
        # Convert to numpy arrays
        values = np.array(values)
        next_values = np.array(next_values)
        
        # Calculate advantages using GAE (Generalized Advantage Estimation)
        # 
        # KEY IMPROVEMENT: Now properly implements recursive advantage calculation
        # - OLD: Only computed TD errors (δ_t) without recursive advantages
        # - NEW: Full GAE formula A_t = δ_t + γλ(1-d_t)A_{t+1}
        # - Benefits: Better advantage estimates, reduced variance, improved policy learning
        advantages = self._compute_gae(rewards, values, next_values, dones)
        
        # Calculate returns (target values for value function)
        returns = advantages + values
        
        # Convert to tensors for training (augment states for ActorCritic)
        actions_tensor = torch.FloatTensor(actions)
        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns)
    
        
        old_log_probs = torch.FloatTensor(log_probs)
        
        # Normalize advantages AFTER getting old log probs
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        # # Additional clipping for advantages to prevent extreme values
        advantages_tensor = torch.clamp(advantages_tensor, -3.0, 3.0)
        
        # PPO training parameters - use the class defaults for stability
        clip_ratio = self.clip_ratio
        target_kl = self.target_kl
        train_pi_iterations = min(self.train_pi_iterations, batch_size // 2)  # Use class default but cap by batch size
        train_v_iterations = min(self.train_v_iterations, batch_size // 4)
        
        # Update policy network (PPO clipped objective)
        policy_loss, policy_kl, entropy_mean = self._update_policy_ppo(states_tensor, actions_tensor, advantages_tensor, 
                               old_log_probs, clip_ratio, target_kl, train_pi_iterations)
        
        # Update value network
        value_loss = self._update_value_ppo(states_tensor, returns_tensor, train_v_iterations)
        
        # Log training metrics to wandb
        try:
            import wandb
            if wandb.run is not None:  # Check if wandb is actually running
                wandb.log({
                    "ppo/policy_loss": policy_loss,
                    "ppo/value_loss": value_loss,
                    "ppo/kl_divergence": policy_kl,
                    "ppo/policy_iterations": train_pi_iterations,
                    "ppo/value_iterations": train_v_iterations,
                    "ppo/batch_size": batch_size,
                    "ppo/advantages_mean": float(advantages_tensor.mean()),
                    "ppo/advantages_std": float(advantages_tensor.std()),
                    "ppo/returns_mean": float(returns_tensor.mean()),
                    "ppo/returns_std": float(returns_tensor.std()),
                    "ppo/entropy_mean": float(entropy_mean),
                    "normalization/method": "min_max_scaling",
                    "normalization/min_values_norm": float(np.linalg.norm(OBS_MIN_VALUES)),
                    "normalization/max_values_norm": float(np.linalg.norm(OBS_MAX_VALUES))
                })
                print(f"📊 Logged PPO metrics to wandb: policy_loss={policy_loss:.3e}, value_loss={value_loss:.3e}")
            else:
                print(f"⚠️  Wandb run not active, skipping PPO metrics logging")
        except ImportError:
            print(f"⚠️  Wandb not installed, skipping PPO metrics logging")
        except Exception as e:
            print(f"⚠️  Failed to log PPO metrics to wandb: {e}")
        
        # Step the learning rate schedulers
        self.pi_scheduler.step()
        self.vf_scheduler.step()
        
        # Enforce minimum learning rate floor
        try:
            for pg in self.pi_optimizer.param_groups:
                if pg.get('lr', 0.0) < self.min_lr:
                    pg['lr'] = self.min_lr
            for pg in self.vf_optimizer.param_groups:
                if pg.get('lr', 0.0) < self.min_lr:
                    pg['lr'] = self.min_lr
        except Exception:
            pass
        
        # Store latest losses for external logging/printing
        try:
            self.last_policy_loss = float(policy_loss)
            self.last_value_loss = float(value_loss)
        except Exception:
            pass
        
        # No need to update observation statistics with min-max normalization
        
        # Increment update counter
        self.update_count += 1
    
    def _compute_gae(self, rewards, values, next_values, dones, gamma=0.99, lam=0.95):
        """
        Compute Generalized Advantage Estimation (GAE)
        
        This function handles rollout data that may contain multiple episodes from multiple workers.
        The data structure is: rollout_len * num_workers transitions, where each worker contributes
        one transition per rollout step.
        
        GAE formula: A_t = δ_t + γλ(1-d_t)A_{t+1}
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        advantages = np.zeros_like(rewards)
        
        # Initialize running advantage
        last_advantage = 0
        
        # Process in reverse order to compute recursive advantages
        for t in reversed(range(len(rewards))):
            # Check if this is a terminal state (episode boundary)
            if dones[t]:
                # Episode boundary: reset advantage and use next_value = 0
                next_value = 0 # next_values[t]
                last_advantage = 0
            else:
                # Non-terminal: use the actual next value
                next_value = next_values[t]
            
            # Compute TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_value - values[t]

            # Compute advantage: A_t = δ_t + γλ(1-d_t)A_{t+1}
            # Note: (1-d_t) ensures advantage doesn't propagate across episode boundaries
            advantages[t] = delta + gamma * lam * (1 - dones[t]) * last_advantage
            
            # Store current advantage for next iteration
            last_advantage = advantages[t]
        
        return advantages
    
    def _update_policy_ppo(self, states, actions, advantages, old_log_probs, 
                           clip_ratio, target_kl, train_pi_iterations):
        """Update policy network using PPO clipped objective"""
        # Get initial loss for comparison
        if len(states.shape) == 1:
            states = states.unsqueeze(0)  # Add batch dimension if needed
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)  # Add batch dimension if needed
        
        # Get initial policy distribution
        p_dist = self.actor_critic.pi.dist(states)
        
        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iterations):
            self.pi_optimizer.zero_grad()
            loss_pi, entropy_mean = self._compute_loss_pi(states, actions, advantages, old_log_probs, clip_ratio)
            loss_pi.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.pi.parameters(),
                self.max_grad_norm
            )
            
            self.pi_optimizer.step()
            
            # Check KL divergence for early stopping
            q_dist = self.actor_critic.pi.dist(states)
            torch_kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean().item()
            
            if torch_kl > target_kl:
                print(f'Reached KL divergence target after {i+1} policy iterations')
                break
        
        return loss_pi.item(), torch_kl, entropy_mean
    
    def _update_value_ppo(self, states, returns, train_v_iterations):
        """Update value network"""
        loss_v_before = self._compute_loss_v(states, returns).item()
        
        # Train value function
        loss_v = loss_v_before  # Initialize loss_v in case loop doesn't run
        for _ in range(train_v_iterations):
            self.vf_optimizer.zero_grad()
            loss_v = self._compute_loss_v(states, returns)
            loss_v.backward()
            self.vf_optimizer.step()
        
        print(f'PPO Value Update - Loss: {loss_v_before:.6f} -> {loss_v:.6f}')
        
        return loss_v
    
    def _compute_loss_pi(self, states, actions, advantages, old_log_probs, clip_ratio):
        """Compute PPO policy loss with clipping"""
        if len(states.shape) == 1:
            states = states.unsqueeze(0)  # Add batch dimension if needed
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)  # Add batch dimension if needed
        if states.shape[-1] != self.obs_dim:
            raise ValueError(f"States observation dimension mismatch: got {states.shape[-1]}, expected {self.obs_dim}")
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Actions dimension mismatch: got {actions.shape[-1]}, expected {self.action_dim}")
        
        try:
            # The pi attribute is the policy network, not a method
            # We need to get the distribution from the policy network
            dist = self.actor_critic.pi.dist(states)
            # Get log probabilities for the given actions
            log_probs = self.actor_critic.pi.log_prob_from_dist(dist, actions)
        except Exception as e:
            raise ValueError(f"Error calling actor_critic.pi: {e}")
        
        ratio = torch.exp(log_probs - old_log_probs)
        
        # PPO clipped objective
        clip_adv = advantages * torch.clamp(
            ratio,
            1 - clip_ratio,
            1 + clip_ratio
        )
        loss_pi = -(torch.min(ratio * advantages, clip_adv)).mean()
        
        # Add entropy bonus
        loss_pi -= ENTROPY_CONSTANT * dist.entropy().mean() ##3 average entropy logging
        
        return loss_pi, dist.entropy().mean()
    
    def _compute_loss_v(self, obs, ret):
        """Compute value function loss"""
        return ((self.actor_critic.v(obs) - ret) ** 2).mean()
    
    def set_training_mode(self, training: bool):
        """Set training mode on/off"""
        self.training_mode = training
        if training:
            self.actor_critic.train()
        else:
            self.actor_critic.eval()
    

    

    
    def load_normalization_stats(self, filepath: str):
        """Load normalization statistics from file"""
        if os.path.exists(filepath):
            stats = torch.load(filepath)
            self.obs_oms.mean.data = torch.FloatTensor(stats['obs_mean'])
            self.obs_oms.std.data = torch.FloatTensor(stats['obs_std'])
            self.obs_oms.count.data = torch.FloatTensor(stats['stats_count'])
            self.obs_stats_updated = stats['obs_stats_updated']
            print(f"📂 Loaded normalization statistics from: {filepath}")
        else:
            print(f"⚠️  No normalization statistics found at: {filepath}")
    
    def get_normalization_info(self) -> dict:
        """Get current normalization statistics for logging (min-max normalization)"""
        return {
            'normalization_method': 'min_max_scaling',
            'min_values_norm': float(np.linalg.norm(OBS_MIN_VALUES)),
            'max_values_norm': float(np.linalg.norm(OBS_MAX_VALUES))
        } 