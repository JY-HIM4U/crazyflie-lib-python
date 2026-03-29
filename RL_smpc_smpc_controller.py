#!/usr/bin/env python3
"""
SMPC Controller for RL+SMPC Training Pipeline
Handles Model Predictive Control integration with Phoenix simulator
"""

import numpy as np
from typing import Tuple, List
from RL_smpc_config import SMPC_ALPHA, SMPC_CONSTRAINT_EDGE_COORDINATES
from RL_smpc_utils import thrust_to_cmd

# Import SMPC modules
from parameters import load_parameters
from smpc import shrinking_horizon_SMPC, DAQP_fast, QP
from quadcopter_dynamics import quadcopter_dynamics_single_step_linear

from RL_smpc_config import ENABLE_POSITION_DISTURBANCE, POSITION_DISTURBANCE_STRENGTH

def shrinking_horizon_SMPC_phoenix(dp_params, mpc_params, sim_params, system_state, reference, Q, R, phoenix_sim=None, upper_layer_constraints=None, 
                 disturbance_enabled=False, disturbance_strength=5e-4):
    """
    Implements the shrinking horizon MPC algorithm with Phoenix simulator integration.
    This function recursively calls the QP solver for the MPC problem, reducing the prediction horizon at each step.
    
    Args:
        dp_params: Dynamic programming parameters
        mpc_params: MPC parameters
        sim_params: Simulation parameters
        system_state: Current state of the system
        reference: Reference state to track
        Q: State cost matrix
        R: Control cost matrix
        phoenix_sim: Existing Phoenix simulator (if None, creates a new one)
        upper_layer_constraints: Constraints from the upper layer (not implemented yet)
        
    Returns:
        terminal_state, state_trajectory, reward, control_actions, control_outputs, phoenix_sim
    """
    reward = 0.0
    x_trajectory = [system_state]
    control_actions = np.zeros((4, mpc_params['J']))
    # Use existing simulator or create one if needed
    if phoenix_sim is None:
        raise RuntimeError("phoenix_sim must be provided to SMPC; don't reset the global simulator here")
    
    # Track control outputs
    control_outputs = []
    
    for j in range(mpc_params['J']):
        try:
            # Solve QP for current step
            current_state = x_trajectory[-1]
            if(disturbance_enabled):
                position_noise = np.random.multivariate_normal(
                    np.zeros(2), 
                    disturbance_strength * np.eye(2)
                )
                current_state[0:2] += position_noise
                
            u_sol, x_sol, epsilon_sol = QP(dp_params, mpc_params, x_trajectory[-1], Q, R, reference, j)       

            if u_sol is None or x_sol is None:
                raise ValueError("QP solver did not return a valid solution.")
            
        except Exception as qp_error:
            print(f"⚠️  QP solver failed at step {j}: {qp_error}")
            # Re-raise the exception to trigger rollout skipping in the worker
            raise qp_error
        
        # Extract control action for this step
        # u_sol contains: [thrust, tau_x, tau_y, tau_z] (torques)
        # x_sol contains: [pos(3), vel(3), rpy(3)] (predicted states)
        
        try:
            # Get thrust from u_sol
            T_max= 0.027*9.81*2.25
            thrust_force = u_sol[0, 0] * T_max  # First control input is thrust
            hover_thrust = 0.027 * 9.81  # 0.265 N
            thrust_cmd = thrust_to_cmd(hover_thrust+thrust_force)
            
            # Get angular rates from u_sol (torques converted to rates)
            # Construct control action: [thrust, roll_rate, pitch_rate, yaw_rate]
            current_control = np.array([
                thrust_cmd,      # thrust (N)
                u_sol[1,0]/ (np.pi/3),      # roll_rate (rad/s) from predicted state
                u_sol[2,0]/ (np.pi/3),      # pitch_rate (rad/s) from predicted state  
                u_sol[3,0]/ (np.pi/3)       # yaw_rate (rad/s) from predicted state
            ])
            
        except Exception as control_error:
            print(f"⚠️  Control action construction failed at step {j}: {control_error}")
            # Re-raise the exception to trigger rollout skipping
            raise control_error
        
        # Check if control actions are reasonable and clip if needed
        current_control[0] = np.clip(current_control[0], 0, 65535)  # Thrust limits
        current_control[1:4] = np.clip(current_control[1:4], -2.0, 2.0)  # Rate limits
        
        # Store control action
        control_actions[:, j] = current_control
        
        try:
            # Apply control to Phoenix simulator for multiple physics steps to match MPC planning interval
            phoenix_sim.apply_control_multiple_steps(current_control)
            
            # Get next state from Phoenix simulator
            next_state = phoenix_sim.get_state()
            
            # Convert Phoenix state to SMPC state format
            from RL_smpc_simulation import StateConverter
            smpc_next_state = StateConverter.phoenix_to_smpc_state(next_state)
            
        except Exception as sim_error:
            print(f"⚠️  Phoenix simulation failed at step {j}: {sim_error}")
            # Re-raise the exception to trigger rollout skipping
            raise sim_error
        
        # Calculate tracking errors (with safety checks)
        try:
            if j+1 < reference.shape[1]:
                ref_error = np.linalg.norm(reference[:3, j+1] - smpc_next_state[:3])
            else:
                ref_error = 0.0
                
            if x_sol.shape[1] > 1:
                smpc_error = np.linalg.norm(x_sol[:3, 1] - smpc_next_state[:3])
            else:
                smpc_error = 0.0
                
        except Exception as error_error:
            print(f"⚠️  Error calculation failed at step {j}: {error_error}")
            # Re-raise the exception to trigger rollout skipping
            raise error_error
        
        # Append to trajectory
        x_trajectory.append(smpc_next_state)
        
        # Track control output
        control_outputs.append({
            'step': j,
            'control_action': current_control.copy(),
            'phoenix_state': next_state.copy(),
            'smpc_state': smpc_next_state.copy(),
            'ref_error': ref_error,
            'smpc_error': smpc_error
        })
        
        # Compute reward for this step (with safety checks)
        try:
            step_reward = 1 / sim_params['simulation_steps_per_input'] * (
                    np.transpose(smpc_next_state - reference[:, j])@Q@(smpc_next_state - reference[:, j])
                    + np.transpose(current_control)@R@current_control)
            reward += step_reward
        except Exception as reward_error:
            print(f"⚠️  Reward calculation failed at step {j}: {reward_error}")
            # Re-raise the exception to trigger rollout skipping
            raise reward_error
    
    # Return the first control action for immediate use, plus tracking data
    first_control_action = control_actions[:, 0] if control_actions.shape[1] > 0 else np.zeros(4)
    
    return x_trajectory[-1], x_trajectory, reward, first_control_action, control_outputs, phoenix_sim


class SMPCController:
    """SMPC controller wrapper"""
    
    def __init__(self, phoenix_sim=None):
        # Load SMPC parameters
        self.com_params, self.dp_params, self.mpc_params, self.sim_params = load_parameters()
        
        # Store reference to Phoenix simulator
        self.phoenix_sim = phoenix_sim
        
        # SMPC parameters
        self.alpha = SMPC_ALPHA  # Safety probability
        self.constraint_edge_coordinates = SMPC_CONSTRAINT_EDGE_COORDINATES  # Example constraints
        
        # Control tracking
        self.control_history = []
        self.disturbance_enabled = ENABLE_POSITION_DISTURBANCE
        self.disturbance_strength = POSITION_DISTURBANCE_STRENGTH
        
    def set_simulator(self, phoenix_sim):
        """Set the Phoenix simulator to use for SMPC"""
        self.phoenix_sim = phoenix_sim
        
    def compute_control(self, current_state: np.ndarray, trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        """
        Compute SMPC control actions
        
        Args:
            current_state: Current 9D SMPC state
            trajectory: Generated trajectory from RL agent
            
        Returns:
            control_actions: [thrust, roll_rate, pitch_rate, yaw_rate]
            terminal_state: Final state from SMPC
            state_trajectory: Full state trajectory from SMPC
            control_outputs: List of control tracking data
        """
        # Get parameters from parent controller
        dp_params = getattr(self, 'dp_params', None)
        mpc_params = getattr(self, 'mpc_params', None)
        sim_params = getattr(self, 'sim_params', None)
        
        # If not available, load them (fallback)
        if dp_params is None:
            com_params, dp_params, mpc_params, sim_params = load_parameters()
        
        # Set up SMPC parameters
        system_state = current_state
        reference = trajectory  # Use the generated trajectory as reference
        
        # Cost matrices
        Q = mpc_params['MPC_Q']
        R = mpc_params['MPC_R']
        
        # Run shrinking horizon SMPC with Phoenix integration
        # Pass the existing simulator to avoid creating new ones
        terminal_state, state_trajectory, reward, control_actions, control_outputs, phoenix_sim = shrinking_horizon_SMPC_phoenix(
            dp_params=dp_params,
            mpc_params=mpc_params,
            sim_params=sim_params,
            system_state=system_state,
            reference=reference,
            Q=Q,
            R=R,
            phoenix_sim=self.phoenix_sim,  # Pass existing simulator
            upper_layer_constraints=None,
            disturbance_enabled=self.disturbance_enabled,
            disturbance_strength=self.disturbance_strength
        )
        
        # Store control history
        self.control_history.append({
            'timestamp': len(self.control_history),
            'current_state': current_state.copy(),
            'trajectory': trajectory.copy(),
            'control_action': control_actions.copy(),
            'terminal_state': terminal_state.copy(),
            'reward': reward,
            'control_outputs': control_outputs
        })
        
        terminal_state_phoenix = self.phoenix_sim.get_state()

        return control_actions, terminal_state, state_trajectory, control_outputs, terminal_state_phoenix 