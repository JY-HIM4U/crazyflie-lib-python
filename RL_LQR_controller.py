#!/usr/bin/env python3
"""
SMPC Controller for RL+SMPC Training Pipeline
Handles Model Predictive Control integration with Phoenix simulator
"""

import numpy as np
from typing import Tuple, List
from RL_smpc_config import (
    SMPC_ALPHA,
    SMPC_CONSTRAINT_EDGE_COORDINATES,
    ENABLE_POSITION_DISTURBANCE,
    POSITION_DISTURBANCE_STRENGTH,
    PHYSICS_TIME_STEP,
    MPC_PLANNING_INTERVAL,
    RL_DECISION_INTERVAL,
)
from RL_smpc_utils import thrust_to_cmd

# Import SMPC modules
from parameters import load_parameters
from LQR import compute_lqt_gains, lqt_control_step
from quadcopter_dynamics import quadcopter_dynamics_single_step_linear
from RL_smpc_simulation import StateConverter, PhoenixSimulator


def quad_dynamics_nonlinear_optimized(state: np.ndarray, action: np.ndarray, dt: float = 0.025) -> np.ndarray:
    """
    Lightweight nonlinear quadcopter dynamics used for model-error diagnostics.
    state: (9,)  -> [x, y, z, vx, vy, vz, roll, pitch, yaw]
    action: (4,) -> [Thrust, Torque_x, Torque_y, Torque_z]
    """
    # Physical constants
    m = 0.027
    g = 9.81
    ix_inv = 1.0
    iy_inv = 1.0
    iz_inv = 1.0

    # Current state
    vx = state[3]
    vy = state[4]
    vz = state[5]
    phi = state[6]
    theta = state[7]
    psi = state[8]

    thrust = action[0]

    # Precompute trig
    s_phi, c_phi = np.sin(phi), np.cos(phi)
    s_the, c_the = np.sin(theta), np.cos(theta)
    s_psi, c_psi = np.sin(psi), np.cos(psi)

    # Acceleration in inertial frame
    tm = thrust / m
    ax = tm * (c_phi * s_the * c_psi + s_phi * s_psi)
    ay = tm * (c_phi * s_the * s_psi - s_phi * c_psi)
    az = tm * (c_phi * c_the) - g

    new_state = np.empty(9, dtype=np.float64)

    # Position update
    new_state[0] = state[0] + vx * dt
    new_state[1] = state[1] + vy * dt
    new_state[2] = state[2] + vz * dt

    # Velocity update
    new_state[3] = vx + ax * dt
    new_state[4] = vy + ay * dt
    new_state[5] = vz + az * dt

    # Euler angle update (angular rate = torque * I_inv)
    new_state[6] = phi + (action[1] * ix_inv) * dt
    new_state[7] = theta + (action[2] * iy_inv) * dt
    new_state[8] = psi + (action[3] * iz_inv) * dt

    return new_state

def shrinking_horizon_LQR_phoenix(
    dp_params,
    mpc_params,
    sim_params,
    system_state,
    noised_system_state,
    reference,
    Q,
    R,
    phoenix_sim=None,
    upper_layer_constraints=None,
    disturbance_enabled=False,
    disturbance_strength=5e-4,
):
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
        phoenix_sim: Existing Phoenix simulator (must be provided)
        upper_layer_constraints: Constraints from the upper layer (not implemented yet)
        
    Returns:
        terminal_state, state_trajectory, reward, control_actions, control_outputs, phoenix_sim, residual_xy
    """
    reward = 0.0
    x_trajectory = [system_state]
    control_actions = np.zeros((4, mpc_params['J']))
    # Use existing simulator or create one if needed
    if phoenix_sim is None:
        raise RuntimeError("phoenix_sim must be provided to SMPC; don't reset the global simulator here")
    
    # Track control outputs
    control_outputs = []
    current_state = system_state

    # Disturbance residual for first MPC step (XY). We will decompose it into
    # (a) pure model mismatch between linear SMPC model and a no-disturbance
    #     nonlinear Phoenix model, and
    # (b) disturbance-only residual between disturbed and no-disturbance Phoenix.
    residual_xy = None
    # Precompute LQT gains for the entire horizon
    A = mpc_params['A']
    B = mpc_params['B']
    lqt = compute_lqt_gains(A, B, Q, R, reference, Q_terminal=Q)
    K_seq = lqt['K_seq']
    F_seq = lqt['F_seq']
    s_seq = lqt['s_seq']

    # Prepare a secondary, no-disturbance Phoenix simulator for residual
    # decomposition. This lets us separate wind / injection disturbances from
    # pure linearization error of the SMPC model.
    phoenix_sim_nominal = None
    # try:
    #     phoenix_state0 = phoenix_sim.get_state()
    #     phoenix_sim_nominal = PhoenixSimulator(
    #         control_mode='attitude_rate',
    #         disturbance_enabled=False,
    #         disturbance_strength=0.0,
    #         disturbance_verbose=False,
    #         wind_enabled=False,
    #         wind_verbose=False,
    #         wind_mean_frac_min=None,
    #         wind_mean_frac_max=None,
    #         wind_sigma_frac_min=None,
    #         wind_sigma_frac_max=None,
    #         position_injection_enabled=False,
    #     )
    #     try:
    #         # Synchronize nominal simulator state with the disturbed simulator.
    #         drone_nom = phoenix_sim_nominal.drone
    #         bc_nom = phoenix_sim_nominal.bc
    #         pos0 = phoenix_state0[0:3]
    #         quat0 = phoenix_state0[3:7]
    #         vel0 = phoenix_state0[7:10]
    #         ang_body0 = phoenix_state0[10:13]
    #         # Convert body-frame angular rates to world frame so that
    #         # CrazyFlieAgent.update_information() recovers the same body rates.
    #         R0 = StateConverter.quaternion_to_rotation_matrix(quat0)
    #         ang_world0 = R0 @ ang_body0
    #         bc_nom.resetBasePositionAndOrientation(drone_nom.body_unique_id, pos0, quat0)
    #         bc_nom.resetBaseVelocity(drone_nom.body_unique_id, vel0, ang_world0)
    #         # Refresh internal agent state to match Bullet
    #         drone_nom.update_information()
    #     except Exception as sync_error:
    #         print(f"⚠️  Failed to synchronize nominal Phoenix simulator state: {sync_error}")
    #         phoenix_sim_nominal = None
    # except Exception as init_error:
    #     print(f"⚠️  Failed to create nominal (no-disturbance) Phoenix simulator: {init_error}")
    #     phoenix_sim_nominal = None

    # MPC model time step (dt_model) vs physics time step:
    # A, B are discretized at dt_model ≈ MPC_PLANNING_INTERVAL, while Phoenix runs at PHYSICS_TIME_STEP.
    # We therefore roll the Phoenix simulator for n_sub physics ticks per MPC step.
    dt_model = float(MPC_PLANNING_INTERVAL)
    n_sub = max(1, int(round(dt_model / float(PHYSICS_TIME_STEP))))

    # For diagnostic nonlinear model rollout (single-step horizon only)
    nonlinear_state_next = None
    residual_xy = np.zeros(2, dtype=np.float64)
    disturbance_true = np.zeros(2, dtype=np.float64)
    for j in range(mpc_params['J']):
        try:
            # LQT control for current step
            u_hat = lqt_control_step(K_seq[j], F_seq[j], s_seq[j + 1], current_state)
            # Create a lightweight predicted 2-step trajectory for diagnostics
            x_pred_next = A @ current_state + B @ u_hat
            x_sol = np.stack([current_state, x_pred_next], axis=1)
            
        except Exception as qp_error:
            print(f"⚠️  LQT control failed at step {j}: {qp_error}")
            # Re-raise the exception to trigger rollout skipping in the worker
            raise qp_error
        
        # Extract control action for this step
        # u_hat contains: [thrust, tau_x, tau_y, tau_z] (torques)
        # x_sol contains: [pos(3), vel(3), rpy(3)] (predicted states)
        
        try:
            # Get thrust from u_sol
            T_max= 0.027*9.81*2.25
            thrust_force = u_hat[0] * T_max  # First control input is thrust
            hover_thrust = 0.027 * 9.81  # 0.265 N
            thrust_cmd = thrust_to_cmd(hover_thrust+thrust_force)
            
            # Get angular rates from u_sol (torques converted to rates)
            # Construct control action: [thrust, roll_rate, pitch_rate, yaw_rate]
            current_control = np.array([
                thrust_cmd,      # thrust (N)
                u_hat[1]/ (np.pi/3),      # roll_rate (rad/s)  * scailing at controller of simulator
                u_hat[2]/ (np.pi/3),      # pitch_rate (rad/s)
                u_hat[3]/ (np.pi/3)       # yaw_rate (rad/s)
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
        
    
        # Apply control to Phoenix simulator for one MPC model interval using n_sub physics ticks
        smpc_next_state_nominal = None

        for aa in range(n_sub):
            phoenix_sim.apply_control_multiple_steps(current_control, num_steps=1)
            next_state = phoenix_sim.get_state()
            smpc_next_state = StateConverter.phoenix_to_smpc_state(next_state)
            x_trajectory.append(smpc_next_state)
            if phoenix_sim_nominal is not None:
                phoenix_sim_nominal.apply_control_multiple_steps(current_control, num_steps=1)
                next_state_nominal = phoenix_sim_nominal.get_state()
                smpc_next_state_nominal = StateConverter.phoenix_to_smpc_state(next_state_nominal)
            # print(f"~~~~~~~~~~Pos difference true - nominal~~~~~~~~~~~: {aa} {smpc_next_state[0:2] - smpc_next_state_nominal[0:2]}")

        # Accumulate trajectory-model residual over MPC horizon
        residual_xy += np.asarray(
            smpc_next_state[0:2] - x_pred_next[0:2],
            dtype=np.float64,
        ).reshape(2)

        # Accumulate disturbance-only residual only when a valid nominal state exists
        if smpc_next_state_nominal is not None:
            disturbance_true += np.asarray(
                smpc_next_state[0:2] - smpc_next_state_nominal[0:2],
                dtype=np.float64,
            ).reshape(2)
        control_outputs.append({
            'step': j,
            'control_action': current_control.copy(),
            'phoenix_state': next_state.copy(),
            'smpc_state': smpc_next_state.copy(),
            # 'ref_error': ref_error,
            # 'smpc_error': smpc_error
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

        # Feedback: use actual Phoenix state for next LQT step (9D: pos, vel, rpy)
        current_state = np.asarray(smpc_next_state, dtype=np.float64).reshape(9)
        
        if phoenix_sim_nominal is not None:
            phoenix_state0 = phoenix_sim.get_state()
            drone_nom = phoenix_sim_nominal.drone
            bc_nom = phoenix_sim_nominal.bc
            pos0 = phoenix_state0[0:3]
            quat0 = phoenix_state0[3:7]
            vel0 = phoenix_state0[7:10]
            ang_body0 = phoenix_state0[10:13]
            # Convert body-frame angular rates to world frame so that
            # CrazyFlieAgent.update_information() recovers the same body rates.
            R0 = StateConverter.quaternion_to_rotation_matrix(quat0)
            ang_world0 = R0 @ ang_body0
            bc_nom.resetBasePositionAndOrientation(drone_nom.body_unique_id, pos0, quat0)
            bc_nom.resetBaseVelocity(drone_nom.body_unique_id, vel0, ang_world0)
            # Refresh internal agent state to match Bullet
            drone_nom.update_information()
    
    # Return the first control action for immediate use, plus tracking data
    first_control_action = control_actions[:, 0] if control_actions.shape[1] > 0 else np.zeros(4)
    # print("!!!!!!!!!!!!!!!!residual_xy!!!!!!!!!!!!!!", residual_xy)
    # print("!!!!!!!!!!!!!!!!Trajectory Error!!!!!!!!!!!!!!", x_trajectory[-1][0:2] - reference[0:2, -1])
    info = getattr(phoenix_sim.physics, "get_disturbance_info", None)
    if callable(info):
        disturb_info = phoenix_sim.physics.get_disturbance_info()
        wind_cfg = disturb_info.get("wind_config", None)
        if wind_cfg is not None:
            # Same region split as in physics: X<0.5, 0.5<=X<1.5, X>=1.5
            if smpc_next_state[0] < 0.5:
                region_idx = 0
            elif smpc_next_state[0] < 1.5:
                region_idx = 1
            else:
                region_idx = 2
            mean_force_xy = np.asarray(wind_cfg[region_idx].get("mean_force_xy", [0.0, 0.0]), dtype=np.float64)
            # print(f"~~~~~~~~~~True Mean Wind force xy~~~~~~~~~~~: {mean_force_xy}")
    # print("!!!!!!!!!!!!!!!!disturbance_true!!!!!!!!!!!!!!", disturbance_true)
    # residual_xy = np.asarray(x_trajectory[-1][0:2] - reference[0:2, -1], dtype=np.float64).reshape(2)
    # residual_xy = np.asarray(disturbance_true, dtype=np.float64).reshape(2)
    return x_trajectory[-1], x_trajectory, reward, first_control_action, control_outputs, phoenix_sim, residual_xy


class LQRController:
    """LQR controller wrapper"""
    
    def __init__(self, phoenix_sim=None):
        # Load SMPC parameters
        self.com_params, self.dp_params, self.mpc_params, self.sim_params = load_parameters()
        
        # Store reference to Phoenix simulator
        self.phoenix_sim = phoenix_sim
        # Last (smoothed) disturbance residual from SMPC model (XY)
        self.last_residual_xy = None
        
        # SMPC parameters
        self.alpha = SMPC_ALPHA  # Safety probability
        self.constraint_edge_coordinates = SMPC_CONSTRAINT_EDGE_COORDINATES  # Example constraints
        
        # Control tracking
        self.control_history = []
        self.disturbance_enabled = ENABLE_POSITION_DISTURBANCE
        self.disturbance_strength = POSITION_DISTURBANCE_STRENGTH

        # Running window for residual_xy smoothing (moving average)
        # Window length ≈ RL_DECISION_INTERVAL / MPC_PLANNING_INTERVAL (in steps)
        self.residual_window: List[np.ndarray] = []
        self.residual_window_size = max(
            1, int(round(float(RL_DECISION_INTERVAL) / float(MPC_PLANNING_INTERVAL)))
        )
        
    def set_simulator(self, phoenix_sim):
        """Set the Phoenix simulator to use for SMPC"""
        self.phoenix_sim = phoenix_sim
        
    def compute_control(self,current_state: np.ndarray,noised_current_state: np.ndarray, trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        """
        Compute SMPC control actions
        
        Args:
            current_state: Current 9D SMPC state
            noised_current_state: Noised current 9D SMPC state
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
        system_state = current_state.copy()
        noised_system_state = noised_current_state.copy()
        reference = trajectory  # Use the generated trajectory as reference
        
        # Cost matrices (allow RL overrides)
        Q = mpc_params['MPC_Q']
        R = mpc_params['MPC_R']
        # # Apply optional overrides from RL QR agent
        # if Q_override is not None:
        #     # Accept either full (9x9) matrix or 9-length diagonal
        #     if Q_override.ndim == 1:
        #         if Q_override.shape[0] != Q.shape[0]:
        #             raise ValueError(f"Q_override diagonal length {Q_override.shape[0]} does not match expected {Q.shape[0]}")
        #         Q = np.diag(np.asarray(Q_override, dtype=np.float64))
        #     else:
        #         Q = np.asarray(Q_override, dtype=np.float64)
        # if R_override is not None:
        #     # Accept scalar or full (4x4) matrix
        #     if np.isscalar(R_override):
        #         R = float(R_override) * np.eye(R.shape[0], dtype=np.float64)
        #     else:
        #         R = np.asarray(R_override, dtype=np.float64)
        
        # Run shrinking horizon SMPC with Phoenix integration
        # Pass the existing simulator to avoid creating new ones
        terminal_state, state_trajectory, reward, control_actions, control_outputs, phoenix_sim, residual_xy = shrinking_horizon_LQR_phoenix(
            dp_params=dp_params,
            mpc_params=mpc_params,
            sim_params=sim_params,
            system_state=system_state,
            noised_system_state=noised_system_state,
            reference=reference,
            Q=Q,
            R=R,
            phoenix_sim=self.phoenix_sim,  # Pass existing simulator
            upper_layer_constraints=None,
            disturbance_enabled=self.disturbance_enabled,
            disturbance_strength=self.disturbance_strength
        )
        # Cache residual as a smoothed moving average over recent calls
        if residual_xy is not None:
            self.last_residual_xy = residual_xy
        else:
            self.last_residual_xy = None

        # Debug: print current smoothed residual for disturbance estimation
        # print("last_residual_xy (smoothed over recent RL/MPC steps):", self.last_residual_xy)
        
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

        # Maintain 5-value return signature for compatibility with existing callers
        return control_actions, terminal_state, state_trajectory, control_outputs, terminal_state_phoenix 