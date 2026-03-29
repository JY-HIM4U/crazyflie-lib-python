#!/usr/bin/env python3
"""
Simulation components for RL+SMPC Training Pipeline
Contains PhoenixSimulator, StateConverter, and TrajectoryGenerator classes
"""

import numpy as np
import pybullet as p
import pybullet_data
from RL_smpc_config import (
    START_POSITION, PHYSICS_TIME_STEP, MPC_PLANNING_INTERVAL,
    ENABLE_POSITION_DISTURBANCE, POSITION_DISTURBANCE_STRENGTH, DISTURBANCE_VERBOSE,
    PHYSICS_SOLVER_ITERATIONS, PHYSICS_DETERMINISTIC_OVERLAPPING_PAIRS, PHYSICS_NUM_SUB_STEPS,
    MAX_ANGULAR_RATE, RATE_SCALE_FACTOR, ACTION_SCALE
)
from RL_smpc_utils import thrust_to_cmd

class StateConverter:
    """Enhanced state converter with proper coordinate system handling"""
    
    def __init__(self):
        # Coordinate transformation matrices
        self.R_world_to_body = None
        self.R_body_to_world = None
        
    @staticmethod
    def phoenix_to_smpc_state(phoenix_state: np.ndarray) -> np.ndarray:
        """
        Convert Phoenix 17D state to SMPC 9D state with proper coordinate handling
        
        Phoenix state: [pos(3), quat(4), vel(3), ang_vel(3), last_action(4)]
        SMPC state: [pos(3), vel(3), rpy(3)]
        """
        # Extract components from Phoenix state
        pos = phoenix_state[0:3]      # x, y, z (world frame)
        quat = phoenix_state[3:7]     # qw, qx, qy, qz (world frame)
        vel = phoenix_state[7:10]     # vx, vy, vz (world frame)
        ang_vel = phoenix_state[10:13] # wx, wy, wz (body frame)
        
        # Convert quaternion to Euler angles using PyBullet
        rpy = np.array(p.getEulerFromQuaternion(quat))
        
        # SMPC expects world-frame velocities, so no conversion needed
        vel_world = vel
        
        # Construct SMPC state: [pos(3), vel_world(3), rpy(3)]
        smpc_state = np.concatenate([pos, vel_world, rpy])
        
        return smpc_state
    
    @staticmethod
    def smpc_to_phoenix_state(smpc_state: np.ndarray, phoenix_state: np.ndarray) -> np.ndarray:
        """
        Convert SMPC 12D state back to Phoenix 17D state format
        Handles coordinate transformations properly
        """
        # Extract SMPC components
        pos = smpc_state[0:3]         # position (world frame)
        vel_world = smpc_state[3:6]   # velocity (world frame)
        rpy = smpc_state[6:9]         # euler angles (world frame)
        ang_vel_body = smpc_state[9:12] # angular velocity (body frame)
        
        # Convert Euler angles to quaternion using PyBullet
        quat = np.array(p.getQuaternionFromEuler(rpy))
        
        # Update Phoenix state with SMPC state
        updated_state = phoenix_state.copy()
        updated_state[0:3] = pos         # position (world frame)
        updated_state[3:7] = quat        # quaternion (world frame)
        updated_state[7:10] = vel_world    # velocity (world frame)
        updated_state[10:13] = ang_vel_body # angular velocity (body frame)
        # Note: last_action (indices 13:17) remains unchanged
        
        return updated_state
    
    @staticmethod
    def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix"""
        qw, qx, qy, qz = quat
        
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*qx**2 - 2*qz**2, 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*qx**2 - 2*qy**2]
        ])
        
        return R


class TrajectoryGenerator:
    """Generates trajectory from velocity commands"""
    
    def __init__(self, dt: float = 0.025):  # Changed to match MPC_PLANNING_INTERVAL
        self.dt = dt
        # MPC horizon will be set by parent controller
        self.mpc_horizon = int(1.0/dt)  # Default RL decision interval
    
    def generate_trajectory(self, current_state: np.ndarray, velocity_cmd: np.ndarray, mpc_horizon: int = None) -> np.ndarray:
        """
        Generate trajectory for MPC horizon based on velocity command
        
        Args:
            current_state: Current 9D SMPC state
            velocity_cmd: [vx, vy, vz] or [vx, vy] velocity command (vz assumed 0 if omitted)
            mpc_horizon: MPC prediction horizon (if None, use default)
            
        Returns:
            trajectory: Array of states for MPC horizon (J + 1 steps)
        """
        
        mpc_horizon = self.mpc_horizon if mpc_horizon is None else mpc_horizon
        
        # Ensure velocity_cmd is a numeric numpy array of shape (3,), with vz=0 if only vx,vy provided
        velocity_cmd = np.asarray(velocity_cmd, dtype=float).flatten()
        if velocity_cmd.shape[0] == 2:
            velocity_cmd = np.array([velocity_cmd[0], velocity_cmd[1], 0.0], dtype=float)
        
        trajectory = np.zeros((9, mpc_horizon + 1))
        trajectory[:, 0] = current_state
        
        # Simple constant velocity model for trajectory generation
        for i in range(mpc_horizon):
            # Update position based on velocity command
            trajectory[0:3, i+1] = trajectory[0:3, i] + velocity_cmd * self.dt * ACTION_SCALE
            trajectory[3:6, i+1] = velocity_cmd * ACTION_SCALE 
            
            # Keep other states constant (simplified)
            trajectory[6:, i+1] = 0
        
        return trajectory


class PhoenixSimulator:
    """Enhanced Phoenix simulator wrapper using direct physics and agent integration"""
    
    def __init__(self, control_mode='attitude_rate', disturbance_enabled=False, disturbance_strength=0.0, disturbance_verbose=False):
        # Import Phoenix components directly
        from phoenix_drone_simulation.envs.control import AttitudeRate, Attitude, PWM
        from phoenix_drone_simulation.envs.agents import CrazyFlieBulletAgent
        from phoenix_drone_simulation.envs.physics import PyBulletPhysics
        from pybullet_utils import bullet_client
        import pybullet as pb
        
        # Control mode
        self.control_mode = control_mode
        
        # Initialize PyBullet client
        self.bc = bullet_client.BulletClient(connection_mode=pb.DIRECT)
        
        # Set up physics engine - Using PHYSICS_TIME_STEP (1ms for 1000Hz)
        self.bc.setGravity(0, 0, -9.81)
        self.bc.setPhysicsEngineParameter(
            fixedTimeStep=PHYSICS_TIME_STEP,  # 0.001s (1ms) for 1000Hz physics
            numSolverIterations=PHYSICS_SOLVER_ITERATIONS,
            deterministicOverlappingPairs=PHYSICS_DETERMINISTIC_OVERLAPPING_PAIRS,
            numSubSteps=PHYSICS_NUM_SUB_STEPS
        )
        
        # Load ground plane
        try:
            # Try to use pybullet_data if available
            import pybullet_data
            self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        except ImportError:
            # Fallback: use a default path or skip
            pass
        
        self.plane_id = self.bc.loadURDF("plane.urdf")
        
        # Initialize drone agent directly - Using PHYSICS_TIME_STEP (1ms)
        self.drone = CrazyFlieBulletAgent(
            bc=self.bc,
            control_mode='AttitudeRate',  # Use exact class name
            time_step=PHYSICS_TIME_STEP,  # 0.001s (1ms) for 1000Hz physics
            aggregate_phy_steps=1,
            latency=0.0, # 0.015, #0.015,
            motor_time_constant= 0.02, # 0.080,
            motor_thrust_noise=0.00 # 0.05
        )
        
        # Initialize physics - Using PHYSICS_TIME_STEP (1ms)
        # Pass disturbance configuration directly to physics constructor
        self.physics = PyBulletPhysics(
            drone=self.drone,
            bc=self.bc,
            time_step=PHYSICS_TIME_STEP,  # 0.001s (1ms) for 1000Hz physics
            use_ground_effect=False,
            disturbance_enabled=disturbance_enabled,
            disturbance_strength=disturbance_strength,
            disturbance_verbose=disturbance_verbose
        )
        
        # State tracking
        self.last_quaternion = None
        
        # Reset to initial state
        self.reset()
        
    def get_state(self) -> np.ndarray:
        """Get current 17D Phoenix state matching the original CrazyFlieAgent format"""
        # Get state from drone agent (this already has the correct format)
        drone_state = self.drone.get_state()
        
        # The drone_state is already in the correct format:
        # [pos(3), quat(4), vel(3), ang_vel(3), last_action(4)] = 17D
        
        # Store quaternion for coordinate transformations
        self.last_quaternion = drone_state[3:7].copy()
        
        return drone_state
    
    def apply_control(self, control_actions: np.ndarray):
        """Apply control using Phoenix physics system - executes ONE physics step"""
        # Extract control actions from SMPC
        thrust_cmd = control_actions[0]      # SMPC thrust (N)
        roll_rate_cmd = control_actions[1]   # SMPC roll rate (rad/s) from predicted state
        pitch_rate_cmd = control_actions[2]  # SMPC pitch rate (rad/s) from predicted state
        yaw_rate_cmd = control_actions[3]    # SMPC yaw rate (rad/s) from predicted state
        
        # Angular rates are already in correct units (rad/s)
        roll_rate_scaled = np.clip(roll_rate_cmd, -MAX_ANGULAR_RATE, MAX_ANGULAR_RATE)
        pitch_rate_scaled = np.clip(pitch_rate_cmd, -MAX_ANGULAR_RATE, MAX_ANGULAR_RATE)
        yaw_rate_scaled = np.clip(yaw_rate_cmd, -MAX_ANGULAR_RATE, MAX_ANGULAR_RATE)
        
        # Construct Phoenix action
        phoenix_action = np.array([
            thrust_cmd,      # thrust (scaled to [-1, 1])
            roll_rate_scaled,   # roll_rate (rad/s)
            pitch_rate_scaled,  # pitch_rate (rad/s)
            yaw_rate_scaled     # yaw_rate (rad/s)
        ])
        
        # Apply action using Phoenix physics
        self.physics.step_forward(phoenix_action)
    
    def apply_control_multiple_steps(self, control_actions: np.ndarray, num_steps: int = None):
        """
        Apply control for multiple physics steps to match MPC planning interval
        
        Args:
            control_actions: [thrust, roll_rate, pitch_rate, yaw_rate] control action
            num_steps: Number of physics steps to execute (if None, calculated from MPC_PLANNING_INTERVAL)
        """
        if num_steps is None:
            # Calculate how many physics steps correspond to MPC planning interval
            num_steps = int(MPC_PLANNING_INTERVAL / PHYSICS_TIME_STEP)
        
        # Execute multiple physics steps with the same control action
        for step in range(num_steps):
            self.apply_control(control_actions)
    
    def reset(self, initial_pos: np.ndarray = None):
        """Reset Phoenix simulator to specified position"""
        # Use START_POSITION if no initial position is specified
        if initial_pos is None:
            initial_pos = START_POSITION.copy()
        
        # Reset drone position
        initial_rpy = np.array([0, 0, 0])
        
        # Reset drone state
        self.drone.xyz = initial_pos.copy()
        self.drone.rpy = initial_rpy.copy()
        self.drone.quaternion = self.bc.getQuaternionFromEuler(initial_rpy)
        self.drone.xyz_dot = np.zeros(3)
        self.drone.rpy_dot = np.zeros(3)
        
        # Reset PyBullet body
        self.bc.resetBasePositionAndOrientation(
            self.drone.body_unique_id,
            initial_pos,
            self.drone.quaternion
        )
        self.bc.resetBaseVelocity(
            self.drone.body_unique_id,
            np.zeros(3),
            np.zeros(3)
        )
        
        # Reset is complete - no need to call physics.reset() 