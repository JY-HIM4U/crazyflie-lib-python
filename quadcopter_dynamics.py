import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy # For deepcopying state dictionaries
from parameters import load_parameters

def quadcopter_dynamics_single_step_linear(state, inputs, sim_params):
    """
    This function simulates the linear dynamics of a quadcopter for a single time step
    based on the provided linear model, using direct construction of the large A and B matrices.

    Args:
        state (dict): A dictionary containing the current state of the quadcopter.
                      Expected keys:
                      'position': np.array([x, y, z]), in meters
                      'velocity': np.array([vx, vy, vz]), in m/s
                      'orientation': np.array([phi, theta, psi]), Euler angles in radians
                      'angular_velocity': np.array([p, q, r]), body-frame angular velocities in rad/s
        inputs (dict): A dictionary containing the control inputs (deviation from steady state).
                       Expected keys:
                       'thrust': float, total thrust force in Newtons (T)
                       'tau_x': float, torque about x-axis (roll)
                       'tau_y': float, torque about y-axis (pitch)
                       'tau_z': float, torque about z-axis (yaw)
        sim_params (dict): A dictionary containing simulation parameters.
                           Expected keys:
                           'dt': float, The time step for the simulation, in seconds.
                           'm': float, Mass of the quadcopter, in kg.
                           'kD': float, Drag coefficient (for linear drag approx).
                           'Ixx': float, Moment of inertia about x-axis.
                           'Iyy': float, Moment of inertia about y-axis.
                           'Izz': float, Moment of inertia about z-axis.
                           'T_max': float, Maximum possible thrust (for B1 matrix scaling).
                           'tau_x_max': float, Maximum possible x-axis torque (for B2 matrix scaling).
                           'tau_y_max': float, Maximum possible y-axis torque (for B2 matrix scaling).
                           'tau_z_max': float, Maximum possible z-axis torque (for B2 matrix scaling).
                           'us_thrust': float, Steady-state thrust (T_s) for the linearization point.

    Returns:
        dict: The updated state of the quadcopter.
    """
    # If state and inputs are dictionaries and not numpy array, unpack them
    if isinstance(state, dict):
        # Unpack state variables
        position = state['position']        # px, py, pz
        velocity = state['velocity']        # vx, vy, vz
        orientation = state['orientation']  # phi, theta, psi
        angular_velocity = state['angular_velocity'] # p, q, r

        # Unpack inputs (assuming these are the actual values, not normalized commands)
        thrust = inputs['thrust']
        tau_x = inputs['tau_x']
        tau_y = inputs['tau_y']
        tau_z = inputs['tau_z']

        # 1. Define the state vector x (12x1)
        # x = [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]^T
        x_vec = np.concatenate([position, velocity, orientation, angular_velocity])
        # 2. Define the input vector (u - u_s) (4x1)
        # The linear model expects deviations from a steady-state input (u_s).
        # Here, u_s is assumed to only affect thrust (T_s), and torques are deviations from zero.
        u_vec = np.array([
            thrust,
            tau_x,
            tau_y,
            tau_z
        ])
    else:
        x_vec = state
        u_vec = inputs

    # Unpack simulation parameters
    dt = sim_params['dt_sim']
    A_linear = sim_params['A']  # Large A matrix (12x12)
    B_linear = sim_params['B']  # Large B matrix (12x4)
    Sigma = sim_params['Sigma']  # Noise covariance matrix (12x12)

    # Constants
    g = 9.81  # Acceleration due to gravity (m/s^2)

    # ------------------ Compute Dynamics Here ------------------




    # 5. Calculate the derivative of the state vector: x_dot = A_linear @ x_vec + B_linear @ (u - u_s)
    new_x_vec = A_linear @ x_vec + B_linear @ u_vec + np.random.multivariate_normal(np.zeros(12), Sigma)

    # # 7. Unpack the new_x_vec into the state dictionary
    # new_position = new_x_vec[0:3]
    # new_velocity = new_x_vec[3:6]
    # new_orientation = new_x_vec[6:9]
    # new_angular_velocity = new_x_vec[9:12]
    #
    # # Create and return the new state dictionary
    # new_state = {
    #     'position': new_position,
    #     'velocity': new_velocity,
    #     'orientation': new_orientation,
    #     'angular_velocity': new_angular_velocity
    # }

    return new_x_vec

# def quadcopter_dynamics_single_step(state, inputs, sim_params):
#     """
#     This function was generated using Gemini. Simulates the nonlinear dynamics of a quadcopter for a single time step.
#
#     Args:
#         state (dict): A dictionary containing the current state of the quadcopter.
#                       Expected keys:
#                       'position': np.array([x, y, z]), in meters
#                       'velocity': np.array([vx, vy, vz]), in m/s
#                       'orientation': np.array([phi, theta, psi]), Euler angles in radians
#                       'angular_velocity': np.array([p, q, r]), body-frame angular velocities in rad/s
#         inputs (dict): A dictionary containing the control inputs.
#                        Expected keys:
#                        'thrust': float, total thrust force in Newtons
#                        'roll_rate_cmd': float, commanded roll rate in rad/s
#                        'pitch_rate_cmd': float, commanded pitch rate in rad/s
#                        'yaw_rate_cmd': float, commanded yaw rate in rad/s
#         dt (float): The time step for the simulation, in seconds.
#         m (float): Mass of the quadcopter, in kg.
#         I (np.array): 3x3 inertia matrix of the quadcopter.
#
#     Returns:
#         dict: The updated state of the quadcopter.
#     """
#
#     # Unpack state variables
#     position = state['position']
#     velocity = state['velocity']
#     orientation = state['orientation']
#     angular_velocity = state['angular_velocity']
#
#     # Unpack the angular velocity components
#     p, q, r = angular_velocity
#
#     # Unpack inputs
#     thrust = inputs['thrust']
#     p_cmd, q_cmd, r_cmd = inputs['roll_rate_cmd'], inputs['pitch_rate_cmd'], inputs['yaw_rate_cmd']
#
#     # Constants
#     g = 9.81  # Acceleration due to gravity (m/s^2)
#
#     # ------------------ Translational Dynamics ------------------
#
#     # Rotation matrix from body frame to world frame (from Euler angles)
#     # This matrix is used to rotate forces from the body frame to the inertial frame.
#     phi, theta, psi = orientation
#     rot_matrix = R.from_euler('zyx', [psi, theta, phi]).as_matrix()
#
#     # Total force in the world frame
#     # Thrust is in the body's z-direction, so it's a vector [0, 0, thrust] in the body frame.
#     # We rotate this vector to the world frame and add gravity.
#     thrust_body = np.array([0, 0, thrust])
#     thrust_world = rot_matrix @ thrust_body
#
#     gravity_world = np.array([0, 0, -sim_params['mass'] * g])
#
#     F_total = thrust_world + gravity_world
#
#     # Update position and velocity (using a simple Euler integration)
#     acceleration = F_total / sim_params['mass']
#     velocity += acceleration * sim_params['dt_sim']
#     position += velocity * sim_params['dt_sim']
#
#     # ------------------ Rotational Dynamics ------------------
#
#     # We are directly using the commanded angular rates as inputs.
#     # A more complete model would include a rate controller (e.g., PID)
#     # that calculates torques to achieve these commanded rates.
#     # Here, we assume the inputs are the angular velocities themselves.
#
#     # Update orientation (Euler angles)
#     # Conversion from body angular velocities (p, q, r) to Euler angle rates (phi_dot, theta_dot, psi_dot)
#     # This is a key part of the nonlinear dynamics.
#     phi_dot = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
#     theta_dot = q * np.cos(phi) - r * np.sin(phi)
#     psi_dot = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)
#
#     # Simple Euler integration for orientation
#     orientation += np.array([phi_dot, theta_dot, psi_dot]) * sim_params['dt_sim']
#
#     # Update angular velocity
#     # For this simplified model, we assume the drone can perfectly achieve the commanded rates.
#     # In a more realistic model, this would be a control output, and the drone's
#     # actual angular acceleration would be calculated from the torques and the inertia matrix.
#     angular_velocity = np.array([p_cmd, q_cmd, r_cmd])
#
#     # Create and return the new state dictionary
#     new_state = {
#         'position': position,
#         'velocity': velocity,
#         'orientation': orientation,
#         'angular_velocity': angular_velocity
#     }
#
#     return new_state

def quadcopter_dynamics_multi_step(state, inputs, sim_params):
    """
    Simulates the nonlinear dynamics of a quadcopter for multiple time steps.

    Args:
        state (dict): The initial state of the quadcopter.
        inputs (dict): The control inputs for each step.
        dt (float): The time step for the simulation, in seconds.
        steps (int): Number of steps to simulate.
        m (float): Mass of the quadcopter, in kg.

    Returns:
        list: A list of states for each time step.
    """
    state_sequence = []
    for _ in range(sim_params['simulation_steps_per_input']):
        next_state = quadcopter_dynamics_single_step_linear(state, inputs, sim_params)
        state = next_state
        state_sequence.append(copy.deepcopy(state))

    return next_state, state_sequence

def test_quadcopter_dynamics():

    # Initial state
    initial_state = np.array([0.0, 0.0, 0.0,  # Position (x, y, z)
                                 0.0, 0.0, 0.0,  # Velocity (vx, vy, vz)
                                    0.0, 0.0, 0.0,  # Orientation (phi, theta, psi)
                                    0.0, 0.0, 0.0])  # Angular velocity (p, q, r)

    # Load parameters
    com_params, dp_params, mpc_params, sim_params = load_parameters()

    # List to store state history for plotting/analysis
    state_history = [copy.deepcopy(initial_state)]
    current_state = initial_state

    # Control inputs for the entire simulation
    # Here, we'll give a constant thrust and a pitch command to move forward.
    # A real controller would generate these commands based on a desired trajectory.

    control_inputs = np.array([0.05,  # Thrust (N)
                                -0.0003,  # Tau_x (roll torque)
                                -0.0001,  # Tau_y (pitch torque)
                                -0.01])   # Tau_z (yaw torque)

    # Simulation loop
    for _ in range(mpc_params['J']):
        # Calculate the next state
        next_state, dummy = quadcopter_dynamics_multi_step(current_state, control_inputs, sim_params)

        # Update current state and store history
        current_state = next_state
        state_history.append(copy.deepcopy(current_state))

    # Print final position and orientation
    print(f"Final Position: {current_state[0:3]}")
    print(f"Final Orientation (Euler): {np.degrees(current_state[6:9])}")

    # You can plot state_history for a full visualization of the trajectory.
    positions = np.array([state[0:3] for state in state_history])
    # Plot the 3D trajectory
    fig = plt.figure(0)
    plt.subplot(111, projection='3d')

    x_data = positions[:, 0]
    y_data = positions[:, 1]
    z_data = positions[:, 2]

    plt.plot(x_data, y_data, z_data, label='Quadcopter Trajectory')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Quadcopter 3D Trajectory')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    fig = plt.figure(1)
    plt.subplot(411)
    plt.plot([state[0:3][0] for state in state_history], label='X Position')
    plt.plot([state[0:3][1] for state in state_history], label='Y Position')
    plt.plot([state[0:3][2] for state in state_history], label='Z Position')
    plt.xlabel('Time Step')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.subplot(412)
    plt.plot([state[3:6][0] for state in state_history], label='X Velocity')
    plt.plot([state[3:6][1] for state in state_history], label='Y Velocity')
    plt.plot([state[3:6][2] for state in state_history], label='Z Velocity')
    plt.xlabel('Time Step')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.subplot(413)
    plt.plot([np.degrees(state[6:9][0]) for state in state_history], label='Phi (Roll)')
    plt.plot([np.degrees(state[6:9][1]) for state in state_history], label='Theta (Pitch)')
    plt.plot([np.degrees(state[6:9][2]) for state in state_history], label='Psi (Yaw)')
    plt.xlabel('Time Step')
    plt.ylabel('Orientation (degrees)')
    plt.legend()
    plt.subplot(414)
    plt.plot([np.degrees(state[9:12][0]) for state in state_history], label='Phi Velocity (Roll)')
    plt.plot([np.degrees(state[9:12][1]) for state in state_history], label='Theta Velocity (Pitch)')
    plt.plot([np.degrees(state[9:12][2]) for state in state_history], label='Psi Velocity (Yaw)')
    plt.xlabel('Time Step')
    plt.ylabel('Angular Velocity (degrees/s)')
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)
