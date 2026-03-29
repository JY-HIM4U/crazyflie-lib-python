import numpy as np
from scipy.linalg import expm
from load_world import load_world

def discretize_linear_system(A, B, dt):
    """
    Performs exact discretization of a continuous-time linear system.

    Given a continuous-time linear system defined by:
    dx/dt = A * x + B * u

    This function computes the equivalent discrete-time system:
    x[k+1] = Ad * x[k] + Bd * u[k]

    Args:
        A (np.array): The continuous-time state matrix (n x n).
        B (np.array): The continuous-time input matrix (n x m).
        dt (float): The discretization time step.

    Returns:
        tuple: A tuple containing:
            - Ad (np.array): The discrete-time state matrix.
            - Bd (np.array): The discrete-time input matrix.
    """
    n = A.shape[0] # Number of states
    m = B.shape[1] # Number of inputs

    # Construct the augmented matrix for computing Ad and Bd simultaneously
    # The augmented matrix M is structured as:
    # M = [ A  B ]
    #     [ 0  0 ]
    # where 0 is an (m x n) zero matrix for the top block and (m x m) for the bottom block.
    # Note: The bottom-right zero matrix is m x m because the input part of the augmented
    # matrix has 'm' columns corresponding to 'u'.
 
    ##############3original########################
    augmented_matrix = np.block([
        [A, B],
        [np.zeros((m, n)), np.zeros((m, m))]
    ])

    # Compute the matrix exponential of M * delta_t
    # exp(M * delta_t) = [ exp(A*delta_t)  Integral(exp(A*tau) * B dtau) ]
    #                    [     0                        I               ]
    #
    # However, a common numerical method actually uses:
    # exp(M * delta_t) = [ Ad  Bd ]
    #                    [ 0   I  ]
    # where M = [A, B; 0, 0] and the bottom-right identity matrix 'I' refers to input part.
    # For a general solution, the augmented matrix should be:
    # M_aug = [ A  B ]
    #         [ 0  0 ]
    # where the bottom-left 0 is m x n and the bottom-right 0 is m x m.
    # Then expm(M_aug * dt) will yield:
    # [ exp(A*dt)  integral(exp(A*t)B dt from 0 to dt) ]
    # [    0                   I                       ]

    # The actual implementation for Bd from expm([A, B; 0, 0]) yields
    # exp(M*dt) = [ Ad  Bd ]
    #             [ 0   I_m]  where I_m is the identity matrix of size m.
    # The scipy.linalg.expm function works this way, extracting Ad and Bd from the top blocks.

    result_exp = expm(augmented_matrix * dt)

    # Extract Ad and Bd from the result_exp
    Ad = result_exp[0:n, 0:n]
    Bd = result_exp[0:n, n:n+m]


    # ##############new########################
    # Ad = np.eye(n) + A * dt
    # Bd = B * dt

    return Ad, Bd

def load_parameters():
    """
    Load parameters for the quadcopter simulation.
    This function sets up the parameters for dynamic programming, MPC, and simulation.
    It returns dictionaries containing these parameters.
    The parameters include:
        - Dynamic programming parameters (N, Delta_t, T)
        - MPC parameters (J, delta_t)
        - Simulation parameters (mass, simulation_steps_per_input, dt_sim)
    :return: dp_params, mpc_params, sim_params
    """
    # ================================== Most Important Parameters ==================================
    # Parameters used for Debugging:
    # N = 20  # Number of time steps in the dynamic programming horizon
    # M_a = 10  # Number of actions in the dynamic programming planner
    # world_file = "worlds/ra_30"  # File containing the world definition; implicitly defines the number of states in the DP planner
    # samples_for_each_state_action_pair = 10  # Number of samples for each state-action pair during the kernel computation

    # Parameters used for final paper:
    N = 40  # Number of time steps in the dynamic programming horizon
    M_a = 100  # Number of actions in the dynamic programming planner
    world_file = "worlds/ra_50"  # File containing the world definition; implicitly defines the number of states in the DP planner
    samples_for_each_state_action_pair = 200  # Number of samples for each state-action pair during the kernel computation
    Sigma_dt = np.zeros((12, 12))  # Process noise covariance matrix for simulation
    Sigma_dt[0:2, 0:2] = 5e-4 * np.eye(2)  # Process noise for position

    # ================================== Quadcopter Parameters ==================================
    # Updated based on cf21x_sys_eq.urdf specifications
    m = 0.027  # Mass of the quadcopter in kg (from URDF)
    kD = 0.1735  # Drag coefficient (keeping original)
    Ixx = 1.7e-5  # Moment of inertia around the x-axis in kg*m^2 (from URDF)
    Iyy = 1.7e-5  # Moment of inertia around the y-axis in kg*m^2 (from URDF)
    Izz = 2.9e-5  # Moment of inertia around the z-axis in kg*m^2 (from URDF)
    g = 9.81  # Gravitational acceleration in m/s^2
    T_max = 0.027 * 9.81 * 2.25  # Maximum thrust in N (mass * g * thrust2weight ratio from URDF)
    tau_x_max = 0.0084 # 0.1  # Maximum torque around the x-axis in N*m (estimated based on motor arm length)
    tau_y_max = 0.0084# 0.1  # Maximum torque around the y-axis in N*m
    tau_z_max = 0.0018 # 0.05  # Maximum torque around the z-axis in N*m

    # A =
    # [ 0   I3   0     0  ]
    # [ 0 -kD/m*I3 A1   0  ]
    # [ 0   0     0    I3 ]
    # [ 0   0     0     0  ]
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))

    # A_c = np.block([
    #     [Z3, I3, Z3, Z3],
    #     [Z3, -(kD/m) * I3,
    #      # A1 block starts here
    #      np.array([
    #         [0, g, 0],
    #         [-g, 0, 0],
    #         [0, 0, 0]
    #      ]), # A1 block ends here
    #      Z3],
    #     [Z3, Z3, Z3, I3],
    #     [Z3, Z3, Z3, Z3]
    # ])

    # # B =
    # # [   0     ]
    # # [   B1    ]
    # # [   0     ]
    # # [   B2    ]
    # Z_3x4 = np.zeros((3, 4))
    # I_3x4= np.zeros((3, 4))
    # I_3x4[1:4,1:4] = np.eye(3)
    # B_c = np.block([
    #     [Z_3x4],
    #     # B1 block starts here
    #     [np.array([
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0],
    #         [(1/m) * T_max, 0, 0, 0]
    #     ])], # B1 block ends here
    #     [Z_3x4],
    #     # B2 block starts here
    #     [np.array([
    #         [0, (1/Ixx) * tau_x_max, 0, 0],
    #         [0, 0, (1/Iyy) * tau_y_max, 0],
    #         [0, 0, 0, (1/Izz) * tau_z_max]
    #     ])] # B2 block ends here
    # ])
    Z_3x4 = np.zeros((3, 4))
    I_3x4= np.zeros((3, 4))
    I_3x4[0:3,1:4] = np.eye(3)
    A_c = np.block([
        [Z3, I3, Z3],
        [Z3, -(kD/m) * I3,
         # A1 block starts here
         np.array([
            [0, g, 0],
            [-g, 0, 0],
            [0, 0, 0]
         ]) # A1 block ends here
         ],
        [Z3, Z3, Z3]
    ])

    # B =
    # [   0     ]
    # [   B1    ]
    # [   0     ]
    # [   B2    ]
    B_c = np.block([
        [Z_3x4],
        # B1 block starts here
        [np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [(1/m) * T_max, 0, 0, 0]
        ])], # B1 block ends here
        [I_3x4]        
        
    ])



    # ================================== Compressor Parameters ==================================
    # safe_set, target_set = load_world(world_file) # Load the safe and target sets from the world
    safe_set, target_set = [[]], [[]]
    hypercube_size = np.array([.1,.1])  # Size of the hypercube in each dimension
    com_params = {
        'safe_set': safe_set,  # Safe set for the compressor
        'target_set': target_set,  # Target set for the compressor
        'hypercube_size': hypercube_size,  # Size of the hypercube in each dimension
    }



    # ================================== Model Predictive Control Parameters ==================================
    # J and delta_t must match fly_LQR usage: delta_t == MPC_PLANNING_INTERVAL, J == mpc_horizon (RL_DECISION_INTERVAL/delta_t)
    J = 20  # Number of steps in the MPC/LQT prediction horizon
    delta_t = 0.025  # Time step duration for the MPC in seconds (must equal MPC_PLANNING_INTERVAL in RL_smpc_config / fly_LQR)
    A_delta_t, B_delta_t = discretize_linear_system(A_c, B_c, delta_t)

    MPC_Q = np.zeros((9, 9))  # Q Matrix of the MPC objective
    MPC_Q[0:3, 0:3] = 1*np.eye(3)
    MPC_Q[3:, 3:] = 1 * np.eye(6)
    MPC_Q[8:9, 8:9] = 1 * np.eye(1)
    MPC_R = 0.01*np.eye(4)  # R Matrix of the MPC objective

    A_ineq = np.array([#[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       #[ 1,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       #[ 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       #[ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       #[ 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       #[ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0],
                       [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0],
                       [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                       [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                       ])

    b_ineq = np.array( [#0,0,    # x limits
                        #0,0,    # y limits
                        #.5,.5,  # z limits
                        1,1,    # vx limits (relaxed for better performance)
                        1,1,    # vy limits (relaxed for better performance)
                        1,1,    # vz limits (relaxed to prevent saturation)
                        0.5,0.5,    # roll angle limits (reduced for stability)
                        0.5,0.5,    # pitch angle limits (reduced for stability)
                        0.2,0.2,    # yaw angle limits
                        6,6,    # roll rate limits (reduced for stability)
                        6,6,    # pitch rate limits (reduced for stability)
                        6,6,])  # yaw rate limits (reduced for stability)

    A_ineq = np.array([
                       [ 0, 0, 0,-1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0,-1, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0,-1, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0,-1, 0, 0],
                       [ 0, 0, 0, 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 0, 0, 0,-1, 0],
                       [ 0, 0, 0, 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 0, 0, 0, 0, 0,-1],
                       [ 0, 0, 0, 0, 0, 0, 0, 0, 1],
                       ])

    b_ineq = np.array( [#0,0,    # x limits
                        #0,0,    # y limits
                        #.5,.5,  # z limits
                        1,1,    # vx limits (relaxed for better performance)
                        1,1,    # vy limits (relaxed for better performance)
                        1,1,    # vz limits (relaxed to prevent saturation)
                        0.5,0.5,    # roll angle limits (reduced for stability)
                        0.5,0.5,    # pitch angle limits (reduced for stability)
                        0.2,0.2
                        ])  # yaw rate limits (reduced for stability)

    u_ss_thrust = m * g / T_max  # Steady-state thrust input

    Au_ineq = np.array([[ 1, 0, 0, 0],
                        [-1, 0, 0, 0],
                        [ 0, 1, 0, 0],
                        [ 0,-1, 0, 0],
                        [ 0, 0, 1, 0],
                        [ 0, 0,-1, 0],
                        [ 0, 0, 0, 1],
                        [ 0, 0, 0,-1]])
    bu_ineq = np.array([1-u_ss_thrust,u_ss_thrust,        #[1-u_ss_thrust, u_ss_thrust,       # Thrust limits
                        6, 6,   # Roll torque limits (reduced for smaller drone)
                        6, 6,   # Pitch torque limits (reduced for smaller drone)
                        6, 6]) # Yaw torque limits (reduced for smaller drone)

    # Terminal constraint for the velocity, angle and angular velocity
    r_underline = np.array([
        # 0,        # Terminal z position constraint
        # 0, 0, 0,  # Terminal velocity constraints (vx, vy, vz)
        0, 0, 0   # Terminal angular velocity constraints (roll rate, pitch rate, yaw rate)
    ])

    mpc_params = {
        'J': J, # Execution horizon of MPC
        'delta_t': delta_t,  # Time step duration for the MPC in seconds
        'A': A_delta_t,  # Discretized state matrix for MPC
        'B': B_delta_t,  # Discretized input matrix for MPC
        'MPC_Q': MPC_Q,  # Q Matrix of the MPC objective
        'MPC_R': MPC_R,  # R Matrix of the MPC objective
        'A_ineq': A_ineq,  # State constraint matrix for MPC
        'b_ineq': b_ineq,  # Right hand side of the state constraint for time-steps 0,...,N-1
        'A_ineq_u': Au_ineq,  # Input constraint matrix for MPC
        'bu_ineq': bu_ineq,  # Right hand side of the input constraint for time-steps 0,...,N-1
        'r_underline': r_underline,  # Terminal constraint for the velocity, angle and angular velocity
    }

    # ================================== Dynamic Programming Parameters ==================================
    Delta_t = J*delta_t # seconds, total time for every command execution
    x_lower_0 = np.zeros((9,), dtype=np.float32)  # Initial state for the dynamic programming planner
    dp_params = {
        'N': N,  # Number of time steps in the dynamic programming horizon
        'Delta_t': Delta_t,  # Time step for dynamic programming
        'T': N*Delta_t,  # Total time for the mission in seconds
        'command_type': 'reference_and_Q', # Type of command space
        'INCLUDE_VELOCITY_IN_PLANNER': True,  # Whether to include velocity in the planner
        'samples_for_each_state_action_pair': samples_for_each_state_action_pair, #100,  # Number of samples for each state-action pair during the kernel computation
        'M_a': M_a, #50,  # Number of actions in the dynamic programming planner
        'x_lower_0': x_lower_0,  # Initial state for the dynamic programming planner in the lower subspace
        'LOAD_LAST_KERNEL': False,  # Whether to load the last computed kernel
    }

    # ================================== Simulation Parameters ==================================
    simulation_steps_per_input = 10  # Number of simulation steps per control input
    dt_sim = mpc_params['delta_t'] / simulation_steps_per_input
    A_dt, B_dt = discretize_linear_system(A_c, B_c, dt_sim)
    sim_params = {
        'simulation_steps_per_input': simulation_steps_per_input,  # Number of simulation steps per control input
        'dt_sim': dt_sim,  # Total simulation time in seconds
        'A': A_dt,  # Discretized state matrix for simulation
        'B': B_dt,  # Discretized input matrix for simulation
        'Sigma': Sigma_dt,  # Process noise covariance matrix for simulation
    }
    return com_params, dp_params, mpc_params, sim_params