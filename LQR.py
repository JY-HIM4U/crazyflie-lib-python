import mosek
import daqp
import cvxpy as cp
import numpy as np
from quadcopter_dynamics import quadcopter_dynamics_multi_step
from parameters import load_parameters
import matplotlib
# matplotlib.use('Agg')  # Commented out to allow interactive plotting
import matplotlib.pyplot as plt
# Configure matplotlib for better display
# plt.ion()  # Turn on interactive mode
import osqp
from ctypes import *
import ctypes.util
import gurobipy
from numba import njit, prange

# -------------------------------
# Linear-Quadratic Tracking (LQT)
# -------------------------------
def compute_lqt_gains(A: np.ndarray,
                      B: np.ndarray,
                      Q: np.ndarray,
                      R: np.ndarray,
                      r_seq: np.ndarray,
                      Q_terminal: np.ndarray | None = None):
    """
    Compute time-varying LQT gains over a finite horizon using the recursion:
        K_k = F_k P_{k+1} A
        P_k = Q + A^T P_{k+1} (A - B K_k)
        s_k = -Q r_k + (A - B K_k)^T s_{k+1}
        F_k = (R + B^T P_{k+1} B)^{-1} B^T
    with terminal conditions P_N = Q_N, s_N = -Q_N r_N.

    Args:
        A: (nx, nx) discrete-time system matrix
        B: (nx, nu) input matrix
        Q: (nx, nx) state tracking weight (for k = 0..N-1)
        R: (nu, nu) input weight (for k = 0..N-1)
        r_seq: (nx, N+1) reference sequence r_0..r_N
        Q_terminal: (nx, nx) terminal weight Q_N (defaults to Q if None)

    Returns:
        dict with:
          - 'K_seq': list of (nu, nx) feedback gains for k=0..N-1
          - 'F_seq': list of (nu, nx) matrices for k=0..N-1
          - 'P_seq': list of (nx, nx) cost-to-go matrices for k=0..N
          - 's_seq': list of (nx,) vectors for k=0..N
    """
    # Ensure numeric, float64, contiguous inputs to avoid BLAS/ABI surprises
    A = np.ascontiguousarray(np.asarray(A, dtype=np.float64))
    B = np.ascontiguousarray(np.asarray(B, dtype=np.float64))
    Q = np.ascontiguousarray(np.asarray(Q, dtype=np.float64))
    R = np.ascontiguousarray(np.asarray(R, dtype=np.float64))
    r_seq = np.ascontiguousarray(np.asarray(r_seq, dtype=np.float64))
    if Q_terminal is not None:
        Q_terminal = np.ascontiguousarray(np.asarray(Q_terminal, dtype=np.float64))

    nx = A.shape[0]
    nu = B.shape[1]
    assert r_seq.shape[0] == nx, "r_seq must have shape (nx, N+1)"
    N = r_seq.shape[1] - 1
    QN = Q if Q_terminal is None else Q_terminal

    P_seq: list[np.ndarray] = [None] * (N + 1)  # type: ignore
    s_seq: list[np.ndarray] = [None] * (N + 1)  # type: ignore
    K_seq: list[np.ndarray] = [None] * N        # type: ignore
    F_seq: list[np.ndarray] = [None] * N        # type: ignore

    # Terminal conditions
    P_seq[N] = QN
    s_seq[N] = -QN @ r_seq[:, N]

    Bt = B.T
    At = A.T

    for k in range(N - 1, -1, -1):
        P_next = P_seq[k + 1]
        # F_k = (R + B^T P_{k+1} B)^{-1} B^T
        M = R + Bt @ P_next @ B
        # Solve instead of explicit inverse for numerical stability
        F_k = np.linalg.solve(M, Bt)
        K_k = F_k @ P_next @ A
        Acl = A - B @ K_k
        P_k = Q + At @ P_next @ Acl
        s_k = -Q @ r_seq[:, k] + Acl.T @ s_seq[k + 1]

        F_seq[k] = F_k
        K_seq[k] = K_k
        P_seq[k] = P_k
        s_seq[k] = s_k

    return {
        'K_seq': K_seq,
        'F_seq': F_seq,
        'P_seq': P_seq,
        's_seq': s_seq
    }

def lqt_control_step(K_k: np.ndarray,
                     F_k: np.ndarray,
                     s_next: np.ndarray,
                     x_k: np.ndarray) -> np.ndarray:
    """
    Compute the LQT control at step k:
        u_hat_k = -K_k x_k - F_k s_{k+1}
    """
    K_k = np.ascontiguousarray(np.asarray(K_k, dtype=np.float64))
    F_k = np.ascontiguousarray(np.asarray(F_k, dtype=np.float64))
    s_next = np.ascontiguousarray(np.asarray(s_next, dtype=np.float64)).reshape(-1)
    x_k = np.ascontiguousarray(np.asarray(x_k, dtype=np.float64)).reshape(-1)
    return -K_k @ x_k - F_k @ s_next

# # 1. Function to define the CVXPY problem structure (call this once at the beginning)
# def define_mpc_problem(mpc_params):
#     """
#     Defines the symbolic structure of the Model Predictive Control (MPC) Quadratic Program.
#     This function should be called only once to set up the CVXPY problem.
#
#     Args:
#         mpc_params (dict): Contains the MPC parameters including state and input matrices,
#                            constraints, and prediction horizon.
#
#     Returns:
#         tuple: A tuple containing:
#             - prob (cp.Problem): The CVXPY problem object.
#             - x (cp.Variable): Reference to the state trajectory variable.
#             - u (cp.Variable): Reference to the input trajectory variable.
#             - epsilon (cp.Variable): Reference to the slack variable.
#             - x0_param (cp.Parameter): Reference to the initial state parameter.
#             - Q_param (cp.Parameter): Reference to the state cost matrix parameter.
#             - R_param (cp.Parameter): Reference to the input cost matrix parameter.
#             - x_ref_param (cp.Parameter): Reference to the reference trajectory parameter.
#             - b_ineq_param (cp.Parameter): Reference to the state constraint RHS parameter.
#             - bu_ineq_param (cp.Parameter): Reference to the input constraint RHS parameter.
#             - r_underline_param (cp.Parameter): Reference to the terminal constraint target parameter.
#     """
#     A_dt = mpc_params['A']
#     B_dt = mpc_params['B']
#     A_ineq = mpc_params['A_ineq']
#     A_ineq_u = mpc_params['A_ineq_u']
#     prediction_horizon = mpc_params['J']
#
#     nx = A_dt.shape[1]
#     nu = B_dt.shape[1]
#     nc_state = A_ineq.shape[0]
#     nc_input = A_ineq_u.shape[0]
#
#     # Define variables for the FULL prediction horizon.
#     # These dimensions remain constant across all MPC iterations.
#     x = cp.Variable((nx, prediction_horizon + 1), name='x')
#     u = cp.Variable((nu, prediction_horizon), name='u')
#     epsilon = cp.Variable(prediction_horizon + 1, name='epsilon') # Slack variable for soft constraints
#
#     # Define parameters for all dynamic inputs.
#     # Their values will be updated in each MPC iteration.
#     x0_param = cp.Parameter(nx, name='x0_param') # Initial state for the current active horizon
#     Q_param = cp.Parameter((nx, nx), PSD=True, name='Q_param') # State cost matrix
#     R_param = cp.Parameter((nu, nu), PSD=True, name='R_param') # Input cost matrix
#     # x_ref_param will hold the entire reference trajectory for the full prediction horizon
#     x_ref_param = cp.Parameter((nx, prediction_horizon + 1), name='x_ref_param')
#     b_ineq_param = cp.Parameter(nc_state, name='b_ineq_param') # RHS of state constraints
#     bu_ineq_param = cp.Parameter(nc_input, name='bu_ineq_param') # RHS of input constraints
#     # Terminal constraint target. Defined for the full state (nx) for fixed structure.
#     # If only a subset of states are constrained, the user must set the irrelevant parts
#     # of r_underline_param.value to the current value of x[irrelevant_idx, prediction_horizon].
#     r_underline_param = cp.Parameter(10, name='r_underline_param')
#
#     # Define the cost function for the FULL prediction horizon.
#     cost = 0
#     for t in range(prediction_horizon):
#         cost += cp.quad_form(x[:, t] - x_ref_param[:, t], Q_param) + cp.quad_form(u[:, t], R_param)
#     # Terminal cost
#     cost += cp.quad_form(x[:, prediction_horizon] - x_ref_param[:, prediction_horizon], Q_param)
#     # Soft constraint penalty
#     cost += cp.quad_form(epsilon, 100000.0 * np.eye(prediction_horizon + 1))
#     cost += 100.0 * cp.sum(epsilon)
#
#     # Define constraints for the FULL prediction horizon.
#     # The initial state constraint for the *active* horizon will be set by fixing x.value.
#     constraints = []
#     # Initial state constraint for the *first* state in the fixed horizon
#     constraints.append(x[:, 0] == x0_param)
#     for t in range(prediction_horizon):
#         # Soft chance constraints on state
#         constraints.append(A_ineq @ x[:, t] <= b_ineq_param + epsilon[t] * np.ones(nc_state))
#         # Dynamics constraint
#         constraints.append(x[:, t + 1] == A_dt @ x[:, t] + B_dt @ u[:, t])
#         # Hard input constraints
#         constraints.append(A_ineq_u @ u[:, t] <= bu_ineq_param)
#         # Non-negative slack variable
#         constraints.append(epsilon[t] >= 0)
#     constraints.append(epsilon[prediction_horizon] >= 0)
#
#     # Define terminal constraint for the full state at the end of the prediction horizon.
#     # This structure is fixed.
#     constraints.append(x[2:12, prediction_horizon] == r_underline_param + epsilon[prediction_horizon] * np.ones(10))
#
#     # Form the CVXPY problem
#     prob = cp.Problem(cp.Minimize(cost), constraints)
#
#     # Return the problem object and references to its variables and parameters
#     # for easy access in the QP function.
#     return prob, x, u, epsilon, x0_param, Q_param, R_param, x_ref_param, b_ineq_param, bu_ineq_param, r_underline_param
#
# # 2. Modified QP function (call this repeatedly in your control loop)
# def QP_warm_start(problem_data, dp_params, mpc_params, x0, Q, R, x_ref, current_time_step, DEBUG_MODE_SMPC = False):
#     """
#     Solves a Quadratic Program (QP) for Model Predictive Control (MPC) using a pre-defined CVXPY problem.
#     This function updates parameters for the current MPC iteration.
#
#     Args:
#         problem_data (tuple): Contains (prob, x, u, epsilon, x0_param, Q_param, R_param, x_ref_param, b_ineq_param, bu_ineq_param, r_underline_param)
#                               as returned by define_mpc_problem.
#         dp_params (dict): Contains dynamic planner parameters.
#         mpc_params (dict): Contains the MPC parameters including state and input matrices, constraints, etc.
#         x0 (array): Initial state of the system for the *current* time step. This will be used as x[:, 0] in the optimization.
#         Q (array): State cost matrix.
#         R (array): Input cost matrix.
#         x_ref (array): Reference state to track for the *entire* prediction horizon.
#                        Must be of shape (nx, prediction_horizon + 1).
#         current_time_step (int): Current time-step in the overall simulation (used for debugging/logging, not directly for problem structure).
#         DEBUG_MODE_SMPC (bool): Debug mode flag for additional output.
#     Returns:
#         u_sol (array): Optimal control inputs.
#         x_sol (array): Optimal state trajectory.
#         epsilon_sol (array): Slack variables for soft constraints.
#     """
#
#     # Unpack the problem data
#     prob, x, u, epsilon, x0_param, Q_param, R_param, x_ref_param, b_ineq_param, bu_ineq_param, r_underline_param = problem_data
#     x0 = np.array(x0).reshape(-1, 1)
#     # Update parameter values for the current iteration
#     # x0_param now represents the *current* system state, which is the starting point for the optimization.
#     x0_param.value = np.squeeze(x0)
#     Q_param.value = Q
#     R_param.value = R
#     # x_ref must be provided for the full prediction horizon (J+1 steps)
#     x_ref_param.value = x_ref
#     b_ineq_param.value = mpc_params['b_ineq']
#     bu_ineq_param.value = mpc_params['bu_ineq']
#
#     # Assign the terminal constraint target.
#     # Ensure mpc_params['r_underline'] is correctly sized (10 elements as per your constraint).
#     r_underline_param.value = mpc_params['r_underline']
#
#     # Solve the problem.
#     # warm_start=True is crucial for speed in iterative solves, allowing the solver
#     # to use the previous solution as an initial guess.
#     prob.solve(solver=cp.MOSEK, warm_start=True)
#
#     # if DEBUG_MODE_SMPC:
#     #     if prob.status == cp.OPTIMAL:
#     #         print("The problem is feasible and optimal.")
#     #         print(f"The optimal value of the objective is {prob.value}.")
#     #     elif prob.status == cp.INFEASIBLE:
#     #         print("The problem is infeasible.")
#     #     elif prob.status == cp.UNBOUNDED:
#     #         print("The problem is unbounded.")
#     #     else:
#     #         print(f"The problem status is {prob.status}.")
#     #     epsilon_sol = epsilon.value
#
#     # Extract the solution for the *entire* horizon.
#     u_sol = u.value
#     x_sol = x.value
#     epsilon_sol = epsilon.value
#
#     return u_sol, x_sol, epsilon_sol

def QP_test():
    H = np.array([[1, 0], [0, 1]], dtype=c_double)
    f = np.array([1, 1], dtype=c_double)
    A = np.array([[1, 2], [1, -1]], dtype=c_double)
    bupper = np.array([1, 2, 3, 4], dtype=c_double)
    blower = np.array([-1, -2, -3, -4], dtype=c_double)
    sense = np.array([0, 0, 0, 0], dtype=c_int)
    (xstar, fval, exitflag, info) = daqp.solve(H, f, A, bupper, blower, sense)


# @njit
def _build_qp_matrices_core(
        include_velocity_in_planner,
        A_dt, B_dt, A_ineq, b_ineq_static, A_ineq_u, bu_ineq_static, r_underline_static,
        N, nx, nu, nc_state, nc_input, nz,
        x0_flat, Q, R, x_ref, current_time_step,
        idx_term, len_idx_term,
        H_out, f_out, A_ineq_combined_out, b_ineq_combined_out
):
    """
    Numba-optimized core function to fill QP matrices in-place.
    This function assumes output arrays are pre-allocated by the wrapper.
    """

    # --- Construct H (Hessian) matrix for 0.5 * z^T H z ---
    # H_out is already initialized to zeros by the wrapper
    # Blocks for control inputs (u_0 to u_{N-1})
    for t in range(N):
        u_idx_start = t * nu
        for i in range(nu):
            for j in range(nu):
                H_out[u_idx_start + i, u_idx_start + j] = 2 * R[i, j]

    # Blocks for state trajectories (x_0 to x_N)
    for t in range(N + 1):
        x_idx_start = nu * N + t * nx
        for i in range(nx):
            for j in range(nx):
                H_out[x_idx_start + i, x_idx_start + j] = 2 * Q[i, j]

    # --- Construct f (linear term) vector ---
    # f_out is already initialized to zeros by the wrapper
    for t in range(N + 1):
        x_idx_start = nu * N + t * nx
        x_ref_slice = x_ref[:, current_time_step + t]

        # Manual matrix-vector multiplication for -2 * Q @ x_ref_slice
        for i in range(nx):
            sum_val = 0.0
            for j in range(nx):
                sum_val += Q[i, j] * x_ref_slice[j]
            f_out[x_idx_start + i] = -2 * sum_val

    # --- Temporary arrays for Equality Constraints (A_eq, b_eq) ---
    # These are created inside njit as their sizes depend on N, which is a function argument.
    # Numba allows np.zeros if dimensions are derived from function arguments.
    nc_eq = nx + nx * N + len_idx_term
    A_eq_temp = np.zeros((nc_eq, nz), dtype=np.float64)
    b_eq_temp = np.zeros(nc_eq, dtype=np.float64)

    row_offset_eq = 0

    # 1. Initial state constraint: x_0 == x0
    x0_col_start = nu * N
    for i in range(nx):
        A_eq_temp[row_offset_eq + i, x0_col_start + i] = 1.0  # Equivalent to np.eye(nx) for diagonal assignment
        b_eq_temp[row_offset_eq + i] = x0_flat[i]
    row_offset_eq += nx

    # 2. Dynamics constraints: x_{t+1} - A_dt @ x_t - B_dt @ u_t == 0
    for t in range(N):
        u_t_col_start = t * nu
        x_t_col_start = nu * N + t * nx
        x_t1_col_start = nu * N + (t + 1) * nx

        for i in range(nx):
            for j in range(nu):
                A_eq_temp[row_offset_eq + i, u_t_col_start + j] = -B_dt[i, j]

        for i in range(nx):
            for j in range(nx):
                A_eq_temp[row_offset_eq + i, x_t_col_start + j] = -A_dt[i, j]

        for i in range(nx):
            A_eq_temp[row_offset_eq + i, x_t1_col_start + i] = 1.0  # Equivalent to np.eye(nx)
        row_offset_eq += nx

    # 3. Terminal constraint: x[idx_term, N] == r_underline_static
    # Only add terminal constraints if there are any
    if len_idx_term > 0:
        x_N_col_start = nu * N + N * nx
        # Advanced indexing (x_N_col_start + idx_term) needs a loop for Numba compatibility
        for i in range(len_idx_term):
            A_eq_temp[row_offset_eq + i, x_N_col_start + idx_term[i]] = 1.0
            b_eq_temp[row_offset_eq + i] = r_underline_static[i]
            # print("r_underline_static",r_underline_static)
    # print("len_idx_term",len_idx_term)
    # --- Temporary arrays for Inequality Constraints (A_ineq_qp, b_ineq_qp) ---
    nc_ineq_qp = nc_state * N + nc_input * N
    A_ineq_qp_temp = np.zeros((nc_ineq_qp, nz), dtype=np.float64)
    b_ineq_qp_temp = np.zeros(nc_ineq_qp, dtype=np.float64)

    row_offset_ineq_qp = 0

    # 1. State constraints: A_ineq @ x_t <= b_ineq_static
    for t in range(N):
        x_t_col_start = nu * N + t * nx
        for i in range(nc_state):
            for j in range(nx):
                A_ineq_qp_temp[row_offset_ineq_qp + i, x_t_col_start + j] = A_ineq[i, j]
            b_ineq_qp_temp[row_offset_ineq_qp + i] = b_ineq_static[i]
        row_offset_ineq_qp += nc_state

    # 2. Input constraints: A_ineq_u @ u_t <= bu_ineq_static
    for t in range(N):
        u_t_col_start = t * nu
        for i in range(nc_input):
            for j in range(nu):
                A_ineq_qp_temp[row_offset_ineq_qp + i, u_t_col_start + j] = A_ineq_u[i, j]
            b_ineq_qp_temp[row_offset_ineq_qp + i] = bu_ineq_static[i]
        row_offset_ineq_qp += nc_input

    # --- Combine all inequality constraints into A_ineq_combined_out and b_ineq_combined_out ---
    current_combined_row_offset = 0

    # Copy A_ineq_qp_temp and b_ineq_qp_temp
    for i in range(nc_ineq_qp):
        for j in range(nz):
            A_ineq_combined_out[current_combined_row_offset + i, j] = A_ineq_qp_temp[i, j]
        b_ineq_combined_out[current_combined_row_offset + i] = b_ineq_qp_temp[i]
    current_combined_row_offset += nc_ineq_qp

    # Copy A_eq_temp (positive) and b_eq_temp (positive)
    for i in range(nc_eq):
        for j in range(nz):
            A_ineq_combined_out[current_combined_row_offset + i, j] = A_eq_temp[i, j]
        b_ineq_combined_out[current_combined_row_offset + i] = b_eq_temp[i]
    current_combined_row_offset += nc_eq

    # Copy -A_eq_temp (negative) and -b_eq_temp (negative)
    for i in range(nc_eq):
        for j in range(nz):
            A_ineq_combined_out[current_combined_row_offset + i, j] = -A_eq_temp[i, j]
        b_ineq_combined_out[current_combined_row_offset + i] = -b_eq_temp[i]
    
    

def build_qp_matrices(dp_params, mpc_params, x0, Q, R, x_ref, current_time_step):
    """
    Wrapper function to build QP matrices using the Numba-optimized core.
    Handles array allocation and ensures data is in a Numba-compatible format.
    """
    # Ensure x0 is a 1D array for Numba compatibility
    x0_flat = np.array(x0).flatten()

    A_dt = mpc_params['A']
    B_dt = mpc_params['B']
    A_ineq = mpc_params['A_ineq']
    # Ensure these are 1D arrays for Numba compatibility
    b_ineq_static = mpc_params['b_ineq'].flatten()
    A_ineq_u = mpc_params['A_ineq_u']
    bu_ineq_static = mpc_params['bu_ineq'].flatten()
    # Handle case where r_underline is not present (terminal constraints removed)
    if 'r_underline' in mpc_params:
        r_underline_static = mpc_params['r_underline'].flatten()
    else:
        r_underline_static = np.array([])  # No terminal constraints
    prediction_horizon = mpc_params['J']

    N = prediction_horizon - current_time_step

    nx = A_dt.shape[1]
    nu = B_dt.shape[1]
    nc_state = A_ineq.shape[0]
    nc_input = A_ineq_u.shape[0]

    # Validate dimensions
    assert x0_flat.shape[0] == nx, "Initial state dimension mismatch"
    assert b_ineq_static.shape[0] == nc_state, "State constraint dimension mismatch"
    assert bu_ineq_static.shape[0] == nc_input, "Input constraint dimension mismatch"

    nz = nu * N + nx * (N + 1)

    # Determine terminal constraint indices based on dp_params
    # Pass boolean directly, and create idx_term as a Numba-compatible array
    include_velocity_in_planner = dp_params['INCLUDE_VELOCITY_IN_PLANNER']
    if include_velocity_in_planner:
        idx_term = np.r_[6:9]
    else:
        idx_term = np.r_[3:9]
    len_idx_term = len(idx_term)
    idx_term = np.array([], dtype=np.int64)  # Empty integer array for Numba compatibility
    len_idx_term = 0

    # Calculate sizes for pre-allocation of output arrays
    nc_eq = nx + nx * N + len_idx_term  # Number of equality constraints
    nc_ineq_qp = nc_state * N + nc_input * N  # Number of original inequality constraints
    total_combined_ineq_rows = nc_ineq_qp + nc_eq + nc_eq  # Total rows for A_ineq_combined_out

    # Pre-allocate output arrays in object mode (pure Python/NumPy)
    H_out = np.zeros((nz, nz), dtype=np.float64)
    f_out = np.zeros(nz, dtype=np.float64)
    A_ineq_combined_out = np.zeros((total_combined_ineq_rows, nz), dtype=np.float64)
    b_ineq_combined_out = np.zeros(total_combined_ineq_rows, dtype=np.float64)
    # print("A_ineq_combined_out",A_ineq_combined_out)
    # print("b_ineq_combined_out",b_ineq_combined_out)
    # Call the njit-decorated core function to fill the pre-allocated arrays
    _build_qp_matrices_core(
        include_velocity_in_planner,
        A_dt, B_dt, A_ineq, b_ineq_static, A_ineq_u, bu_ineq_static, r_underline_static,
        N, nx, nu, nc_state, nc_input, nz,
        x0_flat, Q, R, x_ref, current_time_step,
        idx_term, len_idx_term,
        H_out, f_out, A_ineq_combined_out, b_ineq_combined_out
    )

    return H_out, f_out, A_ineq_combined_out, b_ineq_combined_out


def DAQP_fast(dp_params, mpc_params, x0, Q, R, x_ref, current_time_step, upper_level_constraints=None,
            DEBUG_MODE_SMPC=False):
    """
    Solves a Quadratic Program (QP) for Model Predictive Control (MPC) using the DAQP solver directly.
    Args:
        dp_params (dict): Contains dynamic parameters like 'INCLUDE_VELOCITY_IN_PLANNER'.
        mpc_params (dict): Contains the MPC parameters including state and input matrices, constraints, etc.
        x0 (array): Initial state of the system.
        Q (array): State cost matrix.
        R (array): Input cost matrix.
        x_ref (array): Reference state trajectory for the full prediction horizon (nx, prediction_horizon + 1).
        current_time_step (int): Current time-step.
        upper_level_constraints (list): Upper level constraints (not implemented and ignored here, as in original).
        DEBUG_MODE_SMPC (bool): Debug mode flag for additional output.
    Returns:
        u_sol (array): Optimal control inputs (nu, remaining_horizon).
        x_sol (array): Optimal state trajectory (nx, remaining_horizon + 1).
        epsilon_sol (array): Slack variables for soft constraints (always None here, as in original).
    """
    # QP_test() # This function was not defined in the original snippet, commenting out.

    AUTO_COMPILE = True

    if AUTO_COMPILE:

        nu = mpc_params['B'].shape[1]  # Number of control inputs
        nx = mpc_params['A'].shape[1]  # Number of states
        N = mpc_params['J'] - current_time_step  # Remaining prediction horizon
        # Everything from here...
        # Build the QP matrices using the Numba-optimized function
        H, f, A_ineq_combined, b_ineq_combined = build_qp_matrices(
            dp_params, mpc_params, x0, Q, R, x_ref, current_time_step
        )
        # print("A_ineq_combined",A_ineq_combined)
        # print("b_inequ_combined",b_ineq_combined)

        # ... up to here, shall go in the build_qp_matrices function.
    else:
        # Everything from here...
        x0 = np.array(x0).reshape(-1, 1)  # Ensure x0 is a column vector
        A_dt = mpc_params['A']  # Discretized state matrix
        B_dt = mpc_params['B']  # Discretized input matrix
        A_ineq = mpc_params['A_ineq']  # State constraint matrix
        b_ineq_static = mpc_params['b_ineq']  # Right hand side of the state constraint for time-steps 0,...,N-1
        A_ineq_u = mpc_params['A_ineq_u']  # Input constraint matrix
        bu_ineq_static = mpc_params['bu_ineq']  # Right hand side of the input constraint for time-steps 0,...,N-1
        # Handle case where r_underline is not present (terminal constraints removed)
        if 'r_underline' in mpc_params:
            r_underline_static = mpc_params['r_underline']  # Terminal constraint for the velocity, angle and angular velocity
        else:
            r_underline_static = np.array([])  # No terminal constraints
        prediction_horizon = mpc_params['J']  # Prediction horizon for MPC
        # N represents the remaining horizon for the MPC problem
        N = prediction_horizon - current_time_step
        # Dimensions
        nx = A_dt.shape[1]  # State dimensionality
        nu = B_dt.shape[1]  # Input dimensionality
        nc_state = A_ineq.shape[0]  # Number of state constraints at every time-step
        nc_input = A_ineq_u.shape[0]  # Number of input constraints at every time-step (corrected from .shape[1])
        # Validate dimensions
        assert x0.shape[0] == nx, "Initial state dimension mismatch"
        assert b_ineq_static.shape[0] == nc_state, "State constraint dimension mismatch"
        assert bu_ineq_static.shape[0] == nc_input, "Input constraint dimension mismatch"

        # Define the total number of decision variables (z)
        # z = [u_0, ..., u_{N-1}, x_0, ..., x_N]^T
        nz = nu * N + nx * (N + 1)

        # --- Construct H (Hessian) matrix for 0.5 * z^T H z ---
        # The cost function is sum( (x_t - x_ref_t)^T Q (x_t - x_ref_t) + u_t^T R u_t )
        # This expands to sum( x_t^T Q x_t - 2 x_t^T Q x_ref_t + u_t^T R u_t ) + constant terms
        # So, H will have 2*R blocks for u and 2*Q blocks for x.
        H = np.zeros((nz, nz))
        # Blocks for control inputs (u_0 to u_{N-1})
        for t in range(N):
            u_idx_start = t * nu
            H[u_idx_start: u_idx_start + nu, u_idx_start: u_idx_start + nu] = 2 * R

        # Blocks for state trajectories (x_0 to x_N)

        for t in range(N + 1):
            x_idx_start = nu * N + t * nx
            H[x_idx_start: x_idx_start + nx, x_idx_start: x_idx_start + nx] = 2 * Q

        # --- Construct f (linear term) vector ---
        # f contains -2 * Q * x_ref_t for x_t terms
        f = np.zeros(nz)
        # Linear terms for state trajectories (x_0 to x_{N-1})
        for t in range(N + 1):
            x_idx_start = nu * N + t * nx
            f[x_idx_start: x_idx_start + nx] = -2 * Q @ x_ref[:, current_time_step + t]
        # Linear term for the terminal state (x_N)
        # x_N_idx_start = nu * N + N * nx
        # f[x_N_idx_start: x_N_idx_start + nx] = -2 * Q @ x_ref[:, prediction_horizon]
        # --- Construct Equality Constraints (A_eq, b_eq) ---
        # 1. Initial state constraint: x_0 == x0
        # 2. Dynamics constraints: x_{t+1} - A_dt @ x_t - B_dt @ u_t == 0 for t = 0,...,N-1
        # 3. Terminal constraint: x[idx_term, N] == r_underline_static
        # Determine terminal constraint indices based on dp_params
        if dp_params['INCLUDE_VELOCITY_IN_PLANNER']:
            idx_term = np.r_[6:12]  # Indices for z pos, z vel, angle, angular velocity
        else:
            idx_term = np.r_[3:12]  # Indices for z pos, velocity, angle, angular velocity
        len_idx_term = len(idx_term)
        idx_term=np.array([])
        len_idx_term = 0

        nc_eq = nx + nx * N + len_idx_term  # Total number of equality constraints
        A_eq = np.zeros((nc_eq, nz))
        b_eq = np.zeros(nc_eq)
        row_offset_eq = 0

        # 1. Initial state constraint: x_0 == x0
        x0_col_start = nu * N  # Column index where x_0 starts in z
        A_eq[row_offset_eq: row_offset_eq + nx, x0_col_start: x0_col_start + nx] = np.eye(nx)
        b_eq[row_offset_eq: row_offset_eq + nx] = np.squeeze(x0)
        row_offset_eq += nx

        # 2. Dynamics constraints: x_{t+1} - A_dt @ x_t - B_dt @ u_t == 0
        for t in range(N):
            u_t_col_start = t * nu
            x_t_col_start = nu * N + t * nx
            x_t1_col_start = nu * N + (t + 1) * nx
            A_eq[row_offset_eq: row_offset_eq + nx, u_t_col_start: u_t_col_start + nu] = -B_dt
            A_eq[row_offset_eq: row_offset_eq + nx, x_t_col_start: x_t_col_start + nx] = -A_dt
            A_eq[row_offset_eq: row_offset_eq + nx, x_t1_col_start: x_t1_col_start + nx] = np.eye(nx)
            row_offset_eq += nx

        # 3. Terminal constraint: x[idx_term, N] == r_underline_static
        x_N_col_start = nu * N + N * nx  # Column index where x_N starts in z
        A_eq[row_offset_eq: row_offset_eq + len_idx_term, x_N_col_start + idx_term] = np.eye(len_idx_term)
        b_eq[row_offset_eq: row_offset_eq + len_idx_term] = np.squeeze(r_underline_static)
        
        # row_offset_eq += len_idx_term # Not strictly needed as this is the last equality constraint

        # --- Construct Inequality Constraints (A_ineq_qp, b_ineq_qp) ---
        # 1. State constraints: A_ineq @ x_t <= b_ineq_static for t = 0,...,N-1
        # 2. Input constraints: A_ineq_u @ u_t <= bu_ineq_static for t = 0,...,N-1
        nc_ineq = nc_state * N + nc_input * N  # Total number of inequality constraints
        A_ineq_qp = np.zeros((nc_ineq, nz))
        b_ineq_qp = np.zeros(nc_ineq)
        row_offset_ineq = 0

        # 1. State constraints: A_ineq @ x_t <= b_ineq_static
        for t in range(N):
            x_t_col_start = nu * N + t * nx
            A_ineq_qp[row_offset_ineq: row_offset_ineq + nc_state, x_t_col_start: x_t_col_start + nx] = A_ineq
            b_ineq_qp[row_offset_ineq: row_offset_ineq + nc_state] = np.squeeze(b_ineq_static)
            row_offset_ineq += nc_state

        # 2. Input constraints: A_ineq_u @ u_t <= bu_ineq_static
        for t in range(N):
            u_t_col_start = t * nu
            A_ineq_qp[row_offset_ineq: row_offset_ineq + nc_input, u_t_col_start: u_t_col_start + nu] = A_ineq_u
            b_ineq_qp[row_offset_ineq: row_offset_ineq + nc_input] = np.squeeze(bu_ineq_static)
        # row_offset_ineq += nc_input # Not strictly needed as this is the last inequality constraint

        A_eq_pos = A_eq
        b_eq_pos = b_eq
        # A_eq_neg represents -A_eq*z <= -b_eq
        A_eq_neg = -A_eq
        b_eq_neg = -b_eq
        # Then, combine all inequality constraints into a single set.
        # This includes the original inequality constraints and the new ones from the equalities.
        A_ineq_combined = np.vstack((A_ineq_qp, A_eq_pos, A_eq_neg))
        b_ineq_combined = np.concatenate((b_ineq_qp, b_eq_pos, b_eq_neg))

        # ... up to here, shall go in the build_qp_matrices function.

    # --- Solve the QP using DAQP ---
    try:
        # DAQP's solve function returns a result object
        # result.x contains the optimal solution vector z

        # Pass the explicitly reshaped arrays to the solver
        result = daqp.solve(H, f, A_ineq_combined, b_ineq_combined)
        # Pass the new arrays to the solver
       # result = daqp.solve(H, f, A_ineq_qp, b_ineq_qp, A_eq_contiguous, b_eq_contiguous)
#        result = daqp.solve(H, f, A_ineq_qp, b_ineq_qp, A_eq, b_eq.flatten())
        z_sol = result[0]

        if DEBUG_MODE_SMPC:
            if result.exitflag == daqp.SOLVED:
                print("DAQP: The problem is feasible and optimal.")
                print(f"DAQP: The optimal value of the objective is {result.fval}.")
            elif result.exitflag == daqp.INFEASIBLE:
                print("DAQP: The problem is infeasible.")
            elif result.exitflag == daqp.UNBOUNDED:
                print("DAQP: The problem is unbounded.")
            else:
                print(f"DAQP: The problem status is {result.exitflag}.")

    except Exception as e:
        print(f"Error solving DAQP problem: {e}")
        # Return None for all outputs if the solver fails
        return None, None, None

    # --- Extract the solution from z_sol ---
    # u_sol corresponds to the first N*nu elements of z_sol
    u_sol = z_sol[0: nu * N].reshape(nu, N, order='F')
    # x_sol corresponds to the remaining (N+1)*nx elements of z_sol
    x_sol = z_sol[nu * N:].reshape(nx, N + 1, order='F')
    if (np.abs(x_sol[2,0] - x0[2])>0.5):
        print("DEBUG")

    # epsilon_sol is always None as per the original function's return signature
    return u_sol, x_sol, None

def QP_fast(dp_params, mpc_params, x0, Q, R, x_ref, current_time_step, upper_level_constraints=None, DEBUG_MODE_SMPC = False):
    """
    Solves a Quadratic Program (QP) for Model Predictive Control (MPC).
    Args:
        mpc_params (dict): Contains the MPC parameters including state and input matrices, constraints, etc.
        x0 (array): Initial state of the system.
        Q (array): State cost matrix.
        R (array): Input cost matrix.
        x_ref (array): Reference state to track.
        current_time_step (int): Current time-step.
        upper_level_constraints (list): Upper level constraints not implemented yet.
        DEBUG_MODE_SMPC (bool): Debug mode flag for additional output.
    Returns:
        u_sol (array): Optimal control inputs.
        x_sol (array): Optimal state trajectory.
        epsilon_sol (array): Slack variables for soft constraints.
    """
    #QP_test()

    x0 = np.array(x0).reshape(-1, 1)
    # Upper level constraints not implemented yet.
    A_dt = mpc_params['A']  # Discretized state matrix
    B_dt = mpc_params['B']  # Discretized input matrix
    # x0 is now handled by a cvxpy.Parameter, so no need to reshape here.
    A_ineq = mpc_params['A_ineq']  # State constraint matrix
    b_ineq_static = mpc_params['b_ineq']  # Right hand side of the state constraint for time-steps 0,...,N-1
    A_ineq_u = mpc_params['A_ineq_u']  # Input constraint matrix
    bu_ineq_static = mpc_params['bu_ineq']  # Right hand side of the input constraint for time-steps 0,...,N-1
    # Handle case where r_underline is not present (terminal constraints removed)
    if 'r_underline' in mpc_params:
        r_underline_static = mpc_params['r_underline']  # Terminal constraint for the velocity, angle and angular velocity
    else:
        r_underline_static = np.array([])  # No terminal constraints
    prediction_horizon = mpc_params['J']  # Prediction horizon for MPC
    remaining_horizon = mpc_params['J'] - current_time_step  # Remaining horizon for MPC

    # SOLVING THE MPC PROBLEM
    # Dimensions
    nx = A_dt.shape[1]            # State dimensionality
    nu = B_dt.shape[1]              # Input dimensionality
    nc_state = A_ineq.shape[0]     # Number of state constraints at every time-step
    nc_input = A_ineq_u.shape[0]     # Number of input constraints at every time-step

    # Validate dimensions
    assert np.array(x0).shape[0] == nx, "Initial state dimension mismatch"
    assert b_ineq_static.shape[0] == nc_state, "State constraint dimension mismatch"

    # Define variables
    # IMPORTANT NOTE: The dimensions of x, u, and epsilon depend on 'remaining_horizon',
    # which changes with 'current_time_step'. This means the underlying structure
    # of the optimization problem changes in each call. As a result, a new
    # cvxpy.Problem instance must be created every time this function is called.
    # This prevents the most significant speedup technique in cvxpy (defining the
    # problem once and only updating parameters).
    # However, using cvxpy.Parameter for dynamic *values* and enabling warm-starting
    # still provides performance benefits.
    x = cp.Variable((nx, remaining_horizon + 1))
    u = cp.Variable((nu, remaining_horizon))
    #epsilon = cp.Variable(remaining_horizon + 1)     # Slack variable for soft constraints

    # Define cvxpy.Parameter objects for dynamic inputs.
    # These allow cvxpy to efficiently update numerical values within the problem.
    x0_param = cp.Parameter(nx)
    Q_param = cp.Parameter((nx, nx), PSD=True) # Q is a cost matrix, typically Positive Semidefinite
    R_param = cp.Parameter((nu, nu), PSD=True) # R is a cost matrix, typically Positive Semidefinite
    x_ref_param = cp.Parameter((nx, prediction_horizon + 1)) # x_ref spans the full prediction horizon

    # Assign current numerical values to the parameters
    x0_param.value = np.squeeze(x0)
    Q_param.value = Q
    R_param.value = R
    x_ref_param.value = x_ref # x_ref should be provided for the full prediction horizon

    # Define the cost function
    cost = 0
    for t in range(remaining_horizon):
        # Use parameters for Q, R, and slice x_ref_param for the current time step
        cost += cp.quad_form(x[:, t] - x_ref_param[:, current_time_step + t], Q_param) + cp.quad_form(u[:, t], R_param)
    # Terminal cost: use x_ref_param for the end of the full prediction horizon
    cost += cp.quad_form(x[:, remaining_horizon] - x_ref_param[:, prediction_horizon], Q_param)
    #cost += cp.quad_form(epsilon, 100000.0*np.eye(remaining_horizon+1)) # Soft constraint tightening penalty
    #cost += 100.0* cp.sum(epsilon)

    # Define constraints
    constraints = []
    constraints.append(x[:, 0] == x0_param)       # Initial state constraint using parameter
    for t in range(remaining_horizon):
        constraints.append(A_ineq @ x[:, t] <= np.squeeze(b_ineq_static))# + epsilon[t] * np.ones(nc_state))  # Soft chance constraints on state
        constraints.append(x[:, t + 1] == A_dt @ x[:, t] + B_dt @ u[:, t])  #   Dynamics constraint
        constraints.append(A_ineq_u @ u[:, t] <= bu_ineq_static)    # Hard input constraints
        #constraints.append(epsilon[t] >= 0)    # Non-negative slack variable
    #constraints.append(epsilon[remaining_horizon] >= 0)    # Non-negative slack variable

    # Define terminal constraint for velocity, angle and angular velocity as r_underline
    # if dp_params['INCLUDE_VELOCITY_IN_PLANNER']:
    #     constraints.append(x[np.r_[2, 5:13], remaining_horizon] == r_underline_static)# + epsilon[remaining_horizon] * np.ones(8))
    # else:
    #     constraints.append(x[2:12, remaining_horizon] == r_underline_static)# + epsilon[remaining_horizon] * np.ones(10))


    # Define and solve the problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    # Explicitly enable warm_start. This is crucial for repeated solves,
    # as it allows the solver (MOSEK) to use the previous solution as an initial guess,
    # potentially speeding up convergence even when the problem is re-created.
    prob.solve(solver=cp.GUROBI, warm_start=True)

    # if DEBUG_MODE_SMPC:
    #     if prob.status == cp.OPTIMAL:
    #         print("The problem is feasible and optimal.")
    #         print(f"The optimal value of the objective is {prob.value}.")
    #     elif prob.status == cp.INFEASIBLE:
    #         print("The problem is infeasible.")
    #     elif prob.status == cp.UNBOUNDED:
    #         print("The problem is unbounded.")
    #     else:
    #         print(f"The problem status is {prob.status}.")
    #     epsilon_sol = epsilon.value

    # Extract the solution
    u_sol = u.value
    x_sol = x.value
    #epsilon_sol = epsilon.value
    # END OF SOLVING THE MPC PROBLEM
    return u_sol, x_sol, None#epsilon_sol

def QP(dp_params, mpc_params, x0, Q, R, x_ref, current_time_step, upper_level_constraints=[], DEBUG_MODE_SMPC = False):
    """
    Solves a Quadratic Program (QP) for Model Predictive Control (MPC).
    Args:
        mpc_params (dict): Contains the MPC parameters including state and input matrices, constraints, etc.
        x0 (array): Initial state of the system.
        Q (array): State cost matrix.
        R (array): Input cost matrix.
        x_ref (array): Reference state to track.
        current_time_step (int): Current time-step.
        upper_level_constraints (list): Upper level constraints not implemented yet.
        DEBUG_MODE_SMPC (bool): Debug mode flag for additional output.
    Returns:
        u_sol (array): Optimal control inputs.
        x_sol (array): Optimal state trajectory.
        epsilon_sol (array): Slack variables for soft constraints.
    """

    # Upper level constraints not implemented yet.
    A_dt = mpc_params['A']  # Discretized state matrix
    B_dt = mpc_params['B']  # Discretized input matrix
    x0 = np.array(x0).reshape(-1, 1)  # Ensure x0 is a column vector
    A_ineq = mpc_params['A_ineq']  # State constraint matrix
    b_ineq = mpc_params['b_ineq']  # Right hand side of the state constraint for time-steps 0,...,N-1
    A_ineq_u = mpc_params['A_ineq_u']  # Input constraint matrix
    bu_ineq = mpc_params['bu_ineq']  # Right hand side of the input constraint for time-steps 0,...,N-1
    r_underline = mpc_params['r_underline']  # Terminal constraint for the velocity, angle and angular velocity
    prediction_horizon = mpc_params['J']  # Prediction horizon for MPC
    remaining_horizon = mpc_params['J'] - current_time_step  # Remaining horizon for MPC

    # SOLVING THE MPC PROBLEM
    # Dimensions
    nx = A_dt.shape[0]              # State dimensionality
    nu = B_dt.shape[1]              # Input dimensionality
    nc_state = A_ineq.shape[0]      # Number of state constraints at every time-step
    nc_input = A_ineq_u.shape[0]     # Number of input constraints at every time-step

    # Validate dimensions
    assert x0.shape[0] == nx, "Initial state dimension mismatch"
    assert b_ineq.shape[0] == nc_state, "State constraint dimension mismatch"

    # Define variables
    x = cp.Variable((nx, remaining_horizon + 1))
    u = cp.Variable((nu, remaining_horizon))
    epsilon = cp.Variable(remaining_horizon + 2)     # Slack variable to realize soft constraints

    #debug_cost = np.diag([100,100,0, 0,0,0, 0,0,0, 0,0,0])

    # Define the cost function
    cost = 0
    for t in range(remaining_horizon):
        cost += cp.quad_form(x[:, t] - x_ref[:,current_time_step + t], Q) + cp.quad_form(u[:, t], R)     # Quadratic cost for k=0,...,N-1, no terminal cost
    cost += cp.quad_form(x[:, remaining_horizon] - x_ref[:,prediction_horizon], Q)    # Quadratic cost for k=0,...,N-1, no terminal cost
        #cost += cp.quad_form(x[:, t]-xref, debug_cost) + cp.quad_form(u[:, t], MPC_R)     # Quadratic cost for k=0,...,N-1, no terminal cost
    cost += cp.quad_form(epsilon, 10000.0*np.eye(remaining_horizon+2)) # Soft constraint tightening penalty
    cost += 100.0* cp.sum(epsilon)

    

    # Define constraints
    constraints = []
    constraints.append(x[:, 0] == np.squeeze(x0))       # Initial state constraint
    for t in range(remaining_horizon):
        constraints.append(A_ineq @ x[:, t] <= np.squeeze(b_ineq) + epsilon[t] * np.ones(nc_state))  # Soft chance constraints on state
        constraints.append(x[:, t + 1] == A_dt @ x[:, t] + B_dt @ u[:, t])  #   Dynamics constraint
        constraints.append(A_ineq_u @ u[:, t] <= bu_ineq)    # Hard input constraints
        constraints.append(epsilon[t] >= 0)    # Non-negative slack variable
    constraints.append(epsilon[remaining_horizon+1] >=0 )    # Non-negative slack variable
    constraints.append(epsilon[remaining_horizon] >= 0)

    # Define terminal constraint for velocity, angle and angular velocity as r_underline
    # print("Terminal constraint", r_underline)
    if dp_params['INCLUDE_VELOCITY_IN_PLANNER']:
        constraints.append(x[6:9, remaining_horizon] <= r_underline + epsilon[remaining_horizon] * np.ones(3))
        constraints.append(x[6:9, remaining_horizon] >= r_underline - epsilon[remaining_horizon] * np.ones(3))
    else:
        constraints.append(x[2:9, remaining_horizon] == r_underline + epsilon[remaining_horizon] * np.ones(7))
    # print("Constraints", constraints)

    # Define and solve the problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    # prob = cp.Problem(cp.Minimize(cost), constraints)
    #prob.solve(solver=cp.MOSEK, verbose=True)#
    # prob.solve(solver=cp.MOSEK) 
    prob.solve(cp.GUROBI)

    # if DEBUG_MODE_SMPC:
    #     if prob.status == cp.OPTIMAL:
    #         print("The problem is feasible and optimal.")
    #         print(f"The optimal value of the objective is {prob.value}.")
    #     elif prob.status == cp.INFEASIBLE:
    #         print("The problem is infeasible.")
    #     elif prob.status == cp.UNBOUNDED:
    #         print("The problem is unbounded.")
    #     else:
    #         print(f"The problem status is {prob.status}.")
    #     epsilon_sol = epsilon.value

    # Extract the solution
    u_sol = u.value
    x_sol = x.value
    epsilon_sol = epsilon.value
    # END OF SOLVING THE MPC PROBLEM
    return u_sol, x_sol, epsilon_sol

def shrinking_horizon_SMPC(dp_params, mpc_params, sim_params, system_state, reference, Q, R, upper_level_constraints=None):
    """
    Implements the shrinking horizon MPC algorithm.
    This function recursively calls the QP solver for the MPC problem, reducing the prediction horizon at each step.
    :param mpc_params: Dictionary containing MPC parameters.
    :param system_state: Current state of the system.
    :param action: Current action to be taken.
    :param reference: Reference state to track.
    :param upper_layer_constraints: Constraints from the upper layer (not implemented yet).
    :return: terminal_state, state_trajectory, feasible
    """
    reward = 0.0
    x_trajectory = [system_state]
    # Initialize the QP problem structure
    #problem_data = define_mpc_problem(mpc_params)
    for j in range(mpc_params['J']):
        # u_sol, x_sol, epsilon_sol = DAQP_fast(dp_params, mpc_params, x_trajectory[-1], Q, R, reference, j)
        u_sol, x_sol, epsilon_sol = QP(dp_params, mpc_params, x_trajectory[-1], Q, R, reference, j)
        #u_sol, x_sol, epsilon_sol = QP_warm_start(problem_data, dp_params, mpc_params, x_trajectory[-1], Q, R, reference, j)
        if u_sol is None or x_sol is None:
            raise ValueError("QP solver did not return a valid solution.")
        next_state, state_sequence = quadcopter_dynamics_multi_step(x_trajectory[-1], u_sol[:, 0], sim_params)
        # Append every state in the sequence to the trajectory
        x_trajectory.extend(state_sequence)

        # fig = plt.figure(0)
        # plt.subplot(111, projection='3d')

        # x_data = x_sol[0, :]
        # y_data = x_sol[1, :]
        # z_data = x_sol[2, :]

        # plt.plot(x_data, y_data, z_data, label='Quadcopter Trajectory')

        # # Plotting the reference
        # ref_positions = np.array([reference[0:3, j] for j in range(mpc_params['J']+1)])
        # plt.plot(ref_positions[:, 0], ref_positions[:, 1], ref_positions[:, 2], label='Reference Trajectory', linestyle='--', color='orange')

        # plt.xlabel('X Position (m)')
        # plt.ylabel('Y Position (m)')
        # plt.title('Quadcopter 3D Trajectory')
        # plt.legend()
        # plt.grid(True)
        # plt.show(block=True)

        for states in state_sequence:
            reward += 1 / sim_params['simulation_steps_per_input'] * (
                    np.transpose(states - reference[:,j])@Q@(states - reference[:,j])
                    + np.transpose(u_sol[:, 0])@R@u_sol[:, 0])  # Reward is the cost of the current state and input
    return x_trajectory[-1], x_trajectory, reward

def shrinking_horizon_SMPC_phoenix(dp_params, mpc_params, sim_params, system_state, reference, Q, R):
    """
    Implements the shrinking horizon MPC algorithm with Phoenix integration.
    This function recursively calls the QP solver for the MPC problem, reducing the prediction horizon at each step.
    It also returns control actions and their outputs for each MPC step.
    :param mpc_params: Dictionary containing MPC parameters.
    :param system_state: Current state of the system.
    :param reference: Reference state to track.
    :param Q: State cost matrix.
    :param R: Input cost matrix.
    :return: terminal_state, state_trajectory, reward, control_actions, control_outputs
    """
    reward = 0.0
    x_trajectory = [system_state]
    control_actions = np.zeros((4, mpc_params['J']))
    control_outputs = []

    for j in range(mpc_params['J']):
        # Call the DAQP_fast function to get control actions and state trajectory
        u_sol, x_sol, epsilon_sol = DAQP_fast(dp_params, mpc_params, x_trajectory[-1], Q, R, reference, j)

        if u_sol is None or x_sol is None:
            raise ValueError("QP solver did not return a valid solution.")

        # Extract control action for this step
        current_control = u_sol[:, 0]  # [thrust, roll_rate, pitch_rate, yaw_rate]
        control_actions[:, j] = current_control

        # Store control outputs for this step
        control_outputs.append({
            'step': j,
            'control_action': current_control.copy(),
            'state_trajectory': x_sol.copy()
        })

        # Append the state trajectory for the next MPC step
        x_trajectory.extend(x_sol)

        # Update reward
        reward += 1 / sim_params['simulation_steps_per_input'] * (
                np.transpose(x_sol[:, -1] - reference[:, j + 1]) @ Q @ (x_sol[:, -1] - reference[:, j + 1])
                + np.transpose(current_control) @ R @ current_control
        )

    # Return the first control action for immediate use, plus tracking data
    first_control_action = control_actions[:, 0] if control_actions.shape[1] > 0 else np.zeros(4)

    return x_trajectory[-1], x_trajectory, reward, first_control_action, control_outputs

def test_smpc(initial_system_state=np.zeros((12,), dtype=np.float32)):
    com_params, dp_params, mpc_params, sim_params = load_parameters()
    initial_system_state[0]=0.4

    
    system_state = initial_system_state  # Initial system state

    reference = np.zeros( (12,mpc_params['J']+1), dtype=np.float32)  # Reference trajectory

    REFERENCE_TYPE = "HELIX"

    if REFERENCE_TYPE == "RAMP":
        for j in range(mpc_params['J']+1):
            reference[0, j] = j * 0.01  # Example ramp reference for the x position
            reference[1, j] = j * 0.02  # Example ramp reference for the y position
            reference[2, j] = j * 0.03  # Example ramp reference for the z position
    if REFERENCE_TYPE == "HELIX":
        radius = .5
        for j in range(mpc_params['J']+1):
            theta = 2 * np.pi * j / mpc_params['J']
            reference[0, j] = radius * np.cos(theta)
            reference[1, j] = radius * np.sin(theta)
            reference[2, j] = 0.01*j

    Q = mpc_params['MPC_Q']  # State cost matrix
    R = mpc_params['MPC_R']  # Input cost matrix

    # Extract control actions by calling DAQP_fast for each step
    # print("="*60)
    # print("EXTRACTING CONTROL ACTIONS FROM QP SOLVER")
    # print("="*60)
    
    control_actions = []
    current_state = system_state
    
    for j in range(mpc_params['J']):
        # Call DAQP_fast to get control actions
        u_sol, x_sol, epsilon_sol = QP(dp_params, mpc_params, current_state, Q, R, reference, j)
        # print('current_state',current_state[0],current_state[1],current_state[2],current_state[3],current_state[4],current_state[5])
        # print ('x_sol(initial) ', x_sol[0,0], x_sol[1,0], x_sol[2,0],x_sol[3,0], x_sol[4,0], x_sol[5,0] )
        # print ('x_sol(next)', x_sol[0,1], x_sol[1,1], x_sol[2,1],x_sol[3,1], x_sol[4,1], x_sol[5,1])
        # print ('x_sol(end)', x_sol[0,-1], x_sol[1,-1], x_sol[2,-1],x_sol[3,-1], x_sol[4,-1], x_sol[5,-1])
        if u_sol is not None:
            # Extract first control action from solution
            first_control = u_sol[:, 0]  # [thrust, roll_rate, pitch_rate, yaw_rate]
            control_actions.append(first_control)
            
            # Update state for next iteration
            current_state = x_sol[:, 1]  # Next state from solution
        else:
            print(f"QP solver failed at step {j}")
            break
    
    # Run the original shrinking horizon MPC to get state trajectory
    terminal_state, state_trajectory, reward = shrinking_horizon_SMPC(
        dp_params=dp_params,
        mpc_params=mpc_params,
        sim_params=sim_params,
        system_state=system_state,
        reference=reference,
        Q=Q,
        R=R,
        upper_level_constraints=None
    )
    
    print("="*60)
    print("SMPC TEST RESULTS")
    print("="*60)
    print("Terminal State:", terminal_state[:3])  # Position only
    print("Reward:", reward)
    print(f"State Trajectory Length: {len(state_trajectory)}")
    print(f"Number of Control Actions: {len(control_actions)}")
    
    # Print control actions for each step
    print("\nCONTROL ACTIONS (Thrust, Roll_Rate, Pitch_Rate, Yaw_Rate):")
    print("Step | Thrust | Roll_Rate | Pitch_Rate | Yaw_Rate")
    print("-" * 50)
    for i, control in enumerate(control_actions):
        print(f"{i:4d} | {control[0]:6.3f} | {control[1]:9.3f} | {control[2]:10.3f} | {control[3]:8.3f}")
    
    # Print summary statistics
    if len(control_actions) > 0:
        all_controls = np.array(control_actions)
        print(f"\nCONTROL ACTION STATISTICS:")
        print(f"Thrust - Mean: {np.mean(all_controls[:, 0]):.3f}, Std: {np.std(all_controls[:, 0]):.3f}, Min: {np.min(all_controls[:, 0]):.3f}, Max: {np.max(all_controls[:, 0]):.3f}")
        print(f"Roll Rate - Mean: {np.mean(all_controls[:, 1]):.3f}, Std: {np.std(all_controls[:, 1]):.3f}, Min: {np.min(all_controls[:, 1]):.3f}, Max: {np.max(all_controls[:, 1]):.3f}")
        print(f"Pitch Rate - Mean: {np.mean(all_controls[:, 2]):.3f}, Std: {np.std(all_controls[:, 2]):.3f}, Min: {np.min(all_controls[:, 2]):.3f}, Max: {np.max(all_controls[:, 2]):.3f}")
        print(f"Yaw Rate - Mean: {np.mean(all_controls[:, 3]):.3f}, Std: {np.std(all_controls[:, 3]):.3f}, Min: {np.min(all_controls[:, 3]):.3f}, Max: {np.max(all_controls[:, 3]):.3f}")
    
    print("="*60)

    # Only plot if we have state trajectory
    if len(state_trajectory) > 1:
        state_history = state_trajectory

        fig = plt.figure(0)
        plt.subplot(111, projection='3d')
        positions = np.array([state[0:3] for state in state_history])

        x_data = positions[:, 0]
        y_data = positions[:, 1]
        z_data = positions[:, 2]

        plt.plot(x_data, y_data, z_data, label='Quadcopter Trajectory')

        # Plotting the reference
        ref_positions = np.array([reference[0:3, j] for j in range(mpc_params['J']+1)])
        plt.plot(ref_positions[:, 0], ref_positions[:, 1], ref_positions[:, 2], label='Reference Trajectory', linestyle='--', color='orange')

        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Quadcopter 3D Trajectory')
        plt.legend()
        plt.grid(True)
        
        # Save the 3D plot since we're using non-interactive backend
        plt.savefig('quadcopter_3d_trajectory.png', dpi=300, bbox_inches='tight')
        print("3D trajectory plot saved as 'quadcopter_3d_trajectory.png'")
        plt.close()  # Close the figure to free memory

        fig = plt.figure(1)
        # The state has time discretization dt_sim. Plot the states over time
        N = len(state_history)
        time = np.arange(N) * sim_params['dt_sim']
        time_reference = np.arange(mpc_params['J']+1) * mpc_params['delta_t']

        # Define consistent colors
        colors = {
            'x': 'tab:blue',
            'y': 'tab:green',
            'z': 'tab:red'
        }

        # --- Position ---
        plt.subplot(411)
        plt.plot(time, [state[0] for state in state_history], label='X Position', color=colors['x'])
        plt.plot(time, [state[1] for state in state_history], label='Y Position', color=colors['y'])
        plt.plot(time, [state[2] for state in state_history], label='Z Position', color=colors['z'])
        plt.plot(time_reference, reference[0, :], label='X Ref', linestyle='--', color=colors['x'])
        plt.plot(time_reference, reference[1, :], label='Y Ref', linestyle='--', color=colors['y'])
        plt.plot(time_reference, reference[2, :], label='Z Ref', linestyle='--', color=colors['z'])
        plt.ylabel('Position (m)')
        plt.legend()

        # --- Velocity ---
        plt.subplot(412)
        plt.plot(time, [state[3] for state in state_history], label='X Velocity', color=colors['x'])
        plt.plot(time, [state[4] for state in state_history], label='Y Velocity', color=colors['y'])
        plt.plot(time, [state[5] for state in state_history], label='Z Velocity', color=colors['z'])
        plt.plot(time_reference, reference[3, :], label='X Vel Ref', linestyle='--', color=colors['x'])
        plt.plot(time_reference, reference[4, :], label='Y Vel Ref', linestyle='--', color=colors['y'])
        plt.plot(time_reference, reference[5, :], label='Z Vel Ref', linestyle='--', color=colors['z'])
        plt.ylabel('Velocity (m/s)')
        plt.legend()

        # --- Orientation ---
        plt.subplot(413)
        plt.plot(time, [np.degrees(state[6]) for state in state_history], label='Roll (ϕ)', color=colors['x'])
        plt.plot(time, [np.degrees(state[7]) for state in state_history], label='Pitch (θ)', color=colors['y'])
        plt.plot(time, [np.degrees(state[8]) for state in state_history], label='Yaw (ψ)', color=colors['z'])
        plt.plot(time_reference, np.degrees(reference[6, :]), label='Roll Ref', linestyle='--', color=colors['x'])
        plt.plot(time_reference, np.degrees(reference[7, :]), label='Pitch Ref', linestyle='--', color=colors['y'])
        plt.plot(time_reference, np.degrees(reference[8, :]), label='Yaw Ref', linestyle='--', color=colors['z'])
        plt.ylabel('Orientation (°)')
        plt.legend()

        # --- Angular Velocity ---
        plt.subplot(414)
        plt.plot(time, [np.degrees(state[9]) for state in state_history], label='Roll Rate', color=colors['x'])
        plt.plot(time, [np.degrees(state[10]) for state in state_history], label='Pitch Rate', color=colors['y'])
        plt.plot(time, [np.degrees(state[11]) for state in state_history], label='Yaw Rate', color=colors['z'])
        plt.plot(time_reference, np.degrees(reference[9, :]), label='Roll Rate Ref', linestyle='--', color=colors['x'])
        plt.plot(time_reference, np.degrees(reference[10, :]), label='Pitch Rate Ref', linestyle='--', color=colors['y'])
        plt.plot(time_reference, np.degrees(reference[11, :]), label='Yaw Rate Ref', linestyle='--', color=colors['z'])
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Velocity (°/s)')
        plt.legend()

        # Save the time series plots since we're using non-interactive backend
        plt.tight_layout()
        plt.savefig('quadcopter_time_series.png', dpi=300, bbox_inches='tight')
        print("Time series plots saved as 'quadcopter_time_series.png'")
        
        # Close the figure to free memory
        plt.close()