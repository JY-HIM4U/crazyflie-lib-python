import control
import numpy as np
import imageio.v3 as iio
import matplotlib
matplotlib.use('TkAgg')
from scipy.linalg import expm, sinm, cosm
import random
import matplotlib.pyplot as plt

random.seed(10)

# Generate map of track by loading the respective png
def load_world(track_name):
    """
    Loads a track from a PNG image and creates safety and target sets for a
    control or path planning problem.

    The function assumes the track is represented by a PNG image where the red
    channel (im_r) represents the 'safe' area and the blue channel (im_b)
    represents the 'target' area. Both channels are used to create a safety_map.
    The blue channel is then modified to represent the target set by subtracting
    the red channel, effectively isolating the blue area.

    Args:
        track_name (str): The name of the track file (without the '.png'
                           extension).

    Returns:
        list: A list containing two NumPy arrays:
              - safe_set: A binary NumPy array representing the safe area.
              - target_set: A binary NumPy array representing the target area.
    """
    track_name= "./worlds/ra_50"
    im = iio.imread(track_name + ".png")   # Loads track from png file
    im_r = np.array(im[:, :, 0]) / 255.0
    im_b = np.array(im[:, :, 2]) / 255.0

    card_X_x = np.shape(im)[0]
    card_X_y = np.shape(im)[1]
    safety_map = np.zeros((card_X_x,card_X_y))
    for x_idx in range(card_X_x):  # for all states in x
        for y_idx in range(card_X_y):  # for all states in y
            if im_r[x_idx,y_idx]>0 or im_b[x_idx,y_idx]>0:
                safety_map[x_idx,y_idx] = 1

    im_b = im_b - im_r
    #plt.figure(0)
    #plt.imshow(safety_map, cmap='hot', interpolation='nearest')
    #plt.show() # Plots track
    safe_set = im_r
    target_set = im_b
    return safe_set, target_set

def system_dynamics(discretization_dt, Q_c, MPC_DUAL_MODE_CONTROL, MDP_LENGTH_OF_CELLS_IN_METERS):
    qcp_kD = 0.1735
    qcp_g = 9.81
    qcp_m = 1.862
    qcp_k = -qcp_kD / qcp_m
    qcp_Ixx = 0.0429
    qcp_Iyy = 0.0437
    qcp_Izz = 0.0753
    qcp_T = 62.06
    qcp_taux = 4.6548
    qcp_tauy = qcp_taux
    qcp_tauz = 1.7

    qcp_A = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, qcp_k, 0, 0, 0, qcp_g * 1/MDP_LENGTH_OF_CELLS_IN_METERS, 0, 0, 0, 0],
                      [0, 0, 0, 0, qcp_k, 0, -qcp_g*1/MDP_LENGTH_OF_CELLS_IN_METERS, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, qcp_k, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    qcp_B = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [qcp_T/qcp_m*1/MDP_LENGTH_OF_CELLS_IN_METERS, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, qcp_taux/qcp_Ixx, 0, 0],
                      [0, 0, qcp_tauy/qcp_Iyy, 0],
                      [0, 0, 0, qcp_tauz/qcp_Izz]])


    mpdt = np.block([[qcp_A * discretization_dt, qcp_B * discretization_dt],
                       [np.zeros_like(np.transpose(qcp_B)), np.zeros((4,4))]])
    exmpdt = expm(mpdt)
    qcp_A_dt = np.copy(exmpdt[0:12, 0:12])
    qcp_B_dt = np.copy(exmpdt[0:12, 12:16])

    Q_lqr = np.eye(12)
    R_lqr = 0.01*np.eye(4)

    if MPC_DUAL_MODE_CONTROL:
        qcp_K_dt, lqr_cost, lqr_eigenvalues_closed_loop = control.dlqr(qcp_A_dt, qcp_B_dt, Q_lqr, R_lqr)
        qcp_A_cl_dt = qcp_A_dt - qcp_B_dt@qcp_K_dt
    else:
        qcp_K_dt = np.zeros((4,12))
        qcp_A_cl_dt = qcp_A_dt

    A_ineq = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [-1,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                       ])

    b_ineq_base = np.array([0,0,
                            0,0,
                            .5,.5,
                            1,1,
                            1,1,
                            1,1,
                            1,1,
                            1,1,
                            1,1,
                            6,6,
                            6,6,
                            6,6,])

    Au_ineq = np.array([[ 1, 0, 0, 0],
                        [-1, 0, 0, 0],
                        [ 0, 1, 0, 0],
                        [ 0,-1, 0, 0],
                        [ 0, 0, 1, 0],
                        [ 0, 0,-1, 0],
                        [ 0, 0, 0, 1],
                        [ 0, 0, 0,-1]])
    bu_ineq = np.array([1, 0, 1, 1, 1, 1, 1, 1])

    # Construct the block matrix
    block_matrix = np.block([
        [qcp_A, Q_c],
        [np.zeros_like(qcp_A), -qcp_A.T]
    ])

    # Compute the matrix exponential
    exp_block_matrix = expm(block_matrix * discretization_dt)

    # Extract Phi and Gamma
    Phi = exp_block_matrix[:12, :12]
    Gamma = exp_block_matrix[:12, 12:]

    # Compute the discrete-time process noise covariance matrix
    qcp_Q_dt = Gamma @ Phi.T


    return [qcp_A_dt,qcp_B_dt, qcp_Q_dt, A_ineq, Au_ineq, bu_ineq, b_ineq_base, qcp_A_cl_dt, qcp_K_dt]