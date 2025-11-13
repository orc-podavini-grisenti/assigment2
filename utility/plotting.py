import os
import numpy as np
import matplotlib.pyplot as plt

# ====================== Simple plotting utility ======================
def plot_infinity(t_init, t_final):
    r = 0.1
    t = np.linspace(t_init, t_final, 300)
    x = r * np.cos(2*np.pi*t)
    y = r * 0.5*np.sin(4*np.pi*t)
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, 'x')
    plt.xlim([-0.11, 0.11])
    plt.ylim([-0.06, 0.06])
    plt.title("Infinity-shaped path")
    plt.grid(True)
    plt.show()


def plot_result(N, dt, s_sol, ee_des, ee, dq_sol, q_sol, tau, w_sol, save_dir=None, marker_size=8):
    """
    Plot the results of a trajectory optimization for a robotic manipulator.
    
    This function generates multiple plots showing the evolution of various quantities
    over time, including the trajectory parameter, end-effector position, joint velocities,
    joint positions, joint torques, and auxiliary variables.
    
    Args:
        N (int): Number of time steps in the trajectory
        dt (float): Time step duration in seconds
        s_sol (np.ndarray): Trajectory parameter evolution, shape (N+1,)
        ee_des (np.ndarray): Desired end-effector positions, shape (3, N+1)
        ee (np.ndarray): Actual end-effector positions, shape (3, N+1)
        dq_sol (np.ndarray): Joint velocities, shape (n_joints, N+1)
        q_sol (np.ndarray): Joint positions, shape (n_joints, N+1)
        tau (np.ndarray): Joint torques, shape (n_joints, N)
        w_sol (np.ndarray): Auxiliary optimization variables, shape (N,)
        save_dir (str, optional): Directory path to save the plots. If None, plots are 
                                  displayed but not saved. Default is None.
        marker_size (float, optional): Size of markers (dots and crosses) in plots. 
                                       Default is 8.
    
    Returns:
        None: Displays and optionally saves the generated plots

    Usage:
        Display plots only
        - plot_result(N, dt, s_sol, ee_des, ee, dq_sol, q_sol, tau, w_sol)

        Save plots to directory with default marker size
        - plot_result(N, dt, s_sol, ee_des, ee, dq_sol, q_sol, tau, w_sol, save_dir='./results')

        Save plots with larger markers
        - plot_result(N, dt, s_sol, ee_des, ee, dq_sol, q_sol, tau, w_sol, save_dir='./results', marker_size=12)
    """
    
    # Generate time vector
    tt = np.linspace(0, (N + 1) * dt, N + 1)
    
    # Create save directory if specified and doesn't exist
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Trajectory parameter s over time
    plt.figure(figsize=(10, 4))
    plt.plot([tt[0], tt[-1]], [0, 1], ':', label='straight line', alpha=0.7, linewidth=2.5)
    plt.plot(tt, s_sol, label='s', linewidth=2.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Trajectory parameter s')
    plt.title('Trajectory Parameter Evolution')
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'trajectory_parameter.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 2: End-effector position in xy plane
    plt.figure(figsize=(10, 4))
    plt.plot(ee_des[0,:].T, ee_des[1,:].T, 'rx', label='EE desired', 
             alpha=0.7, markersize=marker_size)
    plt.plot(ee[0,:].T, ee[1,:].T, 'ko', label='EE actual', 
             alpha=0.7, markersize=marker_size)
    plt.xlabel('End-effector pos x [m]')
    plt.ylabel('End-effector pos y [m]')
    plt.title('End-Effector Trajectory (XY Plane)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'ee_trajectory_xy.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: End-effector position components over time
    plt.figure(figsize=(10, 4))
    for i in range(3):
        plt.plot(tt, ee_des[i,:].T, ':', label=f'EE desired {["x", "y", "z"][i]}', alpha=0.7, linewidth=2.5)
        plt.plot(tt, ee[i,:].T, label=f'EE actual {["x", "y", "z"][i]}', alpha=0.7, linewidth=2.5)
    plt.xlabel('Time [s]')
    plt.ylabel('End-effector pos [m]')
    plt.title('End-Effector Position Components')
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'ee_position_components.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 4: Joint velocities over time
    plt.figure(figsize=(10, 4))
    for i in range(dq_sol.shape[0]):
        plt.plot(tt, dq_sol[i,:].T, label=f'Joint {i+1} velocity', alpha=0.7, linewidth=2.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint velocity [rad/s]')
    plt.title('Joint Velocities')
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'joint_velocities.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 5: Joint positions over time
    plt.figure(figsize=(10, 4))
    for i in range(q_sol.shape[0]):
        plt.plot([tt[0], tt[-1]], [q_sol[i,0], q_sol[i,0]], ':', 
                 label=f'Joint {i+1} initial', alpha=0.7, linewidth=2.5)
        plt.plot(tt, q_sol[i,:].T, label=f'Joint {i+1} position', alpha=0.7, linewidth=2.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint position [rad]')
    plt.title('Joint Positions')
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'joint_positions.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 6: Joint torques over time
    plt.figure(figsize=(10, 4))
    for i in range(tau.shape[0]):
        plt.plot(tt[:-1], tau[i,:].T, label=f'Joint {i+1} torque', alpha=0.7, linewidth=2.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint torque [Nm]')
    plt.title('Joint Torques')
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'joint_torques.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 7: Auxiliary optimization variables over time
    plt.figure(figsize=(10, 4))
    plt.plot(tt[:-1], w_sol.T, label='w', alpha=0.7, linewidth=2.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Auxiliary variable w')
    plt.title('Optimization Variables')
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'optimization_variables.png'), dpi=300, bbox_inches='tight')
    plt.show()