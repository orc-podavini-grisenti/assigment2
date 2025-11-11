#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import casadi as cs
import time
from time import sleep

from adam.casadi.computations import KinDynComputations
from example_robot_data.robots_loader import load
import orc.optimal_control.casadi_adam.conf_ur5 as conf_ur5
from orc.utils.robot_wrapper import RobotWrapper
from orc.utils.robot_simulator import RobotSimulator
from orc.utils.viz_utils import addViewerSphere, applyViewerConfiguration

from utility import extract_solution, save_tracking_summary
from plotting_utility import plot_infinity, plot_result


# ====================== Robot and Dynamics Setup ======================
robot = load("ur5")
joints_name_list = [s for s in robot.model.names[1:]]
nq = len(joints_name_list)                                  # nq = number of joints
nx = 2 * nq                                                 # nx = state space dimension, a joint state x = [q, dq], 2 variable for each joint

dt = 0.02                                                   # dt = time step
N = 100                                                     # N = number of steps
q0 = np.zeros(nq)
dq0 = np.zeros(nq)
x_init = np.concatenate([q0, dq0])
w_max = 5
frame_name = "ee_link"                                      # end effecto frame name
r_path = 0.2                                                # path radius

# CasADi symbolic variables
q = cs.SX.sym('q', nq)
dq = cs.SX.sym('dq', nq)
ddq = cs.SX.sym('ddq', nq)
state = cs.vertcat(q, dq)
rhs = cs.vertcat(dq, ddq)
f = cs.Function('f', [state, ddq], [rhs])

# Inverse dynamics
kinDyn = KinDynComputations(robot.urdf, joints_name_list)
H_b = cs.SX.eye(4)
v_b = cs.SX.zeros(6)
bias_forces = kinDyn.bias_force_fun()
mass_matrix = kinDyn.mass_matrix_fun()
h = bias_forces(H_b, q, v_b, dq)[6:]
M = mass_matrix(H_b, q)[6:, 6:]
tau = M @ ddq + h
inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])

# Forward kinematics
fk_fun = kinDyn.forward_kinematics_fun(frame_name)
ee_pos = fk_fun(H_b, q)[:3, 3]
fk = cs.Function('fk', [q], [ee_pos])

y = fk(q0)
c_path = np.array([y[0]-r_path, y[1], y[2]]).squeeze()

lbx = robot.model.lowerPositionLimit.tolist() + (-robot.model.velocityLimit).tolist()
ubx = robot.model.upperPositionLimit.tolist() + robot.model.velocityLimit.tolist()
tau_min = (-robot.model.effortLimit).tolist()
tau_max = robot.model.effortLimit.tolist()


# ====================== Optimization Problem ======================
def create_decision_variables(N, nx, nu, lbx, ubx):
    opti = cs.Opti()
    X, U, S, W = [], [], [], []
    for _ in range(N + 1):
        X += [opti.variable(nx)]
        opti.subject_to(opti.bounded(lbx, X[-1], ubx))
        S += [opti.variable(1)]
        opti.subject_to(opti.bounded(0, S[-1], 1))
    for _ in range(N):
        U += [opti.variable(nu)]
        W += [opti.variable(1)]
    return opti, X, U, S, W


def define_running_cost_and_dynamics(opti, X, U, S, W, N, dt, x_init,
                                     c_path, r_path, w_v, w_a, w_w,
                                     tau_min, tau_max):
    '''
    Define running cost and dynamics constraints for path tracking with progress variable s(t).

    Args:
        opti (obj): casadi optimizer
        X (list): State matrix, shape (N+1, nx).
        U (list): Control matrix, shape (N, nu).
        S (list): State matrix, shape (N+1, 1), s ∈ [0,1].
        W (list): Path progression velocity, shape (N+1, 1).
        N (int): Number of time steps.
        dt (float): Time step size. 
        x_init (array-like): Initial state of the system, shape (nx,).
        c_path (array-like): Coordinates of the center of the path [c_x, c_y, c_z].
        r_path (float): Radius or scale factor of the path trajectory.
        w_v: Weight of the velocity in cost function
        w_a: Weight of the acceleration in cost function
        w_w: Weight of the weighting in cost function
        tau_min (list): Minimum allowable joint torques (control limits).
        tau_max (list): Maximum allowable joint torques (control limits).
    
    Returns:
        cost (casadi.MX/DM): The total running cost expression, including tracking, actuation, and path progression terms.
    '''

    # Initial and Terminal Constraints
    # - x(0) = x_init       Constrain the initial state X[0] to be equal to the initial condition x_init 
    # - s(0) = 0            Constraint the initial the path variable S[0] to 0.0
    # - s(N) = 1            Constrain the final path variable S[-1] to be 1.0
    opti.subject_to(X[0] == x_init)
    opti.subject_to(S[0] == 0.0)
    opti.subject_to(S[-1] == 1.0)

    cost = 0.0
    for k in range(N):
        # Compute the end-effector position using forward kinematics
        q_k = X[k][0:nq] 
        ee_pos = fk(q_k)
        
        # Path Constraint
        # - y(q_k) = p(s_k)     Constrain ee_pos to lie on the desired path in x, y, z
        s_k = S[k]              # current path progress
        # derive end effectore position from the path progress
        p_x = c_path[0] + r_path * cs.cos(2 * cs.pi * s_k)          # p_x(s)=c_x ​+ rcos(2πs)
        p_y = c_path[1] + 0.5 * r_path * cs.sin(4 * cs.pi * s_k)    # p_y(s)=c_y+0.5rsine(4πs)
        p_z = c_path[2]                                             # p_z(s)=c_z
        p_s = cs.vertcat(p_x, p_y, p_z) 
        opti.subject_to(ee_pos == p_s)
       

        # Cost Terms 
        # w_v*∥dq_k∥^2 + w_a*∥u_k∥^2 + w_w*∥w_k∥^2 
        cost += w_v * cs.sumsqr(X[k][nq:])      # velocity tracking cost term
        cost += w_a * cs.sumsqr(U[k])           # actuation effort cost term
        cost += w_w * cs.sumsqr(W[k])           # path progression speed cost term
        

        # Discrete-Time Dynamics constraints
        # - q_k+1 = q_k + ∆t*dq_k 
        # - dq_k+1 = dq_k + ∆t*u_k 
        x_next = X[k] + dt * f(X[k], U[k]) 
        opti.subject_to(X[k+1] == x_next)


        # Path Progression constraints
        # - s_k+1 = s_k + ∆t*w_k
        s_next = S[k] + dt * W[k] 
        opti.subject_to(S[k+1] == s_next)


        # Torque bounds Constra
        # - tau_min <= τ_k <= tau_max
        tau_k = inv_dyn(X[k], U[k]) 
        opti.subject_to(opti.bounded(tau_min, tau_k, tau_max))


        # NB: Missing Joint Position and Velocity Bounds:
        # - q_min <= q_k <= q_max
        # - dq_min <= dq_k <= dq_max
        
        
    return cost

def define_terminal_cost_and_constraints(opti, X, S, c_path, r_path, w_final):
    # TODO: Compute the end-effector position at the final state
    q_N = X[-1][0:nq]   # get final joint positions from X[-1]
    ee_pos_N = fk(q_N)  # get final end-effector position from fk(q_N)

    # TODO: Constrain ee_pos to lie on the desired path in x, y, z at the end
    s_N = S[-1]  # Get final path variable S[-1]
    p_x = c_path[0] + r_path * cs.cos(2 * cs.pi * s_N)
    p_y = c_path[1] + 0.5 * r_path * cs.sin(4 * cs.pi * s_N)
    p_z = c_path[2]
    p_s_N = cs.vertcat(p_x, p_y, p_z)
    opti.subject_to(ee_pos_N == p_s_N)

    cost = 0
    
    # For Question 2, you will add the cyclic cost here
    # cost += w_final * cs.sumsqr(X[-1] - x_init) 
    
    return cost


def create_and_solve_ocp(N, nx, nq, lbx, ubx, dt, x_init,
                         c_path, r_path, w_v, w_a, w_w, w_final,
                         tau_min, tau_max):
    opti, X, U, S, W = create_decision_variables(N, nx, nq, lbx, ubx)
    running_cost = define_running_cost_and_dynamics(opti, X, U, S, W, N, dt, x_init,
                                                    c_path, r_path, w_v, w_a, w_w,
                                                    tau_min, tau_max)
    terminal_cost = define_terminal_cost_and_constraints(opti, X, S, c_path, r_path, w_final)
    opti.minimize(running_cost + terminal_cost)

    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-4}
    opti.solver("ipopt", opts)

    t0 = time.time()
    sol = opti.solve()
    print(f"Solver time: {time.time() - t0:.2f}s")
    return sol, X, U, S, W




# ====================== Simulation and Visualization ======================
r = RobotWrapper(robot.model, robot.collision_model, robot.visual_model)
simu = RobotSimulator(conf_ur5, r)
simu.init(q0, dq0)
simu.display(q0)

REF_SPHERE_RADIUS = 0.02
EE_REF_SPHERE_COLOR = np.array([1, 0, 0, .5])


def display_motion(q_traj, ee_des_traj):
    for i in range(N + 1):
        t0 = time.time()
        simu.display(q_traj[:, i])
        addViewerSphere(r.viz, f'world/ee_ref_{i}', REF_SPHERE_RADIUS, EE_REF_SPHERE_COLOR)
        applyViewerConfiguration(r.viz, f'world/ee_ref_{i}', ee_des_traj[:, i].tolist() + [0, 0, 0, 1.])
        t1 = time.time()
        if(t1-t0 < dt):
            sleep(dt - (t1-t0))



# ====================== Main Execution ======================
if __name__ == "__main__":
    print("Plotting reference infinity curve...")
    plot_infinity(0, 1)
    
    input("Press ENTER to continue...")

    log_w_v, log_w_a, log_w_w, log_w_final = -3, -3, -2, 0
    log_w_p = 2 #Log of trajectory tracking cost 


    sol, X, U, S, W = create_and_solve_ocp(
        N, nx, nq, lbx, ubx, dt, x_init, c_path, r_path,
        10**log_w_v, 10**log_w_a, 10**log_w_w, 10**log_w_final,
        tau_min, tau_max
    )

    q_sol, dq_sol, u_sol, tau, ee, ee_des, s_sol, w_sol = extract_solution(
        sol, X, U, S, W, N, nq, c_path, r_path, inv_dyn, fk 
    )

    print("Displaying robot motion...")
    for i in range(3):
        display_motion(q_sol, ee_des)

    # Plot results
    save_path = './results/Q1'
    plot_result(N, dt, s_sol, ee_des, ee, dq_sol, q_sol, tau, w_sol, save_dir=save_path, marker_size=12)

    # Summary results
    filename = save_path + '/summary.txt'
    save_tracking_summary(q_sol, tau, ee, ee_des, s_sol, dt, N, joints_name_list, tau_min, tau_max, filename)