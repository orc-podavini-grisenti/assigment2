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

from utility.utility import extract_solution, save_tracking_summary
from utility.plotting import plot_infinity, plot_result

from track_path import running_cost_and_dynamics, terminal_cost_and_constraints, cyclic_terminal_cost_and_constraints
from track_trajectory import hard_cyclic_trajectory_tracking, soft_cyclic_trajectory_tracking
from ocp_formulations.min_time_formulation import min_time_path_tracking

# ====================== Robot and Dynamics Setup ======================
robot = load("ur5")
joints_name_list = [s for s in robot.model.names[1:]]
nq = len(joints_name_list)                                  # nq = number of joints
nx = 2 * nq                                                 # nx = state space dimension, a joint state x = [q, dq], 2 variable for each joint

# dt = 0.02                                                   # dt = time step
dt_min = 0.001
dt_max = 0.1

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
f = cs.Function('f', [state, ddq], [rhs])                   # time derivative function of the state 

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


def create_and_solve_ocp(N, nx, nq, lbx, ubx, dt, x_init,
                         c_path, r_path, w_v, w_a, w_w, w_final, w_p, w_time,
                         tau_min, tau_max):
    opti, X, U, S, W = create_decision_variables(N, nx, nq, lbx, ubx)

    ''' Q1 & Q2 Path Tracking Optimisation Problem
    running_cost = running_cost_and_dynamics(opti, X, U, S, W, N, dt, x_init,
                                                    c_path, r_path, w_v, w_a, w_w,
                                                    nq, f, fk, inv_dyn, tau_min, tau_max)
    # Q1 Terminal Cost
    # terminal_cost = terminal_cost_and_constraints(opti, X, S, c_path, r_path, nq, fk)      
    # Q2 Terminal Cost
    terminal_cost = cyclic_terminal_cost_and_constraints(opti, X, S, x_init, c_path, r_path, w_final, nq, fk )    # Q2
    opti.minimize(running_cost + terminal_cost)
    '''

    # Q3 
    # Hard Contraint 
    # total_cost = hard_cyclic_trajectory_tracking(opti, X, U, S, N, dt, x_init, 
    #                                   c_path, r_path, w_v, w_a, w_final, w_p, 
    #                                   nq, f, fk, inv_dyn, tau_min, tau_max )
    # Soft Contraint 
    '''
    total_cost = soft_cyclic_trajectory_tracking(opti, X, U, S, N, dt, x_init, 
                                       c_path, r_path, w_v, w_a, w_final, w_p, 
                                       nq, f, fk, inv_dyn, tau_min, tau_max )
    opti.minimize(total_cost)
    '''

    # Q4
    total_cost = min_time_path_tracking(opti, X, U, S, W, N, dt_min, dt_max, x_init, c_path, 
                                        r_path, w_v, w_a, w_w, w_final, w_time, nq, f, fk, inv_dyn, 
                                        tau_min, tau_max)
    opti.minimize(total_cost)

    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-4}
    opti.solver("ipopt", opts)

    t0 = time.time()
    sol = opti.solve()
    solver_time = time.time() - t0
    print(f"Solver time: {solver_time:.2f}s")
    return sol, X, U, S, W, solver_time




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

    # w_v = Weight Joint Velocity Cost
    # w_a = Weight Input Torque Cost
    # w_w = Weight Path Cost
    # w_final = Weight Final Configuration Cost
    # w_p = Weight Trajectory Error Cost ( EE Psition Error )
    # w_time = Weight Min Time Cost
    
    # Q1, Q2
    # log_w_v, log_w_a, log_w_w, log_w_final = -3, -3, -2, 0
    # log_w_p = 2 #Log of trajectory tracking cost  # Q3

    # Q4
    log_w_p, log_w_v, log_w_a, log_w_w, log_w_final = 2, -6, -6, -6, -6
    log_w_time = -1
    


    sol, X, U, S, W , solver_timer = create_and_solve_ocp(
        N, nx, nq, lbx, ubx, dt, x_init, c_path, r_path,
        10**log_w_v, 10**log_w_a, 10**log_w_w, 10**log_w_final,
        10**log_w_p, 10**log_w_time, tau_min, tau_max
    )

    q_sol, dq_sol, u_sol, tau, ee, ee_des, s_sol, w_sol, dt_sol = extract_solution(
        sol, X, U, S, W, N, dt, nq, c_path, r_path, inv_dyn, fk 
    )

    print("Displaying robot motion...")
    display_motion(q_sol, ee_des)

    # Plot results
    save_path = './results/Q4/path_tracking'
    plot_result(N, dt_sol, s_sol, ee_des, ee, dq_sol, q_sol, tau, w_sol, save_dir = save_path, marker_size=12)

    # Summary results
    filename = save_path + '/summary.txt'
    save_tracking_summary(q_sol, tau, ee, ee_des, s_sol, dt_sol, N, solver_timer, joints_name_list, tau_min, tau_max, filename)