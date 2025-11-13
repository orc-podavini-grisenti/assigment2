"""
Experiment runner module.
Contains functions to run each of the 4 experiments (Q1-Q4).
"""
import time
import numpy as np
import casadi as cs
from pathlib import Path

from robot_setup import RobotSetup

from utility. visualization import RobotMotionViewer
from utility.utility import extract_solution, save_tracking_summary
from utility.plotting import plot_result

# Import formulation functions
from ocp_formulations.track_path import (
    running_cost_and_dynamics,
    terminal_cost_and_constraints,
    cyclic_terminal_cost_and_constraints
)
from ocp_formulations.track_trajectory import (
    hard_cyclic_trajectory_tracking,
    soft_cyclic_trajectory_tracking
)
from ocp_formulations.min_time_formulation import min_time_path_tracking



def run_q1(visualize=True, plot_results=True):
    """Q1: Path Tracking with Terminal Cost."""
    print("\nQ1: Path tracking with standard terminal cost")
    print("Minimizes velocity, torque, and path deviation with terminal cost")
    
    """Generic experiment runner."""
    N = 100
    dt = 0.02
    robot_setup = RobotSetup()

    viewer = RobotMotionViewer(robot_setup)
    input("\nPress ENTER to continue...")
    
    # Weights
    w_v = 10**-3,      # Joint velocity cost
    w_a = 10**-3,      # Input torque cost
    w_w = 10**-2,      # Path cost
    

    # Create the Optimal Controll SOlver
    opti = cs.Opti()

    # Create Decision Variables and Initialize Opti with Them
    X, U, S, W = [], [], [], []

    for _ in range(N + 1):
        # Joint State Variable, with boundary ( both position and velocity )
        X += [opti.variable(robot_setup.nx)]    
        opti.subject_to(opti.bounded(robot_setup.lbx, X[-1], robot_setup.ubx))
        # Path Status Varaible
        S += [opti.variable(1)]
        opti.subject_to(opti.bounded(0, S[-1], 1))
    for _ in range(N):
        # Control Torque Variable
        U += [opti.variable(robot_setup.nu)]
        # Path Gain Variable
        W += [opti.variable(1)]
    
    running_cost = running_cost_and_dynamics(opti, X, U, S, W, N, dt, robot_setup.x_init,
                                            robot_setup.c_path, robot_setup.r_path, 
                                            w_v, w_a, w_w,
                                            robot_setup.nq, robot_setup.f, 
                                            robot_setup.fk, robot_setup.inv_dyn, 
                                            robot_setup.tau_min, robot_setup.tau_max)
    
    terminal_cost = terminal_cost_and_constraints(opti, X, S, robot_setup.c_path, 
                                                robot_setup.r_path, robot_setup.nq, robot_setup.fk)      
    
    opti.minimize(running_cost + terminal_cost)

    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-4}
    opti.solver("ipopt", opts)

    t0 = time.time()
    sol = opti.solve()
    solver_time = time.time() - t0
    print(f"Solver time: {solver_time:.2f}s")

    q_sol, dq_sol, u_sol, tau, ee, ee_des, s_sol, w_sol, dt_sol = extract_solution(
        sol, X, U, S, W, N, dt, robot_setup.nq, robot_setup.c_path, 
        robot_setup.r_path, robot_setup.inv_dyn, robot_setup.fk
    )
    
    # Visualize
    if visualize:
        print("\nDisplaying robot motion...")
        viewer.display(q_sol, ee_des, N, dt)
    
    # Plot results
    save_path = f'./results/Q1'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    if plot_results:
        plot_result(N, dt_sol, s_sol, ee_des, ee, dq_sol, q_sol, tau, w_sol,
                   save_dir=save_path, marker_size=12)
    
    # Save summary
    filename = f'{save_path}/summary.txt'
    save_tracking_summary(
        q_sol, tau, ee, ee_des, s_sol, dt_sol, N, solver_time,
        robot_setup.joints_name_list, robot_setup.tau_min, robot_setup.tau_max, filename
    )
    
    print(f"\nResults saved to: {save_path}")

    



def run_q2(visualize=True, plot_results=True):
    """Q2: Path Tracking with Cyclic Terminal Cost."""
    print("\nQ2: Path tracking with cyclic terminal cost")
    print("Same as Q1 but with cyclic constraint to return to initial state")
    
    """Generic experiment runner."""
    N = 100
    dt = 0.02
    robot_setup = RobotSetup()

    viewer = RobotMotionViewer(robot_setup)
    input("\nPress ENTER to continue...")
    
    # Weights
    w_v = 10**-3,      # Joint velocity cost
    w_a = 10**-3,      # Input torque cost
    w_w = 10**-2,      # Path cost
    w_final = 10**0    # Cyclic terminal cost
    

    # Create the Optimal Controll SOlver
    opti = cs.Opti()

    # Create Decision Variables and Initialize Opti with Them
    X, U, S, W = [], [], [], []

    for _ in range(N + 1):
        # Joint State Variable, with boundary ( both position and velocity )
        X += [opti.variable(robot_setup.nx)]    
        opti.subject_to(opti.bounded(robot_setup.lbx, X[-1], robot_setup.ubx))
        # Path Status Varaible
        S += [opti.variable(1)]
        opti.subject_to(opti.bounded(0, S[-1], 1))
    for _ in range(N):
        # Control Torque Variable
        U += [opti.variable(robot_setup.nu)]
        # Path Gain Variable
        W += [opti.variable(1)]
    
    running_cost = running_cost_and_dynamics(opti, X, U, S, W, N, dt, robot_setup.x_init,
                                            robot_setup.c_path, robot_setup.r_path, 
                                            w_v, w_a, w_w,
                                            robot_setup.nq, robot_setup.f, 
                                            robot_setup.fk, robot_setup.inv_dyn, 
                                            robot_setup.tau_min, robot_setup.tau_max)
    
    terminal_cost = cyclic_terminal_cost_and_constraints(opti, X, S, robot_setup.x_init, 
                                                        robot_setup.c_path, robot_setup.r_path, 
                                                        w_final, robot_setup.nq, robot_setup.fk )    
    
    opti.minimize(running_cost + terminal_cost)

    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-4}
    opti.solver("ipopt", opts)

    t0 = time.time()
    sol = opti.solve()
    solver_time = time.time() - t0
    print(f"Solver time: {solver_time:.2f}s")

    q_sol, dq_sol, u_sol, tau, ee, ee_des, s_sol, w_sol, dt_sol = extract_solution(
        sol, X, U, S, W, N, dt, robot_setup.nq, robot_setup.c_path, 
        robot_setup.r_path, robot_setup.inv_dyn, robot_setup.fk
    )
    
    # Visualize
    if visualize:
        print("\nDisplaying robot motion...")
        viewer.display(q_sol, ee_des, N, dt)
    
    # Plot results
    save_path = f'./results/Q2/w_final_{w_final}'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    if plot_results:
        plot_result(N, dt_sol, s_sol, ee_des, ee, dq_sol, q_sol, tau, w_sol,
                   save_dir=save_path, marker_size=12)
    
    # Save summary
    filename = f'{save_path}/summary.txt'
    save_tracking_summary(
        q_sol, tau, ee, ee_des, s_sol, dt_sol, N, solver_time,
        robot_setup.joints_name_list, robot_setup.tau_min, robot_setup.tau_max, filename
    )
    
    print(f"\nResults saved to: {save_path}")




def run_q3(visualize=True, plot_results=True):
    """Q3: Cyclic Trajectory Tracking with Soft Constraints."""
    print("\nQ3: Cyclic trajectory tracking (soft constraints)")
    print("Tracks specific trajectory points with position error penalty")
    
    """Generic experiment runner."""
    N = 100
    dt = 0.02
    robot_setup = RobotSetup()

    viewer = RobotMotionViewer(robot_setup)
    input("\nPress ENTER to continue...")
    
    # Weights
    w_v = 10**-3,      # Joint velocity cost
    w_a = 10**-3,      # Input torque cost
    w_p = 10**2,       # Trajectory cost
    w_final = 10**0    # Cyclic terminal cost
    
    # Create the Optimal Controll SOlver
    opti_h = cs.Opti()      # Optimizer for Hard Constraint OCP
    opti_s = cs.Opti()      # Optimizer for Soft Constraint OCP


    # Hard Contraint 
    print('\n HARD CONSTRAINT TRAJECTORY')
    # Create Decision Variables and Initialize Opti with Them
    X, U, S = [], [], []

    for _ in range(N + 1):
        # Joint State Variable, with boundary ( both position and velocity )
        X += [opti_h.variable(robot_setup.nx)]    
        opti_h.subject_to(opti_h.bounded(robot_setup.lbx, X[-1], robot_setup.ubx))
        # Path Status Varaible
        S += [opti_h.variable(1)]
        opti_h.subject_to(opti_h.bounded(0, S[-1], 1))
    for _ in range(N):
        # Control Torque Variable
        U += [opti_h.variable(robot_setup.nu)]
    
    total_cost = hard_cyclic_trajectory_tracking(opti_h, X, U, S, N, dt, robot_setup.x_init, 
                                                robot_setup.c_path, robot_setup.r_path, 
                                                w_v, w_a, w_final, w_p, 
                                                robot_setup.nq, robot_setup.f, 
                                                robot_setup.fk, robot_setup.inv_dyn, 
                                                robot_setup.tau_min, robot_setup.tau_max )
    try: 
        opti_h.minimize(total_cost)
        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-4}
        opti_h.solver("ipopt", opts)
        sol = opti_h.solve()
    except RuntimeError as e:
        print('Failed Hard Contraint OCP; ', e.args)

   
    # Soft Contraint 
    print('\n SOFT CONSTRAINT TRAJECTORY')
    # Create Decision Variables and Initialize Opti with Them
    X, U, S = [], [], []

    for _ in range(N + 1):
        # Joint State Variable, with boundary ( both position and velocity )
        X += [opti_s.variable(robot_setup.nx)]    
        opti_s.subject_to(opti_s.bounded(robot_setup.lbx, X[-1], robot_setup.ubx))
        # Path Status Varaible
        S += [opti_s.variable(1)]
        opti_s.subject_to(opti_s.bounded(0, S[-1], 1))
    for _ in range(N):
        # Control Torque Variable
        U += [opti_s.variable(robot_setup.nu)]

    total_cost = soft_cyclic_trajectory_tracking(opti_s, X, U, S, N, dt, robot_setup.x_init, 
                                                robot_setup.c_path, robot_setup.r_path, 
                                                w_v, w_a, w_final, w_p, 
                                                robot_setup.nq, robot_setup.f, 
                                                robot_setup.fk, robot_setup.inv_dyn, 
                                                robot_setup.tau_min, robot_setup.tau_max )
    opti_s.minimize(total_cost)      
    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-4}
    opti_s.solver("ipopt", opts)

    t0 = time.time()
    sol = opti_s.solve()
    solver_time = time.time() - t0
    print(f"Solver time: {solver_time:.2f}s")

    q_sol, dq_sol, u_sol, tau, ee, ee_des, s_sol, w_sol, dt_sol = extract_solution(
        sol, X, U, S, None , N, dt, robot_setup.nq, robot_setup.c_path, 
        robot_setup.r_path, robot_setup.inv_dyn, robot_setup.fk
    )
    
    # Visualize
    if visualize:
        print("\nDisplaying robot motion...")
        viewer.display(q_sol, ee_des, N, dt)
    
    # Plot results
    save_path = f'./results/Q3'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    if plot_results:
        plot_result(N, dt_sol, s_sol, ee_des, ee, dq_sol, q_sol, tau, w_sol,
                   save_dir=save_path, marker_size=12)
    
    # Save summary
    filename = f'{save_path}/summary.txt'
    save_tracking_summary(
        q_sol, tau, ee, ee_des, s_sol, dt_sol, N, solver_time,
        robot_setup.joints_name_list, robot_setup.tau_min, robot_setup.tau_max, filename
    )
    
    print(f"\nResults saved to: {save_path}")




def run_q4(visualize=True, plot_results=True):
    """Q4: Minimum Time Path Tracking."""
    print("\nQ4: Minimum time path tracking")
    print("Optimizes trajectory duration with variable time steps")
    
    """Generic experiment runner."""
    N = 100
    robot_setup = RobotSetup()

    viewer = RobotMotionViewer(robot_setup)
    input("\nPress ENTER to continue...")
    
    w_p = 10**2
    w_v = 10**-6
    w_a = 10**-6
    w_w = 10**-6
    w_final = 10**-6
    w_time = 10**-1    
    
    dt_min = 0.001
    dt_max = 0.1

    # Create the Optimal Controll SOlver
    opti = cs.Opti()

    # Create Decision Variables and Initialize Opti with Them
    X, U, S, W= [], [], [], []

    for _ in range(N + 1):
        # Joint State Variable, with boundary ( both position and velocity )
        X += [opti.variable(robot_setup.nx)]    
        opti.subject_to(opti.bounded(robot_setup.lbx, X[-1], robot_setup.ubx))
        # Path Status Varaible
        S += [opti.variable(1)]
        opti.subject_to(opti.bounded(0, S[-1], 1))
    for _ in range(N):
        # Control Torque Variable
        U += [opti.variable(robot_setup.nu)]
        # Path Gain Variable
        W += [opti.variable(1)]

    # Add time step as optimisation variable
    dt = opti.variable(1)

    # Time Step Boundary Constraints
    # - dt_min <= dt <= dt_max
    opti.subject_to( opti.bounded(dt_min, dt, dt_max))


    total_cost = min_time_path_tracking(opti, X, U, S, W, N, dt, robot_setup.x_init, 
                                        robot_setup.c_path, robot_setup.r_path, 
                                        w_v, w_a, w_w, w_final, w_time, robot_setup.nq, 
                                        robot_setup.f, robot_setup.fk, robot_setup.inv_dyn, 
                                        robot_setup.tau_min, robot_setup.tau_max)
    opti.minimize(total_cost)

    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-4}
    opti.solver("ipopt", opts)

    t0 = time.time()
    sol = opti.solve()
    solver_time = time.time() - t0
    print(f"Solver time: {solver_time:.2f}s")

    q_sol, dq_sol, u_sol, tau, ee, ee_des, s_sol, w_sol, dt_sol = extract_solution(
        sol, X, U, S, W, N, dt, robot_setup.nq, robot_setup.c_path, 
        robot_setup.r_path, robot_setup.inv_dyn, robot_setup.fk
    )
    
    # Plot results
    save_path = f'./results/Q4'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    if plot_results:
        plot_result(N, dt_sol, s_sol, ee_des, ee, dq_sol, q_sol, tau, w_sol,
                   save_dir=save_path, marker_size=12)
    
    # Save summary
    filename = f'{save_path}/summary.txt'
    save_tracking_summary(
        q_sol, tau, ee, ee_des, s_sol, dt_sol, N, solver_time,
        robot_setup.joints_name_list, robot_setup.tau_min, robot_setup.tau_max, filename
    )

    # Visualize
    if visualize:
        print("\nDisplaying robot motion...")
        viewer.display(q_sol, ee_des, N, dt_sol)
    
    print(f"\nResults saved to: {save_path}")

