"""
Question 4 Experiment .
Minimum Time Path & Trajectory Tracking with Variable Time Steps 
"""

import time
import casadi as cs
from pathlib import Path

from robot_setup import RobotSetup

from utility.visualization import RobotMotionViewer
from utility.utility import extract_solution, save_tracking_summary
from utility.plotting import plot_result


def run_q4(visualize=True, plot_results=True):
    """Q4: Minimum Time Path Tracking."""
    print("\nQ4: Minimum time path & trajectory tracking")
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

    '''======================= PATH TRACKING OCP FORMULATION ======================='''
    print("\n======================= PATH TRACKING OCP FORMULATION =======================")
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

    # OCP Formulation
    ocp = min_time_path_tracking(opti, X, U, S, W, N, dt, robot_setup.x_init, 
                                 robot_setup.c_path, 
                                 robot_setup.r_path, 
                                 w_v, w_a, w_w, w_final, w_time, 
                                 robot_setup.nq, 
                                 robot_setup.f, 
                                 robot_setup.fk, 
                                 robot_setup.inv_dyn, 
                                 robot_setup.tau_min, 
                                 robot_setup.tau_max)
    opti.minimize(ocp)

    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-4, "ipopt.hessian_approximation":"limited-memory"}
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
    save_path = f'./results/Q4/path_tracking'
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



    '''======================= TRAJECTORY TRACKING OCP FORMULATION ======================='''
    print("\n======================= TRAJECTORY TRACKING OCP FORMULATION =======================")
    # Formulation of the OCP for trajectory tracking with minimum time
    # Create the Optimal Controll SOlver
    opti = cs.Opti()

    # Create Decision Variables and Initialize Opti with Them
    X, U = [], []

    for _ in range(N + 1):
        # Joint State Variable, with boundary ( both position and velocity )
        X += [opti.variable(robot_setup.nx)]    
        opti.subject_to(opti.bounded(robot_setup.lbx, X[-1], robot_setup.ubx))
    for _ in range(N):
        # Control Torque Variable
        U += [opti.variable(robot_setup.nu)]

    # initialize S vector for path progression
    S = []
    S.append(0.0)
    for k in range(1, N + 1):
        if k>0:
            S.append(S[k-1] + 1/N)

    # Add time step as optimisation variable
    dt = opti.variable(1)

    # Time Step Boundary Constraints
    # - dt_min <= dt <= dt_max
    opti.subject_to( opti.bounded(dt_min, dt, dt_max))

    # OCP Formulation
    ocp = min_time_trajectory_tracking(opti, X, U, S, N, dt, robot_setup.x_init, 
                                      robot_setup.c_path, 
                                      robot_setup.r_path, 
                                      w_v, w_a, w_p, w_final, w_time, 
                                      robot_setup.nq, 
                                      robot_setup.f, 
                                      robot_setup.fk, 
                                      robot_setup.inv_dyn, 
                                      robot_setup.tau_min, 
                                      robot_setup.tau_max)
    opti.minimize(ocp)
    
    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-4}
    opti.solver("ipopt", opts)

    t0 = time.time()
    sol = opti.solve()
    solver_time = time.time() - t0
    print(f"Solver time: {solver_time:.2f}s")

    q_sol, dq_sol, u_sol, tau, ee, ee_des, s_sol, w_sol, dt_sol = extract_solution(
        sol, X, U, S, None, N, dt, robot_setup.nq, robot_setup.c_path, 
        robot_setup.r_path, robot_setup.inv_dyn, robot_setup.fk
    )
    
    # Plot results
    save_path = f'./results/Q4/trajectory_tracking'
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





#                 N-1
#     min          ∑   ( w_v*∥dq_k∥^2 + w_a∥u_k∥^2 + w_w∥w_k∥^2 ) * dt +   w_time*dt^2     +     w_final*∥x(N) - x_init∥^2    
#  X,U,S,W,dt     k=0                                                 |-Min Time Cost-|
#               |--------------------- Running Cost ----------------------------------|        |------ Terminal Cost -----|  
#
# subject to    - q_k+1 = q_k + dt*dq_k , dq_k+1 = dq_k + dt*u_k      ∀k ∈[0,N −1]
#               - q_min <= q_k <= q_max,  dq_max <= dq_k <= dq_max    ∀k ∈[1,N]
#               - tau_min <= τ_k <= tau_max                           ∀k ∈[1,N-1]
#               - s_k+1 = s_k + dt*w_k                                ∀k ∈[1,N-1]
#               - y(q_k) = p(s_k)                                     ∀k ∈[1,N]
#               - dt_min <= dt <= dt_max
#               - x(0) = x_init   
#               - s(0) = 0  
#               - s(N) = 1 


def min_time_path_tracking(opti, X, U, S, W, N, dt, x_init,
                            c_path, r_path, w_v, w_a, w_w, w_final, w_time, 
                            nq, f, fk, inv_dyn, tau_min, tau_max):
    '''
    Define running cost and dynamics constraints for path tracking with progress variable s(t).

    Args:
        opti (obj): casadi optimizer
        X (list): State matrix, shape (N+1, nx).
        U (list): Control matrix, shape (N, nu).
        S (list): State matrix, shape (N+1, 1), s ∈ [0,1].
        W (list): Path progression velocity, shape (N+1, 1).
        N (int): Number of time steps.
        dt (float): Time step size variable
        x_init (array-like): Initial state of the system, shape (nx,).
        c_path (array-like): Coordinates of the center of the path [c_x, c_y, c_z].
        r_path (float): Radius or scale factor of the path trajectory.
        w_v: Weight of the velocity in cost function
        w_a: Weight of the acceleration in cost function
        w_w: Weight of the weighting in cost function
        w_final: Weight for cyclic terminal cost
        w_time: Weight for mini time cost
        nq: Number of Joints
        f: Time derivative function of the state 
        fk: Forward Kinematics Function
        inv_dyn: Inverse Dynamic Function
        tau_min (list): Minimum allowable joint torques (control limits).
        tau_max (list): Maximum allowable joint torques (control limits).
    
    Returns:
        cost (casadi.MX/DM): Total running cost
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
       

        # Running Cost Terms 
        # - ( w_v*∥dq_k∥^2 + w_a∥u_k∥^2 + w_w∥w_k∥^2 ) * dt 
        instant_cost = 0
        instant_cost += w_v * cs.sumsqr(X[k][nq:])      # velocity tracking cost term
        instant_cost += w_a * cs.sumsqr(U[k])           # actuation effort cost term
        instant_cost += w_w * cs.sumsqr(W[k])           # path progression speed cost term
        cost += instant_cost * dt

        # Discrete-Time Dynamics constraints
        # - q_k+1 = q_k + dt*dq_k 
        # - dq_k+1 = dq_k + dt*u_k 
        x_next = X[k] + dt * f(X[k], U[k]) 
        opti.subject_to(X[k+1] == x_next)


        # Path Progression constraints
        # - s_k+1 = s_k + dt*w_k
        s_next = S[k] + dt * W[k] 
        opti.subject_to(S[k+1] == s_next)


        # Torque bounds Constra
        # - tau_min <= τ_k <= tau_max
        tau_k = inv_dyn(X[k], U[k]) 
        opti.subject_to(opti.bounded(tau_min, tau_k, tau_max))


        # Min Time Step RUNNING Cost
        # - w_time*dt^2
        cost += w_time * (dt**2)

    
    # Compute the end-effector position at the final state
    q_N = X[-1][0:nq]   # get final joint positions from X[-1]
    ee_pos_N = fk(q_N)  # get final end-effector position from fk(q_N)

    # Constrain ee_pos to lie on the desired path in x, y, z at the end
    # - y(q_N) = p(s_N)
    s_N = S[-1]  # Get final path variable S[-1]
    p_x = c_path[0] + r_path * cs.cos(2 * cs.pi * s_N)
    p_y = c_path[1] + 0.5 * r_path * cs.sin(4 * cs.pi * s_N)
    p_z = c_path[2]
    p_s_N = cs.vertcat(p_x, p_y, p_z)
    opti.subject_to(ee_pos_N == p_s_N)
    
    # terminal cost that penalizes the distance between the initial and final state
    # - w_final*∥x(N) - x_init∥^2
    cost += w_final * cs.sumsqr(X[-1] - X[0]) 

    # Min Time Step TERMINAL Cost
    # - w_time*dt
    # cost += w_time * (dt**2)
        
    return cost







#                 N-1
#     min          ∑   ( w_v*∥dq_k∥^2 + w_a∥u_k∥^2 + w_p*∥y_q - p_s∥^2 ) * dt +   w_time*dt^2    +  ...    
#    X,U,dt       k=0
#               |------------------------------ Running Cost -----------------------------|    
# 
#            ...  dt*w_p*∥y(q_N) - p(s_N)∥^2 + w_final*∥x(N) - x_init∥^2    
#
#               |--------------------- Terminal Cost -------------------| 
#
# subject to    - q_k+1 = q_k + dt*dq_k , dq_k+1 = dq_k + dt*u_k      ∀k ∈[0,N −1]
#               - q_min <= q_k <= q_max,  dq_max <= dq_k <= dq_max    ∀k ∈[1,N]
#               - tau_min <= τ_k <= tau_max                           ∀k ∈[1,N-1]
#               - s_k+1 = s_k + 1/N                                   ∀k ∈[1,N-1]
#               - dt_min <= dt <= dt_max
#               - x(0) = x_init   


def min_time_trajectory_tracking(opti, X, U, S, N, dt, x_init,
                                 c_path, r_path, w_v, w_a, w_p, w_final, w_time, 
                                 nq, f, fk, inv_dyn, tau_min, tau_max):
    '''
    Define running cost and dynamics constraints for path tracking with progress variable s(t).

    Args:
        opti (obj): casadi optimizer
        X (list): State matrix, shape (N+1, nx).
        U (list): Control matrix, shape (N, nu).
        S (list): State matrix, Fix
        W (list): Path progression velocity, shape (N+1, 1).
        N (int): Number of time steps.
        dt (float): Time step size variable
        x_init (array-like): Initial state of the system, shape (nx,).
        c_path (array-like): Coordinates of the center of the path [c_x, c_y, c_z].
        r_path (float): Radius or scale factor of the path trajectory.
        w_v: Weight of the velocity in cost function
        w_a: Weight of the acceleration in cost function
        w_p: Weight of the path tracking in cost function
        w_final: Weight for cyclic terminal cost
        w_time: Weight for mini time cost
        nq: Number of Joints
        f: Time derivative function of the state 
        fk: Forward Kinematics Function
        inv_dyn: Inverse Dynamic Function
        tau_min (list): Minimum allowable joint torques (control limits).
        tau_max (list): Maximum allowable joint torques (control limits).
    
    Returns:
        cost (casadi.MX/DM): Total running cost
    '''

    # Initial and Terminal Constraints
    # - x(0) = x_init       Constrain the initial state X[0] to be equal to the initial condition x_init 
    opti.subject_to(X[0] == x_init)

    cost = 0.0
    for k in range(N):
        # Compute the end-effector position using forward kinematics
        q_k = X[k][0:nq] 
        ee_pos = fk(q_k)
        
        # Path Constraint
        s_k = S[k]              # current path progress
        # derive end effectore position from the path progress
        p_x = c_path[0] + r_path * cs.cos(2 * cs.pi * s_k)          # p_x(s)=c_x ​+ rcos(2πs)
        p_y = c_path[1] + 0.5 * r_path * cs.sin(4 * cs.pi * s_k)    # p_y(s)=c_y+0.5rsine(4πs)
        p_z = c_path[2]                                             # p_z(s)=c_z
        p_s = cs.vertcat(p_x, p_y, p_z)
       
        # Running Cost Terms 
        # ( w_v*∥dq_k∥^2 + w_a∥u_k∥^2 + w_p*∥y_q - p_s∥^2 ) * dt 
        instant_cost = 0
        instant_cost += w_v * cs.sumsqr(X[k][nq:])      # velocity tracking cost term
        instant_cost += w_a * cs.sumsqr(U[k])           # actuation effort cost term
        instant_cost += w_p * cs.sumsqr(ee_pos - p_s)   # path tracking cost term
        cost += instant_cost * dt


        # Discrete-Time Dynamics constraints
        # - q_k+1 = q_k + dt*dq_k 
        # - dq_k+1 = dq_k + dt*u_k 
        x_next = X[k] + dt * f(X[k], U[k]) 
        opti.subject_to(X[k+1] == x_next)


        # Torque bounds Constra
        # - tau_min <= τ_k <= tau_max
        tau_k = inv_dyn(X[k], U[k]) 
        opti.subject_to(opti.bounded(tau_min, tau_k, tau_max))


        # Min Time Step RUNNING Cost
        # - w_time*dt^2
        cost += w_time * (dt**2)

    
    # Compute the end-effector position at the final state
    q_N = X[-1][0:nq]   # get final joint positions from X[-1]
    ee_pos_N = fk(q_N)  # get final end-effector position from fk(q_N)

    # Constrain ee_pos to lie on the desired path in x, y, z at the end
    # - y(q_N) = p(s_N)
    s_N = S[-1]  # Get final path variable S[-1]
    p_x = c_path[0] + r_path * cs.cos(2 * cs.pi * s_N)
    p_y = c_path[1] + 0.5 * r_path * cs.sin(4 * cs.pi * s_N)
    p_z = c_path[2]
    p_s_N = cs.vertcat(p_x, p_y, p_z)

    final_step_cost = w_p * cs.sumsqr(ee_pos_N - p_s_N)   # path tracking cost term
    cost += final_step_cost * dt
    
    
    # terminal cost that penalizes the distance between the initial and final state
    # - w_final*∥x(N) - x_init∥^2
    cost += w_final * cs.sumsqr(X[-1] - x_init) 

    # Min Time Step TERMINAL Cost
    # - w_time*dt^2
    # cost += w_time * (dt**2)
        
    return cost

