"""
Question 3 Experiment .
Cyclic Trajectory Tracking with Hard & Soft Constraints 
"""


import time
import numpy as np
import casadi as cs
from pathlib import Path

from robot_setup import RobotSetup

from utility.visualization import RobotMotionViewer
from utility.utility import extract_solution, save_tracking_summary
from utility.plotting import plot_result


def run_q3(visualize=True, plot_results=True):
    """Q3: Cyclic Trajectory Tracking with Hard & Soft Constraints."""
    print("\nQ3: Cyclic trajectory tracking")
    print("Tracks specific trajectory points")
    
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
    X, U = [], []

    for _ in range(N + 1):
        # Joint State Variable, with boundary ( both position and velocity )
        X += [opti_h.variable(robot_setup.nx)]    
        opti_h.subject_to(opti_h.bounded(robot_setup.lbx, X[-1], robot_setup.ubx))
    for _ in range(N):
        # Control Torque Variable
        U += [opti_h.variable(robot_setup.nu)]

    # FIXED path progression: w(t) = 1/N*dt means uniform time progression
        # - s_k+1 = s_k + dt/T     where T = N*dt
        #         = s_k + 1/N  
    S = []
    S.append(0.0)
    for k in range(1, N + 1):
        if k>0:
            S.append(S[k-1] + 1/N)

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
    X, U = [], []

    for _ in range(N + 1):
        # Joint State Variable, with boundary ( both position and velocity )
        X += [opti_s.variable(robot_setup.nx)]    
        opti_s.subject_to(opti_s.bounded(robot_setup.lbx, X[-1], robot_setup.ubx))
    for _ in range(N):
        # Control Torque Variable
        U += [opti_s.variable(robot_setup.nu)]

    # initialize S vector for path progression
    S = []
    S.append(0.0)
    for k in range(1, N + 1):
        if k>0:
            S.append(S[k-1] + 1/N)

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




import casadi as cs

# TRAJECTORY TRACKING THROUGH HARD CONSTRAINT 
#               N-1
#   min          ∑ ( w_v*∥dq_k∥^2 + w_a∥u_k∥^2 )     +     w_final*∥x(N) - x_init∥^2
#   X,U         k=0
#              |--------- Running Cost ---------|        |----- Terminal Cost ------|
#
# subject to    - q_k+1 = q_k + ∆t*dq_k , dq_k+1 = dq_k + ∆t*u_k      ∀k ∈[0,N −1]
#               - q_min <= q_k <= q_max,  dq_max <= dq_k <= dq_max    ∀k ∈[1,N]
#               - tau_min <= τ_k <= tau_max                           ∀k ∈[1,N-1]
#               - s_k+1 = s_k + 1/N                                   ∀k ∈[1,N-1]
#               - y(q_k) = p(s_k)                                     ∀k ∈[1,N]
#               - x(0) = x_init   
#               - s(0) = 0  
#               - s(N) = 1 

def hard_cyclic_trajectory_tracking(opti, X, U, S, N, dt, x_init,
                                c_path, r_path, w_v, w_a, w_final, w_p,
                                nq, f, fk, inv_dyn, tau_min, tau_max):
    """
    Time-parameterized trajectory tracking with hard path constraints for cyclic motion.
    This formulation enforces that the end-effector strictly follows a prescribed spatial 
    path with a fixed time progression.
    
    Key differences from path tracking:
    - S variable is now determined by time: s(t) = t/T 
    - No optimization over W (path progression velocity)
    
    Args:
        opti: CasADi Opti instance
        X: State matrix (N+1, nx)
        U: Control matrix (N, nu)
        S: Path parameter (N+1, 1) - now fixed by time
        N: Number of time steps
        dt: Time step size
        x_init: Initial state
        c_path: Path center [c_x, c_y, c_z]
        r_path: Path radius
        w_v: Weight for velocity cost
        w_a: Weight for actuation cost
        w_final: Weight for cyclic terminal cost
        w_p: Weight for trajectory tracking cost (typically 10^2)
        nq: Number of joints
        f: State dynamics function
        fk: Forward kinematics function
        inv_dyn: Inverse dynamics function
        tau_min: Minimum torque limits
        tau_max: Maximum torque limits
    
    Returns:
        cost: Total cost (running + terminal)
    """
    
    # Initial and Terminal Constraints
    # - x(0) = x_init   
    # - s(0) = 0  
    # - s(N) = 1 
    opti.subject_to(X[0] == x_init)
    
    cost = 0.0
    
    for k in range(N):
        # Compute end-effector position
        q_k = X[k][0:nq]
        ee_pos = fk(q_k)
        
        # TRAJECTORY TRACKING: Hard Constraint
        # - y(q_k) = p(s_k)     Constrain ee_pos to lie on the desired path in x, y, z
        s_k = S[k]              # current path progress
        # derive end effectore position from the path progress
        p_x = c_path[0] + r_path * cs.cos(2 * cs.pi * s_k)          # p_x(s)=c_x ​+ rcos(2πs)
        p_y = c_path[1] + 0.5 * r_path * cs.sin(4 * cs.pi * s_k)    # p_y(s)=c_y+0.5rsine(4πs)
        p_z = c_path[2]                                             # p_z(s)=c_z
        p_s = cs.vertcat(p_x, p_y, p_z) 
        opti.subject_to(ee_pos == p_s)
        
        # Running costs
        # - w_v*∥dq_k∥^2 + w_a∥u_k∥^2
        cost += w_v * cs.sumsqr(X[k][nq:])  # velocity cost
        cost += w_a * cs.sumsqr(U[k])       # actuation cost
        # Note: No w_w term since W is not optimized
        

        # Discrete-Time Dynamics constraints
        # - q_k+1 = q_k + ∆t*dq_k 
        # - dq_k+1 = dq_k + ∆t*u_k 
        x_next = X[k] + dt * f(X[k], U[k])
        opti.subject_to(X[k+1] == x_next)


        # Torque constraints
        # - tau_min <= τ_k <= tau_max
        tau_k = inv_dyn(X[k], U[k])
        opti.subject_to(opti.bounded(tau_min, tau_k, tau_max))
    

    # TRAJECTORY TRACKING: Hard Constraint at terminal time
    # - y(q_N) = p(s_N)
    q_N = X[-1][0:nq]
    ee_pos_N = fk(q_N)
    s_N = S[-1]
    p_x_N = c_path[0] + r_path * cs.cos(2 * cs.pi * s_N)
    p_y_N = c_path[1] + 0.5 * r_path * cs.sin(4 * cs.pi * s_N)
    p_z_N = c_path[2]
    p_s_N = cs.vertcat(p_x_N, p_y_N, p_z_N)
    opti.subject_to(ee_pos_N == p_s_N)


    # Terminal cost for cyclic behavior
    # - w_final*∥x(N) - x_init∥^2
    cost += w_final * cs.sumsqr(X[-1] - x_init)
    
    return cost




# TRAJECTORY TRACKING THROUGH SOFT CONSTRAINT - COST FUNCTION
#               N-1
#   min          ∑ ( w_v*∥dq_k∥^2 + w_a∥u_k∥^2 + w_p*∥y_q - p_s∥^2 )     +     w_p*∥y(q_N) - p(s_N)∥^2  + w_final*∥x(N) - x_init∥^2
#   X,U         k=0
#               |---------------- Running Cost --------------------|         |------------------ Terminal Cost ------------------|
#
# subject to    - q_k+1 = q_k + ∆t*dq_k , dq_k+1 = dq_k + ∆t*u_k      ∀k ∈[0,N −1]
#               - q_min <= q_k <= q_max,  dq_max <= dq_k <= dq_max    ∀k ∈[1,N]
#               - tau_min <= τ_k <= tau_max                           ∀k ∈[1,N-1]
#               - s_k+1 = s_k + 1/N                                   ∀k ∈[1,N-1]
#               - x(0) = x_init   
#               - s(0) = 0  
#               - s(N) = 1 

def soft_cyclic_trajectory_tracking(opti, X, U, S, N, dt, x_init,
                                c_path, r_path, w_v, w_a, w_final, w_p,
                                nq, f, fk, inv_dyn, tau_min, tau_max):
    """
    Time-parameterized trajectory tracking with soft path constraints for cyclic motion.
    This formulation penalizes deviations from a prescribed spatial path rather than 
    enforcing strict adherence.
    
    Args:
        opti: CasADi Opti instance
        X: State matrix (N+1, nx)
        U: Control matrix (N, nu)
        S: Path parameter (N+1, 1) - now fixed by time
        N: Number of time steps
        dt: Time step size
        x_init: Initial state
        c_path: Path center [c_x, c_y, c_z]
        r_path: Path radius
        w_v: Weight for velocity cost
        w_a: Weight for actuation cost
        w_final: Weight for cyclic terminal cost
        w_p: Weight for trajectory tracking cost (typically 10^2)
        nq: Number of joints
        f: State dynamics function
        fk: Forward kinematics function
        inv_dyn: Inverse dynamics function
        tau_min: Minimum torque limits
        tau_max: Maximum torque limits
    
    Returns:
        cost: Total cost (running + terminal)
    """
    
    # Initial and Terminal Constraints
    # - x(0) = x_init   
    opti.subject_to(X[0] == x_init)
    
    cost = 0.0
    
    for k in range(N):
        # Compute end-effector position
        q_k = X[k][0:nq]
        ee_pos = fk(q_k)
        
        # TRAJECTORY TRACKING: Soft Constraint via cost term
        # Instead of enforcing ee_pos == p(s_k) as a hard constraint,
        # we penalize deviation from the desired trajectory
        s_k = S[k]
        p_x = c_path[0] + r_path * cs.cos(2 * cs.pi * s_k)
        p_y = c_path[1] + 0.5 * r_path * cs.sin(4 * cs.pi * s_k)
        p_z = c_path[2]
        p_s = cs.vertcat(p_x, p_y, p_z)
        
        # Trajectory tracking quadratic cost: penalize deviation from desired path
        # - w_p*∥y_q - p_s∥^2
        cost += w_p * cs.sumsqr(ee_pos - p_s)
        
        # Running costs
        # - w_v*∥dq_k∥^2 + w_a∥u_k∥^2
        cost += w_v * cs.sumsqr(X[k][nq:])  # velocity cost
        cost += w_a * cs.sumsqr(U[k])       # actuation cost
        # Note: No w_w term since W is not optimized
        

        # Discrete-Time Dynamics constraints
        # - q_k+1 = q_k + ∆t*dq_k 
        # - dq_k+1 = dq_k + ∆t*u_k 
        x_next = X[k] + dt * f(X[k], U[k])
        opti.subject_to(X[k+1] == x_next)
        

        # Torque constraints
        # - tau_min <= τ_k <= tau_max
        tau_k = inv_dyn(X[k], U[k])
        opti.subject_to(opti.bounded(tau_min, tau_k, tau_max))
    

    # TRAJECTORY TRACKING: Soft Constraint at terminal time
    # - w_p*∥y(q_N) - p(s_N)∥^2 
    q_N = X[-1][0:nq]
    ee_pos_N = fk(q_N)
    s_N = S[-1]
    p_x_N = c_path[0] + r_path * cs.cos(2 * cs.pi * s_N)
    p_y_N = c_path[1] + 0.5 * r_path * cs.sin(4 * cs.pi * s_N)
    p_z_N = c_path[2]
    p_s_N = cs.vertcat(p_x_N, p_y_N, p_z_N)
    cost += w_p * cs.sumsqr(ee_pos_N - p_s_N)


    # Terminal cost for cyclic behavior
    # - w_final*∥x(N) - x_init∥^2
    cost += w_final * cs.sumsqr(X[-1] - x_init)
    
    return cost