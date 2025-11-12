import casadi as cs

#               N-1
#   min          ∑   ( w_v*∥dq_k∥^2 + w_a∥u_k∥^2 + w_w∥w_k∥^2 )     +     w_final*∥x(N) - x_init∥^2
#  X,U,S,W      k=0
#               |---------------- Running Cost ---------------|        |------ Terminal Cost ------|
#
# subject to    - q_k+1 = q_k + ∆t*dq_k , dq_k+1 = dq_k + ∆t*u_k      ∀k ∈[0,N −1]
#               - q_min <= q_k <= q_max,  dq_max <= dq_k <= dq_max    ∀k ∈[1,N]
#               - tau_min <= τ_k <= tau_max                           ∀k ∈[1,N-1]
#               - s_k+1 = s_k + ∆t*w_k                                ∀k ∈[1,N-1]
#               - y(q_k) = p(s_k)                                     ∀k ∈[1,N]
#               - x(0) = x_init   
#               - s(0) = 0  
#               - s(N) = 1 


def running_cost_and_dynamics(opti, X, U, S, W, N, dt, x_init,
                                     c_path, r_path, w_v, w_a, w_w,
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
        dt (float): Time step size. 
        x_init (array-like): Initial state of the system, shape (nx,).
        c_path (array-like): Coordinates of the center of the path [c_x, c_y, c_z].
        r_path (float): Radius or scale factor of the path trajectory.
        w_v: Weight of the velocity in cost function
        w_a: Weight of the acceleration in cost function
        w_w: Weight of the weighting in cost function
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


# Question 1 Terminal Cost 
def terminal_cost_and_constraints(opti, X, S, c_path, r_path, nq, fk):
    """
    Adds terminal constraints to an optimal control problem.

    Args:
        opti: CasADi Opti instance for defining constraints and objectives.
        X: List or CasADi variable of states over the horizon.
        S: List or CasADi variable of path parameters over the horizon.
        c_path: Center of the desired path (3D coordinates).
        r_path: Radius or scaling factor for the path.
        nq: Number of joint positions in the state vector.
        fk: Forward kinematics function that maps joint positions to end-effector position.

    Returns:
        cost (casadi.MX/DM): Terminal cost
    """
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

    cost = 0 
    
    return cost


# Question 2 Terminal Cost 
def cyclic_terminal_cost_and_constraints(opti, X, S, x_init, c_path, r_path, w_final, nq, fk):
    """
    Adds cyclic terminal cost to an optimal control problem. The terminal cost penalize final robot 
    configuration x(N) different from the initial configuration x(0) = x_init

    Args:
        opti: CasADi Opti instance for defining constraints and objectives.
        X: List or CasADi variable of states over the horizon.
        S: List or CasADi variable of path parameters over the horizon.
        x_init: Initial state vector (used for terminal cost).
        c_path: Center of the desired path (3D coordinates).
        r_path: Radius or scaling factor for the path.
        w_final: Weight for the terminal cost.
        nq: Number of joint positions in the state vector.
        fk: Forward kinematics function that maps joint positions to end-effector position.

    Returns:
        cost (casadi.MX/DM): Terminal cost
    """
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

    cost = 0
    
    # Q2 Addition: 
    # terminal cost that penalizes the distance between the initial and final state
    # - w_final*∥x(N) - x_init∥^2
    cost += w_final * cs.sumsqr(X[-1] - x_init) 
    
    return cost