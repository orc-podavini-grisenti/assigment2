import numpy as np

def extract_solution(sol, X, U, S, W, N, nq, c_path, r_path, inv_dyn, fk):
    """
    Extract and process the solution from an optimal control problem for robotic trajectory tracking.
    
    This function takes the raw optimization solution and computes all relevant quantities including
    joint states, torques, end-effector positions, and desired trajectories.
    
    Args:
        sol: CasADi optimization solution object containing optimal values
        X: List of state decision variables for each time step (length N+1)
        U: List of control decision variables (accelerations) for each time step (length N)
        S: List of trajectory parameter decision variables for each time step (length N+1)
        W: List of auxiliary decision variables for each time step (length N)
        N (int): Number of time steps in the trajectory
        nq (int): Number of joints (degrees of freedom)
        c_path (array-like): Center point of the desired path [x, y, z] in meters
        r_path (float): Radius parameter for the desired path trajectory in meters
        inv_dyn (function): Inverse dynamics function 
        fk (function): Forward kinematics function 
    
    Returns:
        tuple: A tuple containing:
            - q_sol (np.ndarray): Joint positions, shape (nq, N+1) [rad]
            - dq_sol (np.ndarray): Joint velocities, shape (nq, N+1) [rad/s]
            - ddq_sol (np.ndarray): Joint accelerations, shape (nq, N) [rad/s²]
            - tau (np.ndarray): Joint torques, shape (nq, N) [Nm]
            - ee (np.ndarray): Actual end-effector positions, shape (3, N+1) [m]
            - ee_des (np.ndarray): Desired end-effector positions, shape (3, N+1) [m]
            - s_sol (np.ndarray): Trajectory parameter evolution, shape (N+1,) [0 to 1]
            - w_sol (np.ndarray): Auxiliary optimization variables, shape (N,)
    """
    # Extract state trajectory from optimization solution
    # State vector x contains [q; dq] where q is joint positions and dq is joint velocities
    x_sol = np.array([sol.value(X[k]) for k in range(N + 1)]).T
    q_sol = x_sol[:nq, :]
    dq_sol = x_sol[nq:, :]

    # Extract joint acceleration controls from optimization solution
    ddq_sol = np.array([sol.value(U[k]) for k in range(N)]).T

    # Extract trajectory parameter s (ranges from 0 to 1 along the path)
    s_sol = np.array([sol.value(S[k]) for k in range(N + 1)]).T

    # Extract auxiliary optimization variables
    w_sol = np.array([sol.value(W[k]) for k in range(N)]).T

    # Compute joint torques using inverse dynamics
    # For each time step, calculate the torques needed to achieve the desired accelerations
    tau = np.zeros((nq, N))
    for i in range(N):
        tau[:, i] = inv_dyn(x_sol[:, i], ddq_sol[:, i]).toarray().squeeze()

    # Compute actual end-effector positions using forward kinematics
    # For each time step, calculate where the end-effector actually is
    ee = np.zeros((3, N + 1))
    for i in range(N + 1):
        ee[:, i] = fk(x_sol[:nq, i]).toarray().squeeze()

    # Compute desired end-effector positions along the reference path
    # The path is a lemniscate-like curve: circular in x, figure-8 pattern in y
    ee_des = np.zeros((3, N + 1))
    for i in range(N + 1):
        ee_des[:, i] = np.array([c_path[0] + r_path*np.cos(2*np.pi*s_sol[i]),
                                 c_path[1] + r_path*0.5*np.sin(4*np.pi*s_sol[i]),
                                 c_path[2]])
    return q_sol, dq_sol, ddq_sol, tau, ee, ee_des, s_sol, w_sol




def save_tracking_summary(q_sol, tau, ee, ee_des, s_sol, dt, N, solver_timer, 
                            joint_names=None, tau_min=None, tau_max=None, 
                            filename='tracking_summary.csv'):
    """
    Save a comprehensive CSV summary of the tracking results.
    
    Args:
        q_sol (np.ndarray): Joint positions, shape (nq, N+1) [rad]
        tau (np.ndarray): Joint torques, shape (nq, N) [Nm]
        ee (np.ndarray): Actual end-effector positions, shape (3, N+1) [m]
        ee_des (np.ndarray): Desired end-effector positions, shape (3, N+1) [m]
        s_sol (np.ndarray): Trajectory parameter evolution, shape (N+1,)
        dt (float): Time step duration in seconds
        N (int): Number of time steps
        solver_timer (float): Time occured to the solver to solve the optimal problem
        joint_names (list, optional): List of joint names. If None, uses generic names.
        tau_min: Torque min limits [Nm]
        tau_max: Torque min limits [Nm]
        filename (str, optional): Output CSV filename. Default is 'trajectory_summary.csv'
    
    Returns:
        dict: Dictionary containing all computed metrics for further analysis
    """
    
    nq = q_sol.shape[0]
    
    # Set default joint names if not provided
    if joint_names is None:
        joint_names = [f'Joint_{i+1}' for i in range(nq)]
    
    # time information
    total_time = (N + 1) * dt
    
    # POSITION ERROR ANALYSIS
    ee_error = ee - ee_des  # shape (3, N+1)          # Position error at each time step
    ee_error_norm = np.linalg.norm(ee_error, axis=0)  # Euclidean distance error
    
    # Error statistics
    ee_error_mean = np.mean(ee_error_norm)
    ee_error_max = np.max(ee_error_norm)
    ee_error_std = np.std(ee_error_norm)
    ee_error_rms = np.sqrt(np.mean(ee_error_norm**2))
    ee_error_final = ee_error_norm[-1]
    
    
    # JOINT TORQUE ANALYSIS   
    # Check torque limits and identify violations
    joints_near_limits = []
    tau_limit_violations = []
    
    # Check for violations and near-limit conditions (within 10%)
    threshold = 0.9  # 90% of limit
    for i in range(nq):
        max_torque = np.max(tau[i, :])
        min_torque = np.min(tau[i, :])
        
        # Check if close to limits
        if max_torque > threshold * tau_max[i] or min_torque < threshold * tau_min[i]:
            utilization = max(abs(max_torque / tau_max[i]), 
                            abs(min_torque / tau_min[i])) * 100
            joints_near_limits.append(f"{joint_names[i]} ({utilization:.1f}%)")
        
        # Check for actual violations
        if max_torque > tau_max[i] or min_torque < tau_min[i]:
            tau_limit_violations.append(joint_names[i])
    
    
    # PATH PARAMETER ANALYSIS
    s_completion = s_sol[-1]  # Should be close to 1.0 for complete path


    # CYCLIC MOTION ANALYSIS (Initial vs Final Configuration)
    q_initial = q_sol[:, 0]   # Initial joint positions
    q_final = q_sol[:, -1]    # Final joint positions
    
    # Joint position errors
    q_error = q_final - q_initial  # Error per joint [rad]
    q_error_abs = np.abs(q_error)  # Absolute error per joint
    q_error_norm = np.linalg.norm(q_error)  # Euclidean norm of position error
    q_error_mean = np.mean(q_error_abs)  # Mean absolute error

    
    # WRITE TXT SUMMARY
    with open(filename, 'w') as txtfile:
        
        # Header information
        txtfile.write('=' * 80 + '\n')
        txtfile.write('PATH TRACKING SUMMARY\n')
        txtfile.write('=' * 80 + '\n\n')
        
        # General Information
        txtfile.write('GENERAL INFORMATION\n')
        txtfile.write('-' * 80 + '\n')
        txtfile.write(f'{"Metric":<40} {"Value":>15} {"Unit":>10}\n')
        txtfile.write('-' * 80 + '\n')
        txtfile.write(f'{"Total trajectory time":<40} {total_time:>15.4f} {"s":>10}\n')
        txtfile.write(f'{"Number of time steps":<40} {N:>15} {"-":>10}\n')
        txtfile.write(f'{"Time step size":<40} {dt:>15.6f} {"s":>10}\n')
        txtfile.write(f'{"Number of joints":<40} {nq:>15} {"-":>10}\n')
        txtfile.write(f'{"Path completion":<40} {s_completion:>15.4f} {"-":>10}\n')
        txtfile.write(f'{"Solver Time":<40} {solver_timer:>15.4f} {"s":>10}\n')
        txtfile.write('\n\n')
        
        # End-Effector Tracking Performance
        txtfile.write('END-EFFECTOR TRACKING PERFORMANCE\n')
        txtfile.write('-' * 80 + '\n')
        txtfile.write(f'{"Metric":<40} {"Value":>15} {"Unit":>10}\n')
        txtfile.write('-' * 80 + '\n')
        txtfile.write(f'{"Mean position error":<40} {ee_error_mean*1000:>15.4f} {"mm":>10}\n')
        txtfile.write(f'{"Max position error":<40} {ee_error_max*1000:>15.4f} {"mm":>10}\n')
        txtfile.write(f'{"RMS position error":<40} {ee_error_rms*1000:>15.4f} {"mm":>10}\n')
        txtfile.write(f'{"Std position error":<40} {ee_error_std*1000:>15.4f} {"mm":>10}\n')
        txtfile.write(f'{"Final position error":<40} {ee_error_final*1000:>15.4f} {"mm":>10}\n')
        txtfile.write('\n\n')

         # Cyclic Motion Analysis
        txtfile.write('CYCLIC MOTION ANALYSIS (Initial vs Final Configuration)\n')
        txtfile.write('-' * 80 + '\n')
        txtfile.write(f'{"Metric":<40} {"Value":>15} {"Unit":>10}\n')
        txtfile.write('-' * 80 + '\n')
        txtfile.write(f'{"Total joint error (norm)":<40} {np.rad2deg(q_error_norm):>15.4f} {"deg":>10}\n')
        txtfile.write(f'{"Mean absolute joint error":<40} {np.rad2deg(q_error_mean):>15.4f} {"deg":>10}\n')
        txtfile.write('\n')
        
        # Per-joint cyclic error details
        txtfile.write('Per-Joint Configuration Error:\n')
        txtfile.write(f'{"Joint Name":<20} {"Initial [deg]":>15} {"Final [deg]":>15} {"Error [deg]":>15}\n')
        txtfile.write('-' * 80 + '\n')
        for i in range(nq):
            txtfile.write(f'{joint_names[i]:<20} '
                         f'{np.rad2deg(q_initial[i]):>15.4f} '
                         f'{np.rad2deg(q_final[i]):>15.4f} '
                         f'{np.rad2deg(q_error[i]):>15.4f}\n')
        txtfile.write('\n')
        
        # Joint Limit Analysis
        txtfile.write('JOINT LIMIT ANALYSIS\n')
        txtfile.write('-' * 80 + '\n')
        txtfile.write(f'{"Category":<40} {"Joints":>39}\n')
        txtfile.write('-' * 80 + '\n')
        
        if joints_near_limits:
            joints_str = '; '.join(joints_near_limits)
            txtfile.write(f'{"Joints near torque limits (>90%)":<40} {joints_str:>39}\n')
        else:
            txtfile.write(f'{"Joints near torque limits (>90%)":<40} {"None":>39}\n')
        
        if tau_limit_violations:
            violations_str = '; '.join(tau_limit_violations)
            txtfile.write(f'{"Torque limit violations":<40} {violations_str:>39}\n')
        else:
            txtfile.write(f'{"Torque limit violations":<40} {"None":>39}\n')
        
        txtfile.write('\n')
        txtfile.write('=' * 80 + '\n')
    
    print(f"Trajectory summary saved to: {filename}")
    
    # Return dictionary with all metrics for programmatic access
    summary_dict = {
        'total_time': total_time,
        'ee_error_mean': ee_error_mean,
        'ee_error_max': ee_error_max,
        'ee_error_rms': ee_error_rms,
        'ee_error_final': ee_error_final,
        'joints_near_limits': joints_near_limits,
        'tau_limit_violations': tau_limit_violations,
        's_completion': s_completion,
    }
    
    return summary_dict
