"""
Robot setup and initialization module.
Contains all robot-specific configurations and dynamics functions.
"""
import numpy as np
import casadi as cs
from adam.casadi.computations import KinDynComputations
from example_robot_data.robots_loader import load


class RobotSetup:
    """Encapsulates robot configuration and dynamics."""
    
    def __init__(self, robot_name="ur5", frame_name="ee_link"):
        self.robot = load(robot_name)
        self.frame_name = frame_name
        
        # Extract robot parameters
        self.joints_name_list = [s for s in self.robot.model.names[1:]]
        self.nq = len(self.joints_name_list)
        self.nu = len(self.joints_name_list)
        self.nx = 2 * self.nq
        
        # Limits
        self.lbx = (self.robot.model.lowerPositionLimit.tolist() + 
                    (-self.robot.model.velocityLimit).tolist())
        self.ubx = (self.robot.model.upperPositionLimit.tolist() + 
                    self.robot.model.velocityLimit.tolist())
        self.tau_min = (-self.robot.model.effortLimit).tolist()
        self.tau_max = self.robot.model.effortLimit.tolist()
        
        # Initialize symbolic functions
        self._setup_symbolic_functions()
        
        # Initial configuration
        self.q0 = np.zeros(self.nq)
        self.dq0 = np.zeros(self.nq)
        self.x_init = np.concatenate([self.q0, self.dq0])
        
        # Path parameters
        self.r_path = 0.2
        y = self.fk(self.q0)
        self.c_path = np.array([y[0] - self.r_path, y[1], y[2]]).squeeze()
    
    def _setup_symbolic_functions(self):
        """Setup CasADi symbolic functions for dynamics and kinematics."""
        # Symbolic variables
        q = cs.SX.sym('q', self.nq)
        dq = cs.SX.sym('dq', self.nq)
        ddq = cs.SX.sym('ddq', self.nq)
        state = cs.vertcat(q, dq)
        rhs = cs.vertcat(dq, ddq)
        
        # State derivative function
        self.f = cs.Function('f', [state, ddq], [rhs])
        
        # Inverse dynamics
        kinDyn = KinDynComputations(self.robot.urdf, self.joints_name_list)
        H_b = cs.SX.eye(4)
        v_b = cs.SX.zeros(6)
        
        bias_forces = kinDyn.bias_force_fun()
        mass_matrix = kinDyn.mass_matrix_fun()
        
        h = bias_forces(H_b, q, v_b, dq)[6:]
        M = mass_matrix(H_b, q)[6:, 6:]
        tau = M @ ddq + h
        
        self.inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])
        
        # Forward kinematics
        fk_fun = kinDyn.forward_kinematics_fun(self.frame_name)
        ee_pos = fk_fun(H_b, q)[:3, 3]
        self.fk = cs.Function('fk', [q], [ee_pos])
    
    def get_config(self):
        """Return configuration dictionary."""
        return {
            'nq': self.nq,
            'nx': self.nx,
            'lbx': self.lbx,
            'ubx': self.ubx,
            'tau_min': self.tau_min,
            'tau_max': self.tau_max,
            'x_init': self.x_init,
            'c_path': self.c_path,
            'r_path': self.r_path,
            'q0': self.q0,
            'dq0': self.dq0,
            'joints_name_list': self.joints_name_list
        }