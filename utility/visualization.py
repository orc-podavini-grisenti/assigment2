"""
Visualization module for robot motion.
"""
import time
import numpy as np
from time import sleep

from orc.utils.robot_wrapper import RobotWrapper
from orc.utils.robot_simulator import RobotSimulator
from orc.utils.viz_utils import addViewerSphere, applyViewerConfiguration
import orc.optimal_control.casadi_adam.conf_ur5 as conf_ur5


class RobotMotionViewer:
    def __init__(self, robot_setup):
        """
        Initialize robot model, wrapper, and simulator.
        """
        self.robot_setup = robot_setup

        # Create robot wrapper
        self.r = RobotWrapper(
            robot_setup.robot.model,
            robot_setup.robot.collision_model,
            robot_setup.robot.visual_model
        )

        # Create simulator
        self.simu = RobotSimulator(conf_ur5, self.r)
        self.simu.init(robot_setup.q0, robot_setup.dq0)
        self.simu.display(robot_setup.q0)

        # Visualization parameters
        self.REF_SPHERE_RADIUS = 0.02
        self.EE_REF_SPHERE_COLOR = np.array([1, 0, 0, 0.5])

    def display(self, q_traj, ee_des_traj, N, dt):
        """
        Display the robot motion with end-effector reference spheres.
        """
        for i in range(N + 1):

            t0 = time.time()

            # Update robot configuration
            self.simu.display(q_traj[:, i])

            # Add reference sphere
            sphere_name = f'world/ee_ref_{i}'
            addViewerSphere(
                self.r.viz,
                sphere_name,
                self.REF_SPHERE_RADIUS,
                self.EE_REF_SPHERE_COLOR
            )
            applyViewerConfiguration(
                self.r.viz,
                sphere_name,
                ee_des_traj[:, i].tolist() + [0, 0, 0, 1.]
            )

            # Maintain real-time display
            elapsed = time.time() - t0
            if elapsed < dt:
                sleep(dt - elapsed)
