import math

import casadi as ca
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from models.geometry_utils import RectangleRegion
from sim.logger import (
    ControllerLogger,
    GlobalPlannerLogger,
    LocalPlannerLogger,
    SystemLogger,
)


class System:
    def __init__(self, time=0.0, state=None, geometry=None, dynamics=None):
        self._time = time
        self._state = state
        self._geometry = geometry
        self._dynamics = dynamics


class Robot:
    def __init__(self, system):
        self._system = system
        self._system_logger = SystemLogger()

    def set_global_planner(self, global_planner):
        self._global_planner = global_planner
        self._global_planner_logger = GlobalPlannerLogger()

    def set_local_planner(self, local_planner):
        self._local_planner = local_planner
        self._local_planner_logger = LocalPlannerLogger()

    def set_controller(self, controller):
        self._controller = controller
        self._controller_logger = ControllerLogger()

    def run_global_planner(self, sys, obstacles, goal_pos):
        # TODO: global path shall be generated with `system` and `obstacles`.
        self._global_path = self._global_planner.generate_path(sys, obstacles, goal_pos)
        self._global_planner.logging(self._global_planner_logger)

    def run_local_planner(self):
        # TODO: local path shall be generated with `obstacles`.
        self._local_trajectory = self._local_planner.generate_trajectory(self._system, self._global_path)
        self._local_planner.logging(self._local_planner_logger)

    def run_controller(self, obstacles):
        self._control_action = self._controller.generate_control_input(
            self._system, self._global_path, self._local_trajectory, obstacles
        )
        self._controller.logging(self._controller_logger)

    def run_system(self):
        self._system.update(self._control_action)
        self._system.logging(self._system_logger)


class SingleAgentSimulation:
    def __init__(self, robot, obstacles, goal_position):
        self._robot = robot
        self._obstacles = obstacles
        self._goal_position = goal_position

    def run_navigation(self, navigation_time):
        self._robot.run_global_planner(self._robot._system, self._obstacles, self._goal_position)
        while self._robot._system._time < navigation_time:
            self._robot.run_local_planner()
            self._robot.run_controller(self._obstacles)
            self._robot.run_system()

    def plot_world(self):
        # TODO: make this plotting function general applicable to different systems
        fig, ax = plt.subplots()
        plt.axis("equal")
        global_paths = self._robot._global_planner_logger._paths
        global_path = global_paths[0]
        ax.plot(global_path[:, 0], global_path[:, 1], "bo--", linewidth=1, markersize=4)
        closedloop_traj = np.vstack(self._robot._system_logger._xs)
        ax.plot(closedloop_traj[:, 0], closedloop_traj[:, 1], "k-", linewidth=2, markersize=4)
        for obs in self._obstacles:
            obs_patch = obs.get_plot_patch()
            ax.add_patch(obs_patch)
        plt.savefig("figures/world.eps", format="eps", dpi=1000, pad_inches=0)
        plt.savefig("figures/world.png", format="png", dpi=1000, pad_inches=0)

    def animate_world(self):
        # TODO: make this plotting function general applicable to different systems
        fig, ax = plt.subplots()
        plt.axis("equal")
        global_paths = self._robot._global_planner_logger._paths
        global_path = global_paths[0]
        ax.plot(global_path[:, 0], global_path[:, 1], "bo--", linewidth=1, markersize=4)

        local_paths = self._robot._local_planner_logger._trajs
        local_path = local_paths[0]
        (reference_traj_line,) = ax.plot(local_path[:, 0], local_path[:, 1])

        optimized_trajs = self._robot._controller_logger._xtrajs
        optimized_traj = optimized_trajs[0]
        (optimized_traj_line,) = ax.plot(optimized_traj[:, 0], optimized_traj[:, 1])

        closedloop_traj = np.vstack(self._robot._system_logger._xs)
        ax.plot(closedloop_traj[:, 0], closedloop_traj[:, 1], "k-", linewidth=2, markersize=4)
        for obs in self._obstacles:
            obs_patch = obs.get_plot_patch()
            ax.add_patch(obs_patch)

        robot_patch = patches.Polygon(np.zeros((1, 2)), alpha=1.0, closed=True, fc="None", ec="tab:brown")
        ax.add_patch(robot_patch)

        def update(index):
            local_path = local_paths[index]
            reference_traj_line.set_data(local_path[:, 0], local_path[:, 1])
            optimized_traj = optimized_trajs[index]
            optimized_traj_line.set_data(optimized_traj[:, 0], optimized_traj[:, 1])
            polygon_patch_next = self._robot._system._geometry.get_plot_patch(closedloop_traj[index, :])
            robot_patch.set_xy(polygon_patch_next.get_xy())

        anim = animation.FuncAnimation(fig, update, frames=len(closedloop_traj), interval=1000 * 0.1)
        anim.save("animation/world.mp4", dpi=300, writer=animation.writers["ffmpeg"](fps=60))
