from utils import RectangleRegion
from logger import SystemLogger, ControllerLogger
import math
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation


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
        self._global_planner_logger = []

    def set_local_planner(self, local_planner):
        self._local_planner = local_planner
        self._local_planner_logger = []

    def set_controller(self, controller):
        self._controller = controller
        self._controller_logger = ControllerLogger()

    def run_global_planner(self):
        # TODO: global path shall be generated with `system` and `obstacles`.
        self._global_path = self._global_planner.generate_path()
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
    def __init__(self, robot, obstacles):
        self._robot = robot
        self._obstacles = obstacles

    def run_navigation(self, navigation_time):
        self._robot.run_global_planner()
        while self._robot._system._time < navigation_time:
            self._robot.run_local_planner()
            self._robot.run_controller(self._obstacles)
            self._robot.run_system()

    def plot_world(self):
        # TODO: make this plotting function general applicable to different systems
        fig, ax = plt.subplots()
        plt.axis("equal")
        global_paths = self._robot._global_planner_logger
        global_path = global_paths[0]
        ax.plot(global_path[:, 0], global_path[:, 1], "bo--", linewidth=1, markersize=4)
        closedloop_traj = np.vstack(self._robot._system_logger._xs)
        ax.plot(closedloop_traj[:, 0], closedloop_traj[:, 1], "k-", linewidth=2, markersize=4)
        for obs in self._obstacles:
            rec_patch = obs.get_plot_patch()
            ax.add_patch(rec_patch)
        plt.savefig("figures/world.eps", format="eps", dpi=1000, pad_inches=0)
        plt.savefig("figures/world.png", format="png", dpi=1000, pad_inches=0)

    def animate_world(self):
        # TODO: make this plotting function general applicable to different systems
        fig, ax = plt.subplots()
        plt.axis("equal")
        global_paths = self._robot._global_planner_logger
        global_path = global_paths[0]
        ax.plot(global_path[:, 0], global_path[:, 1], "bo--", linewidth=1, markersize=4)

        local_paths = self._robot._local_planner_logger
        local_path = local_paths[0]
        (reference_traj_line,) = ax.plot(local_path[:, 0], local_path[:, 0])

        controller_logger = self._robot._controller_logger
        optimized_traj = controller_logger._xtrajs[0]
        (optimized_traj_line,) = ax.plot(optimized_traj[:, 0], optimized_traj[:, 1])

        closedloop_traj = np.vstack(self._robot._system_logger._xs)
        ax.plot(closedloop_traj[:, 0], closedloop_traj[:, 1], "k-", linewidth=2, markersize=4)
        for obs in self._obstacles:
            rec_patch = obs.get_plot_patch()
            ax.add_patch(rec_patch)

        polygon_points = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])
        vehicle_polygon = patches.Polygon(polygon_points, alpha=1.0, closed=True, fc="None", ec="tab:brown")
        ax.add_patch(vehicle_polygon)

        def update(index):
            local_path = local_paths[index]
            reference_traj_line.set_data(local_path[:, 0], local_path[:, 1])
            optimized_traj = controller_logger._xtrajs[index]
            optimized_traj_line.set_data(optimized_traj[:, 0], optimized_traj[:, 1])
            # TODO: wrap this updated function
            length, width = self._robot._system._geometry._length, self._robot._system._geometry._width
            x, y, theta = (
                closedloop_traj[index, 0],
                closedloop_traj[index, 1],
                closedloop_traj[index, 3],
            )
            vehicle_points = np.array(
                [
                    [
                        x + length / 2 * np.cos(theta) - width / 2 * np.sin(theta),
                        y + length / 2 * np.sin(theta) + width / 2 * np.cos(theta),
                    ],
                    [
                        x + length / 2 * np.cos(theta) + width / 2 * np.sin(theta),
                        y + length / 2 * np.sin(theta) - width / 2 * np.cos(theta),
                    ],
                    [
                        x - length / 2 * np.cos(theta) + width / 2 * np.sin(theta),
                        y - length / 2 * np.sin(theta) - width / 2 * np.cos(theta),
                    ],
                    [
                        x - length / 2 * np.cos(theta) - width / 2 * np.sin(theta),
                        y - length / 2 * np.sin(theta) + width / 2 * np.cos(theta),
                    ],
                ]
            )
            vehicle_polygon.set_xy(vehicle_points)

        anim = animation.FuncAnimation(fig, update, frames=len(closedloop_traj), interval=100)
        anim.save("animation/world.gif", dpi=200, writer="imagemagick")
