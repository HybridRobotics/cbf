import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
import statistics as st

from control.dcbf_optimizer import NmpcDcbfOptimizerParam
from control.dcbf_controller import NmpcDcbfController
from models.dubin_car import (
    DubinCarDynamics,
    DubinCarGeometry,
    DubinCarStates,
    DubinCarSystem,
)
from models.geometry_utils import *
from models.kinematic_car import (
    KinematicCarDynamics,
    KinematicCarRectangleGeometry,
    KinematicCarMultipleGeometry,
    KinematicCarTriangleGeometry,
    KinematicCarPentagonGeometry,
    KinematicCarStates,
    KinematicCarSystem,
)
from planning.path_generator.search_path_generator import (
    AstarLoSPathGenerator,
    AstarPathGenerator,
    ThetaStarPathGenerator,
)
from planning.trajectory_generator.constant_speed_generator import (
    ConstantSpeedTrajectoryGenerator,
)
from sim.simulation import Robot, SingleAgentSimulation


def plot_world(simulation, snapshot_indexes, figure_name="world", local_traj_indexes=[], maze_type=None):
    # TODO: make this plotting function general applicable to different systems
    if maze_type == "maze":
        fig, ax = plt.subplots(figsize=(8.3, 5.0))
    elif maze_type == "oblique_maze":
        fig, ax = plt.subplots(figsize=(6.7, 5.0))
    # degrees_rot = 90
    # transform = mpl.transforms.Affine2D().rotate_deg(degrees_rot) + ax.transData
    # extract data
    global_paths = simulation._robot._global_planner_logger._paths
    global_path = global_paths[0]
    closedloop_traj = np.vstack(simulation._robot._system_logger._xs)
    local_paths = simulation._robot._local_planner_logger._trajs
    optimized_trajs = simulation._robot._controller_logger._xtrajs
    # plot robot
    for index in snapshot_indexes:
        for i in range(simulation._robot._system._geometry._num_geometry):
            polygon_patch = simulation._robot._system._geometry.get_plot_patch(closedloop_traj[index, :], i, 0.25)
            # polygon_patch.set_transform(transform)
            ax.add_patch(polygon_patch)
    # plot global reference
    ax.plot(global_path[:, 0], global_path[:, 1], "o--", color="grey", linewidth=1.5, markersize=2)
    # plot closed loop trajectory
    # ax.plot(closedloop_traj[:, 0], closedloop_traj[:, 1], "k-", linewidth=1, markersize=4, transform=transform)
    # plot obstacles
    for obs in simulation._obstacles:
        obs_patch = obs.get_plot_patch()
        # obs_patch.set_transform(transform)
        ax.add_patch(obs_patch)
    # plot local reference and local optimized trajectories
    for index in local_traj_indexes:
        local_path = local_paths[index]
        ax.plot(local_path[:, 0], local_path[:, 1], "-", color="blue", linewidth=3, markersize=4)
        optimized_traj = optimized_trajs[index]
        ax.plot(
            optimized_traj[:, 0],
            optimized_traj[:, 1],
            "-",
            color="gold",
            linewidth=3,
            markersize=4,
        )
    # set figure properties
    plt.tight_layout()
    # save figure
    plt.savefig("figures/" + figure_name + ".eps", format="eps", dpi=500, pad_inches=0)
    plt.savefig("figures/" + figure_name + ".png", format="png", dpi=500, pad_inches=0)


def animate_world(simulation, animation_name="world", maze_type=None):
    # TODO: make this plotting function general applicable to different systems
    if maze_type == "maze":
        fig, ax = plt.subplots(figsize=(8.3, 5.0))
    elif maze_type == "oblique_maze":
        fig, ax = plt.subplots(figsize=(6.7, 5.0))
    global_paths = simulation._robot._global_planner_logger._paths
    global_path = global_paths[0]
    ax.plot(global_path[:, 0], global_path[:, 1], "bo--", linewidth=1.5, markersize=4)

    local_paths = simulation._robot._local_planner_logger._trajs
    local_path = local_paths[0]
    (reference_traj_line,) = ax.plot(local_path[:, 0], local_path[:, 1], "-", color="blue", linewidth=3, markersize=4)

    optimized_trajs = simulation._robot._controller_logger._xtrajs
    optimized_traj = optimized_trajs[0]
    (optimized_traj_line,) = ax.plot(
        optimized_traj[:, 0],
        optimized_traj[:, 1],
        "-",
        color="gold",
        linewidth=3,
        markersize=4,
    )

    closedloop_traj = np.vstack(simulation._robot._system_logger._xs)
    for obs in simulation._obstacles:
        obs_patch = obs.get_plot_patch()
        ax.add_patch(obs_patch)

    robot_patch = []
    for i in range(simulation._robot._system._geometry._num_geometry):
        robot_patch.append(patches.Polygon(np.zeros((1, 2)), alpha=1.0, closed=True, fc="None", ec="tab:brown"))
        ax.add_patch(robot_patch[i])
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
    plt.tight_layout()

    def update(index):
        local_path = local_paths[index]
        reference_traj_line.set_data(local_path[:, 0], local_path[:, 1])
        optimized_traj = optimized_trajs[index]
        optimized_traj_line.set_data(optimized_traj[:, 0], optimized_traj[:, 1])
        # plt.xlabel(str(index))
        for i in range(simulation._robot._system._geometry._num_geometry):
            polygon_patch_next = simulation._robot._system._geometry.get_plot_patch(closedloop_traj[index, :], i)
            robot_patch[i].set_xy(polygon_patch_next.get_xy())
        if index == len(closedloop_traj) - 1:
            ax.plot(closedloop_traj[:, 0], closedloop_traj[:, 1], "k-", linewidth=3, markersize=4)

    anim = animation.FuncAnimation(fig, update, frames=len(closedloop_traj), interval=1000 * 0.1)
    anim.save("animations/" + animation_name + ".mp4", dpi=300, writer=animation.writers["ffmpeg"](fps=10))


def kinematic_car_triangle_simulation_test():
    start_pos, goal_pos, grid, obstacles = create_env("maze")
    robot = Robot(
        KinematicCarSystem(
            state=KinematicCarStates(x=np.block([start_pos[:2], np.array([0.0, start_pos[2]])])),
            geometry=KinematicCarTriangleGeometry(np.array([[0.14, 0.00], [-0.03, 0.05], [-0.03, -0.05]])),
            dynamics=KinematicCarDynamics(),
        )
    )
    global_path_margin = 0.07
    robot.set_global_planner(ThetaStarPathGenerator(grid, quad=False, margin=global_path_margin))
    robot.set_local_planner(ConstantSpeedTrajectoryGenerator())
    robot.set_controller(NmpcDcbfController(dynamics=KinematicCarDynamics(), opt_param=NmpcDcbfOptimizerParam))
    sim = SingleAgentSimulation(robot, obstacles, goal_pos)
    sim.run_navigation(20.0)
    plot_world(sim, np.arange(0, 200, 5), figure_name="triangle")
    animate_world(sim, animation_name="triangle")


def kinematic_car_pentagon_simulation_test():
    start_pos, goal_pos, grid, obstacles = create_env("maze")
    robot = Robot(
        KinematicCarSystem(
            state=KinematicCarStates(x=np.block([start_pos[:2], np.array([0.0, start_pos[2]])])),
            geometry=KinematicCarPentagonGeometry(
                np.array([[0.15, 0.00], [0.03, 0.05], [-0.01, 0.02], [-0.01, -0.02], [0.03, -0.05]])
            ),
            dynamics=KinematicCarDynamics(),
        )
    )
    global_path_margin = 0.06
    robot.set_global_planner(ThetaStarPathGenerator(grid, quad=False, margin=global_path_margin))
    robot.set_local_planner(ConstantSpeedTrajectoryGenerator())
    robot.set_controller(NmpcDcbfController(dynamics=KinematicCarDynamics(), opt_param=NmpcDcbfOptimizerParam()))
    sim = SingleAgentSimulation(robot, obstacles, goal_pos)
    sim.run_navigation(20.0)
    plot_world(sim, np.arange(0, 200, 5), figure_name="pentagon")
    animate_world(sim, animation_name="pentagon")


def kinematic_car_all_shapes_simulation_test(maze_type, robot_shape):
    start_pos, goal_pos, grid, obstacles = create_env(maze_type)
    geometry_regions = KinematicCarMultipleGeometry()

    if robot_shape == "rectangle":
        geometry_regions.add_geometry(KinematicCarRectangleGeometry(0.15, 0.06, 0.1))
        if maze_type == "maze":
            robot_indexes = [0, 17, 24, 33, 39, 48, 58, 66, 76, 86, 92, 102, 112, 129]
            traj_indexes = [0, 24, 33, 48, 66, 92, 112]
        elif maze_type == "oblique_maze":
            robot_indexes = [1, 9, 22, 26, 34, 41, 47, 56, 64, 70, 79, 88, 93, 124, 131, 143, 164]
            traj_indexes = [1, 19, 47, 70, 93]
    if robot_shape == "pentagon":
        geometry_regions.add_geometry(
            KinematicCarTriangleGeometry(
                np.array([[0.15, 0.00], [0.03, 0.05], [-0.01, 0.02], [-0.01, -0.02], [0.03, -0.05]])
            )
        )
        if maze_type == "maze":
            robot_indexes = [2, 15, 33, 43, 50, 61, 69, 81, 99, 111, 125, 130, 142, 164]
            traj_indexes = [2, 33, 50, 69, 99, 125, 164]
        elif maze_type == "oblique_maze":
            robot_indexes = [0, 11, 23, 41, 53, 62, 69, 78, 83, 92, 110, 113, 138, 151, 164, 185]
            traj_indexes = [0, 23, 62, 78, 92, 110, 151]
    if robot_shape == "triangle":
        geometry_regions.add_geometry(
            KinematicCarTriangleGeometry(0.75 * np.array([[0.14, 0.00], [-0.03, 0.05], [-0.03, -0.05]]))
        )
        if maze_type == "maze":
            robot_indexes = [0, 10, 18, 27, 35, 40, 46, 51, 61, 72, 86, 91, 102, 113, 199]
            traj_indexes = [0, 18, 35, 46, 61, 86, 102]
        elif maze_type == "oblique_maze":
            robot_indexes = [0, 11, 17, 26, 32, 43, 50, 58, 74, 78, 110, 123, 168, 182, 199]
            traj_indexes = [0, 17, 32, 50, 74, 168]
    if robot_shape == "lshape":
        geometry_regions.add_geometry(
            KinematicCarTriangleGeometry(0.4 * np.array([[0, 0.1], [0.02, 0.08], [-0.2, -0.1], [-0.22, -0.08]]))
        )
        geometry_regions.add_geometry(
            KinematicCarTriangleGeometry(0.4 * np.array([[0, 0.1], [-0.02, 0.08], [0.2, -0.1], [0.22, -0.08]]))
        )
        if maze_type == "maze":
            robot_indexes = [3, 13, 27, 36, 46, 56, 65, 74, 85, 98, 114, 121, 224, 299]
            traj_indexes = [3, 27, 46, 74, 114, 224]
        elif maze_type == "oblique_maze":
            robot_indexes = [0, 12, 21, 32, 40, 49, 59, 67, 81, 94, 122, 129, 141, 177]
            traj_indexes = [0, 21, 40, 59, 81, 129]
    robot = Robot(
        KinematicCarSystem(
            state=KinematicCarStates(x=np.block([start_pos[:2], np.array([0.0, start_pos[2]])])),
            geometry=geometry_regions,
            dynamics=KinematicCarDynamics(),
        )
    )
    global_path_margin = 0.05
    robot.set_global_planner(AstarLoSPathGenerator(grid, quad=False, margin=global_path_margin))
    robot.set_local_planner(ConstantSpeedTrajectoryGenerator())
    opt_param = NmpcDcbfOptimizerParam()
    # TODO: Wrap these parameters
    if robot_shape == "rectangle" or robot_shape == "lshape":
        opt_param.terminal_weight = 10.0
    elif robot_shape == "pentagon":
        if maze_type == 1:
            opt_param.terminal_weight = 2.0
        elif maze_type == 2:
            opt_param.terminal_weight = 5.0
    elif robot_shape == "triangle":
        opt_param.terminal_weight = 2.0
    robot.set_controller(NmpcDcbfController(dynamics=KinematicCarDynamics(), opt_param=opt_param))
    sim = SingleAgentSimulation(robot, obstacles, goal_pos)
    sim.run_navigation(30.0)
    print("median: ", st.median(robot._controller._optimizer.solver_times))
    print("std: ", st.stdev(robot._controller._optimizer.solver_times))
    print("min: ", min(robot._controller._optimizer.solver_times))
    print("max: ", max(robot._controller._optimizer.solver_times))
    print("Simulation finished.")
    name = robot_shape + "_" + maze_type
    plot_world(sim, robot_indexes, figure_name=name, local_traj_indexes=traj_indexes, maze_type=maze_type)
    animate_world(sim, animation_name=name, maze_type=maze_type)


def create_env(env_type):
    if env_type == "s_path":
        s = 1.0  # scale of environment
        start = np.array([0.0 * s, 0.2 * s, 0.0])
        goal = np.array([1.0 * s, 0.8 * s])
        bounds = ((-0.2 * s, 0.0 * s), (1.2 * s, 1.2 * s))
        cell_size = 0.05 * s
        grid = (bounds, cell_size)
        obstacles = []
        obstacles.append(RectangleRegion(0.0 * s, 1.0 * s, 0.9 * s, 1.0 * s))
        obstacles.append(RectangleRegion(0.0 * s, 0.4 * s, 0.4 * s, 1.0 * s))
        obstacles.append(RectangleRegion(0.6 * s, 1.0 * s, 0.0 * s, 0.7 * s))
        return start, goal, grid, obstacles
    elif env_type == "maze":
        s = 0.15  # scale of environment
        start = np.array([0.5 * s, 5.5 * s, -math.pi / 2.0])
        goal = np.array([12.5 * s, 0.5 * s])
        bounds = ((0.0 * s, 0.0 * s), (13.0 * s, 6.0 * s))
        cell_size = 0.25 * s
        grid = (bounds, cell_size)
        obstacles = []
        obstacles.append(RectangleRegion(0.0 * s, 3.0 * s, 0.0 * s, 3.0 * s))
        obstacles.append(RectangleRegion(1.0 * s, 2.0 * s, 4.0 * s, 6.0 * s))
        obstacles.append(RectangleRegion(2.0 * s, 6.0 * s, 5.0 * s, 6.0 * s))
        obstacles.append(RectangleRegion(6.0 * s, 7.0 * s, 4.0 * s, 6.0 * s))
        obstacles.append(RectangleRegion(4.0 * s, 5.0 * s, 0.0 * s, 4.0 * s))
        obstacles.append(RectangleRegion(5.0 * s, 7.0 * s, 2.0 * s, 3.0 * s))
        obstacles.append(RectangleRegion(6.0 * s, 9.0 * s, 1.0 * s, 2.0 * s))
        obstacles.append(RectangleRegion(8.0 * s, 9.0 * s, 2.0 * s, 4.0 * s))
        obstacles.append(RectangleRegion(9.0 * s, 12.0 * s, 3.0 * s, 4.0 * s))
        obstacles.append(RectangleRegion(11.0 * s, 12.0 * s, 4.0 * s, 5.0 * s))
        obstacles.append(RectangleRegion(8.0 * s, 10.0 * s, 5.0 * s, 6.0 * s))
        obstacles.append(RectangleRegion(10.0 * s, 11.0 * s, 0.0 * s, 2.0 * s))
        obstacles.append(RectangleRegion(12.0 * s, 13.0 * s, 1.0 * s, 2.0 * s))
        obstacles.append(RectangleRegion(0.0 * s, 13.0 * s, 6.0 * s, 7.0 * s))
        obstacles.append(RectangleRegion(-1.0 * s, 0.0 * s, -1.0 * s, 7.0 * s))
        obstacles.append(RectangleRegion(0.0 * s, 13.0 * s, -1.0 * s, 0.0 * s))
        obstacles.append(RectangleRegion(13.0 * s, 14.0 * s, -1.0 * s, 7.0 * s))
        return start, goal, grid, obstacles
    elif env_type == "oblique_maze":
        s = 0.15  # scale of environemtn
        start = np.array([1.0 * s, 1.5 * s, 0.0])
        goal = np.array([8.5 * s, 6.5 * s])
        bounds = ((0.0 * s, 1.0 * s), (10.0 * s, 7.0 * s))
        cell_size = 0.2 * s
        grid = (bounds, cell_size)
        obstacles = []
        # TODO: Overload the constructor of RectangleRegion() 
        obstacles.append(RectangleRegion(-1.0 * s, 0.0 * s, 0.0 * s, 8.0 * s))
        obstacles.append(RectangleRegion(0.0 * s, 10.0 * s, 0.0 * s, 1.0 * s))
        obstacles.append(RectangleRegion(0.0 * s, 8.0 * s, 7.0 * s, 8.0 * s))
        obstacles.append(RectangleRegion(10.0 * s, 11.0 * s, 0.0 * s, 8.0 * s))
        obstacles.append(
            PolytopeRegion.convex_hull(s * np.array([[0.0, 2.0], [1.25, 3.875], [2.875, 3.125], [2.5, 2.25]]))
        )
        obstacles.append(PolytopeRegion.convex_hull(s * np.array([[1, 4.75], [0.0, 5.0], [0.875, 7], [1.875, 6.375]])))
        obstacles.append(
            PolytopeRegion.convex_hull(s * np.array([[2.75, 1], [4.2, 3.25], [5.125, 3.75], [6.625, 2.5], [6.5, 1.0]]))
        )
        obstacles.append(PolytopeRegion.convex_hull(s * np.array([[6.0, 7.0], [6, 6], [6.5, 7.0]])))
        obstacles.append(
            PolytopeRegion.convex_hull(
                s * np.array([[2.375, 4.875], [2.875, 5.875], [4.5, 5.875], [4.75, 4], [3.375, 4]])
            )
        )
        obstacles.append(PolytopeRegion.convex_hull(s * np.array([[6.75, 1.0], [7.25, 2.375], [8.5, 2.0], [8.5, 1.0]])))
        obstacles.append(PolytopeRegion.convex_hull(s * np.array([[8.625, 1.0], [10.0, 2.5], [10.0, 1.0]])))
        obstacles.append(PolytopeRegion.convex_hull(s * np.array([[10.0, 2.875], [9.5, 5.75], [10.0, 5.875]])))
        obstacles.append(
            PolytopeRegion.convex_hull(
                s * np.array([[8.875, 3.125], [8.0, 5.5], [6.875, 6.375], [5.875, 5.875], [6.25, 4.375], [7.125, 3.5]])
            )
        )
        return start, goal, grid, obstacles


if __name__ == "__main__":
    # kinematic_car_triangle_simulation_test()
    # kinematic_car_pentagon_simulation_test()
    maze_types = ["maze", "oblique_maze"]
    robot_shapes = ["triangle", "rectangle", "pentagon", "lshape"]
    for maze_type in maze_types:
        for robot_shape in robot_shapes:
            kinematic_car_all_shapes_simulation_test(maze_type, robot_shape)
