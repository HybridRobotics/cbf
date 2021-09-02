import math

from control.dcbf_controller import NmpcDcbfController
from models.dubin_car import *
from models.geometry_utils import *
from planning.path_generator.search_path_generator import *
from planning.trajectory_generator.constant_speed_generator import (
    ConstantSpeedTrajectoryGenerator,
)
from sim.simulation import SingleAgentSimulation


def dubin_car_system_test():
    sys = DubinCarSystem(
        state=DubinCarStates(x=np.array([0.0, 0.2, 0.0, 0.0])),
        geometry=DubinCarGeometry(0.08, 0.14),
        dynamics=DubinCarDynamics(),
    )
    sys.update(np.array([0.0, 0.0]))
    print(sys.get_state())


def dubin_car_planner_test():
    start_pos, goal_pos, grid, obstacles = create_env("s_path")
    sys = DubinCarSystem(
        state=DubinCarStates(x=np.block([start_pos[:2], np.array([0.0, start_pos[2]])])),
        geometry=DubinCarGeometry(0.08, 0.14),
        dynamics=DubinCarDynamics(),
    )
    # Reduce margin (radius) for tighter corners
    global_path_margin = 0.05
    global_planner = AstarPathGenerator(grid, quad=False, margin=global_path_margin)
    global_path = global_planner.generate_path(sys, obstacles, goal_pos)
    local_planner = ConstantSpeedTrajectoryGenerator()
    local_trajectory = local_planner.generate_trajectory(sys, global_path)
    print(local_trajectory)


def dubin_car_controller_test():
    start_pos, goal_pos, grid, obstacles = create_env("s_path")
    sys = DubinCarSystem(
        state=DubinCarStates(x=np.block([start_pos[:2], np.array([0.0, start_pos[2]])])),
        geometry=DubinCarGeometry(0.08, 0.14),
        dynamics=DubinCarDynamics(),
    )
    # Reduce margin (radius) for tighter corners
    global_path_margin = 0.05
    global_planner = AstarPathGenerator(grid, quad=False, margin=global_path_margin)
    global_path = global_planner.generate_path(sys, obstacles, goal_pos)
    local_planner = ConstantSpeedTrajectoryGenerator()
    local_trajectory = local_planner.generate_trajectory(sys, global_path)
    controller = NmpcDcbfController(dynamics=DubinCarDynamics())
    action = controller.generate_control_input(sys, global_path, local_trajectory, obstacles)


def dubin_car_simulation_test():
    start_pos, goal_pos, grid, obstacles = create_env("maze")
    robot = Robot(
        DubinCarSystem(
            state=DubinCarStates(x=np.block([start_pos[:2], np.array([0.0, start_pos[2]])])),
            geometry=DubinCarGeometry(0.14, 0.08),
            dynamics=DubinCarDynamics(),
        )
    )
    # Reduce margin (radius) for tighter corners
    global_path_margin = 0.07
    robot.set_global_planner(ThetaStarPathGenerator(grid, quad=False, margin=global_path_margin))
    robot.set_local_planner(ConstantSpeedTrajectoryGenerator())
    robot.set_controller(NmpcDcbfController(dynamics=DubinCarDynamics()))
    sim = SingleAgentSimulation(robot, obstacles, goal_pos)
    sim.run_navigation(40.0)
    sim.plot_world()
    sim.animate_world()


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
        s = 0.2  # scale of environment
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
        obstacles.append(RectangleRegion(-1.0 * s, 0.0 * s, 0.0 * s, 6.0 * s))
        obstacles.append(RectangleRegion(0.0 * s, 13.0 * s, -1.0 * s, 0.0 * s))
        obstacles.append(RectangleRegion(13.0 * s, 14.0 * s, 0.0 * s, 6.0 * s))
        return start, goal, grid, obstacles


if __name__ == "__main__":
    dubin_car_system_test()
    dubin_car_planner_test()
    dubin_car_controller_test()
    dubin_car_simulation_test()
