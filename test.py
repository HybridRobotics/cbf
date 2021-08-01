import numpy as np

import utils
from controller import DualityController
from planner import Planner


def generate_planner():
    # Provide at least 2 distinct waypoints
    # globalpath = np.array([[0.0, 0.2], [0.1, 0.2]])
    globalpath = np.array([[0.0, 0.2], [0.5, 0.2], [0.5, 0.8], [1.0, 0.8]])
    reference_speed = 0.2
    num_horizon = 4
    localpath_timestep = 0.2
    return Planner(globalpath, reference_speed, num_horizon, localpath_timestep)


def generate_controller():
    sys_timestep = 0.05
    sim_timestep = 0.01
    ctrl_timestep = 0.2
    vehicle_length, vehicle_width = 0.08, 0.14
    num_horizon_opt, num_horizon_cbf = 4, 4
    gamma = 0.90
    dist_margin = 0.01
    return DualityController(
        sys_timestep,
        sim_timestep,
        ctrl_timestep,
        vehicle_length,
        vehicle_width,
        num_horizon_opt,
        num_horizon_cbf,
        gamma,
        dist_margin,
    )


def generate_obstacles():
    obstacles = []
    obstacles.append(utils.RectangleRegion(0.0, 1.0, 0.90, 1.0))
    obstacles.append(utils.RectangleRegion(0.0, 0.4, 0.3, 1.0))
    obstacles.append(utils.RectangleRegion(0.6, 1.0, 0.0, 0.7))
    return obstacles


def main():
    # Planner
    planner = generate_planner()
    # Controller
    controller = generate_controller()
    controller.set_state(np.array([0.0, 0.2, 0.0, 0.0]))
    controller.set_input(np.array([0.0, 0.0]))
    controller.set_planner(planner)
    # Add obstacles
    obstacles = generate_obstacles()
    controller.set_obstacles(obstacles)
    # Choose obstacle avoidance policy
    # controller.set_obstacle_avoidance_policy("point2region")
    controller.set_obstacle_avoidance_policy("region2region")
    # Setup simulation
    simulation_time = 5.0
    controller.sim(simulation_time)
    controller.plot_world()
    controller.plot_states()
    controller.animate_world()


if __name__ == "__main__":
    main()
