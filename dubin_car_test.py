from dubin_car import *
from planner import *
from controller import *
from utils import *


def dubin_car_system_test():
    sys = DubinCarSystem(
        state=DubinCarStates(x=np.array([0.0, 0.2, 0.0, 0.0])),
        geometry=DubinCarGeometry(0.08, 0.14),
        dynamics=DubinCarDynamics(),
    )
    sys.update(np.array([0.0, 0.0]))
    print(sys.get_state())


def dubin_car_planner_test():
    sys = DubinCarSystem(
        state=DubinCarStates(x=np.array([0.0, 0.2, 0.0, 0.0])),
        geometry=DubinCarGeometry(0.08, 0.14),
        dynamics=DubinCarDynamics(),
    )
    global_planner = GlobalPlanner()
    global_path = global_planner.generate_path()
    local_planner = PurePursuitPlanner()
    local_trajectory = local_planner.generate_trajectory(sys, global_path)
    print(local_trajectory)


def dubin_car_controller_test():
    sys = DubinCarSystem(
        state=DubinCarStates(x=np.array([0.0, 0.2, 0.0, 0.0])),
        geometry=DubinCarGeometry(0.08, 0.14),
        dynamics=DubinCarDynamics(),
    )
    global_planner = GlobalPlanner()
    global_path = global_planner.generate_path()
    local_planner = PurePursuitPlanner()
    local_trajectory = local_planner.generate_trajectory(sys, global_path)
    obstacles = []
    obstacles.append(RectangleRegion(0.0, 1.0, 0.90, 1.0))
    obstacles.append(RectangleRegion(0.0, 0.4, 0.4, 1.0))
    obstacles.append(RectangleRegion(0.6, 1.0, 0.0, 0.7))
    controller = NmpcDcbfController(dynamics=DubinCarDynamics())
    action = controller.generate_control_input(sys, global_path, local_trajectory, obstacles)


def dubin_car_simulation_test():
    robot = Robot(
        DubinCarSystem(
            state=DubinCarStates(x=np.array([0.0, 0.2, 0.0, 0.0])),
            geometry=DubinCarGeometry(0.08, 0.14),
            dynamics=DubinCarDynamics(),
        )
    )
    robot.set_global_planner(GlobalPlanner())
    robot.set_local_planner(PurePursuitPlanner())
    robot.set_controller(NmpcDcbfController(dynamics=DubinCarDynamics()))
    obstacles = []
    obstacles.append(RectangleRegion(0.0, 1.0, 0.90, 1.0))
    obstacles.append(RectangleRegion(0.0, 0.4, 0.4, 1.0))
    obstacles.append(RectangleRegion(0.6, 1.0, 0.0, 0.7))
    sim = SingleAgentSimulation(robot, obstacles)
    sim.run_navigation(10.0)
    sim.plot_world()
    sim.animate_world()


if __name__ == "__main__":
    dubin_car_system_test()
    dubin_car_planner_test()
    dubin_car_controller_test()
    dubin_car_simulation_test()
