from planning.trajectory_generator.constant_speed_generator import *


class State:
    def __init__(self, x):
        self._x = x


class System:
    def __init__(self, state):
        self._state = state


# # local trajectory generation test
global_path = np.array([[0.0, 0.2], [0.5, 0.2], [0.5, 0.8], [1.0, 0.8]])

# single and repeated waypoints, enpoint test
sys_1 = System(State(np.array([1.0, 0.0])))
sys_2 = System(State(np.array([2.0, 0.0])))
traj_generator_1 = ConstantSpeedTrajectoryGenerator()
traj_generator_2 = ConstantSpeedTrajectoryGenerator()
path_1 = traj_generator_1.generate_trajectory(sys_1, np.array([[1.0, 0.0]]))
path_2 = traj_generator_2.generate_trajectory(sys_2, np.array([[2.1, 0.0], [2.1, 0.0]]))
print(path_1)
print(path_2)

# function iteration test
traj_generator_3 = ConstantSpeedTrajectoryGenerator()
path_3 = traj_generator_3.generate_trajectory(
    sys_1, np.array([[1.12, 0.0], [1.14, 0.015], [1.18, 0.015], [1.22, 0.015], [1.26, 0.015]])
)
print(path_3)

# general path
traj_generator_4 = ConstantSpeedTrajectoryGenerator()
path_4 = traj_generator_4.generate_trajectory(sys_1, global_path)
print(path_4)
