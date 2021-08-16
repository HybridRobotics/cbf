import numpy as np


class GlobalPlanner:
    # TODO: this class shall be renamed as replace by a search-based planner
    def __init__(self):
        self._global_path = None

    def generate_path(self):
        # TODO: Remove hardcoded global path
        self._global_path = np.array([[0.0, 0.2], [0.5, 0.2], [0.5, 0.8], [1.0, 0.8]])
        return self._global_path

    def logging(self, logger):
        logger.append(self._global_path)


class PurePursuitPlanner:
    # TODO: Refactor this class to make it light-weight.
    # TODO: Create base class for this local planner
    def __init__(self):
        # TODO: wrap params
        self._global_path_index = 0
        # TODO: number of waypoints shall equal to length of global path
        self._num_waypoint = 4
        # local path
        self._reference_speed = 0.2
        self._num_horizon = 4
        self._local_path_timestep = 0.1
        self._local_trajectory = None

    def generate_trajectory(self, system, global_path):
        pos = system._state._x[0:2]
        return self.generate_trajectory_internal(pos, global_path)

    def generate_trajectory_internal(self, pos, global_path):
        proj_dist_buffer = 0.05
        local_index = self._global_path_index
        trunc_path = np.vstack([global_path[local_index:, :], global_path[-1, :]])
        curv_vec = trunc_path[1:, :] - trunc_path[:-1, :]
        curv_length = np.linalg.norm(curv_vec, axis=1)

        if curv_length[0] == 0.0:
            curv_direct = np.zeros((2,))
        else:
            curv_direct = curv_vec[0, :] / curv_length[0]
        proj_dist = np.dot(pos - trunc_path[0, :], curv_direct)

        if proj_dist >= curv_length[0] - proj_dist_buffer and local_index < self._num_waypoint - 1:
            self._global_path_index += 1
            return self.generate_trajectory_internal(pos, global_path)

        # t_c = (proj_dist + proj_dist_buffer) / self._reference_speed
        t_c = proj_dist / self._reference_speed
        t_s = t_c + self._local_path_timestep * np.linspace(0, self._num_horizon - 1, self._num_horizon)

        curv_time = np.cumsum(np.hstack([0.0, curv_length / self._reference_speed]))
        curv_time[-1] += (
            t_c + 2 * self._local_path_timestep * self._num_horizon + proj_dist_buffer / self._reference_speed
        )

        path_idx = np.searchsorted(curv_time, t_s, side="right") - 1
        path = np.vstack(
            [
                np.interp(t_s, curv_time, trunc_path[:, 0]),
                np.interp(t_s, curv_time, trunc_path[:, 1]),
            ]
        ).T
        path_vel = self._reference_speed * np.ones((self._num_horizon, 1))
        path_head = np.arctan2(curv_vec[path_idx, 1], curv_vec[path_idx, 0]).reshape(self._num_horizon, 1)
        self._local_trajectory = np.hstack([path, path_vel, path_head])
        return self._local_trajectory

    def logging(self, logger):
        logger.append(self._local_trajectory)
