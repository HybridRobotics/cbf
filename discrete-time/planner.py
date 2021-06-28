import numpy as np
import sympy as sp
from scipy.interpolate import interp1d
import math


class Planner:
    def __init__(self, globalpath, reference_speed, num_horizon, localpath_timestep):
        # global path
        self._global_path = globalpath
        self._global_path_index = 0
        self._num_waypoint = np.shape(globalpath)[0]
        # local path
        self._reference_speed = reference_speed
        self._num_horizon = num_horizon
        self._local_path_timestep = localpath_timestep

    def global_path(self):
        """Return global path
        """
        return self._global_path

    def local_path(self, pos):
        """[Generate smooth trajectory with sampling waypoints to track]
        """
        proj_dist_buffer = 0.05

        local_index = self._global_path_index
        trunc_path = np.vstack([self._global_path[local_index:, :], self._global_path[-1, :]])
        curv_vec = trunc_path[1:, :] - trunc_path[:-1, :]
        curv_length = np.linalg.norm(curv_vec, axis=1)

        if curv_length[0] == 0.0:
            curv_direct = np.zeros((2,))
        else:
            curv_direct = curv_vec[0, :] / curv_length[0]
        proj_dist = np.dot(pos - trunc_path[0, :], curv_direct)

        if proj_dist >= curv_length[0] - proj_dist_buffer and local_index < self._num_waypoint - 1:
            self._global_path_index += 1
            return self.local_path(pos)

        # t_c = (proj_dist + proj_dist_buffer) / self._reference_speed
        t_c = proj_dist / self._reference_speed
        t_s = t_c + self._local_path_timestep * np.linspace(0, self._num_horizon - 1, self._num_horizon)

        curv_time = np.cumsum(np.hstack([0.0, curv_length / self._reference_speed]))
        curv_time[-1] += (
            t_c + 2 * self._local_path_timestep * self._num_horizon + proj_dist_buffer / self._reference_speed
        )

        path_idx = np.searchsorted(curv_time, t_s, side="right") - 1
        path = np.vstack([np.interp(t_s, curv_time, trunc_path[:, 0]), np.interp(t_s, curv_time, trunc_path[:, 1])]).T
        path_vel = self._reference_speed * np.ones((self._num_horizon, 1))
        path_head = np.arctan2(curv_vec[path_idx, 1], curv_vec[path_idx, 0]).reshape(self._num_horizon, 1)
        return np.hstack([path, path_vel, path_head])
