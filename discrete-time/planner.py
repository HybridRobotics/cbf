import numpy as np
import sympy as sp
import math


class Planner:
    def __init__(self, globalpath, reference_speed, num_horizon, localpath_timestep):
        # global path
        self._global_path = globalpath
        self._global_path_index = 0
        # local path
        self._reference_speed = reference_speed
        self._num_horizon = num_horizon
        self._local_path_timestep = localpath_timestep

    ### Planner: smooth trajectory with sampling waypoints (to track)
    def global_path(self):
        """Return global path
        """
        return self._global_path

    def local_path(self, pos):
        """Return local path
        """
        proj_dist_buffer = 0.05  # Hacking value to avoid getting stuck locally
        path = []
        # initialize node index
        local_index = self._global_path_index
        curr_node = self._global_path[local_index, :]
        next_node = self._global_path[local_index + 1, :]
        curv_vec = next_node - curr_node
        curv_length = np.linalg.norm(curv_vec)
        curv_direct = curv_vec / curv_length
        # trim local path
        proj_dist = np.dot(pos - curr_node, curv_direct)
        if proj_dist <= 0:
            proj_dist = 0
        elif proj_dist >= curv_length - proj_dist_buffer:
            self._global_path_index += 1
            self.local_path(pos)
        else:
            pass
        # generate path
        curv_dist = proj_dist
        for index in range(self._num_horizon):
            # it might be sampled into next segment
            while curv_dist >= curv_length:
                curv_dist -= curv_length
                local_index += 1
                curr_node = self._global_path[local_index, :]
                next_node = self._global_path[local_index + 1, :]
                curv_vec = next_node - curr_node
                curv_length = np.linalg.norm(curv_vec)
                curv_direct = curv_vec / curv_length
            # Add point
            point = curr_node + curv_dist * curv_direct
            # path.append(point)
            path.append(
                np.array([point[0], point[1], self._reference_speed, math.atan2(curv_direct[1], curv_direct[0])])
            )
            # Update curvilinear distance for next point
            curv_dist += self._reference_speed * self._local_path_timestep
        return np.vstack(path)
