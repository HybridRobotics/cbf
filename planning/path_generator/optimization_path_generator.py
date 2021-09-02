import sys

import numpy as np

from planning.path_generator.opt_planner import *


class OptPathGenerator:
    def __init__(self, margin=0.05, horizon=10):
        self._global_path = None
        self._margin = margin
        self._horizon = horizon

    def generate_path(self, sys, obstacles, goal_pos):
        optim = optimization_planner(obstacles, margin=self._margin)
        try:
            path = optim.optimize(sys.get_state()[:2], goal_pos, self._horizon)
        except:
            print("Global path not found using optimization")
        self._global_path = path.T
        if False:
            plot_global_map(self._global_path, obstacles)
        return self._global_path

    def logging(self, logger):
        logger._paths.append(self._global_path)
