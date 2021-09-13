import sys

import numpy as np

from planning.path_generator.astar import *


def plot_global_map(path, obstacles):
    fig, ax = plt.subplots()
    for o in obstacles:
        patch = o.get_plot_patch()
        ax.add_patch(patch)
    ax.plot(path[:, 0], path[:, 1])
    plt.xlim([-1 * 0.15, 11 * 0.15])
    plt.ylim([0 * 0.15, 8 * 0.15])
    plt.show()


class AstarPathGenerator:
    def __init__(self, grid, quad, margin):
        self._global_path = None
        self._grid = GridMap(bounds=grid[0], cell_size=grid[1], quad=quad)
        self._margin = margin

    def generate_path(self, sys, obstacles, goal_pos):
        graph = GraphSearch(graph=self._grid, obstacles=obstacles, margin=self._margin)
        path = graph.a_star(sys.get_state()[:2], goal_pos)
        self._global_path = np.array([p.pos for p in path])
        print(self._global_path)
        if self._global_path == []:
            print("Global Path not found.")
            sys.exit(1)
        if True:
            plot_global_map(self._global_path, obstacles)
        return self._global_path

    def logging(self, logger):
        logger._paths.append(self._global_path)


class AstarLoSPathGenerator:
    def __init__(self, grid, quad, margin):
        self._global_path = None
        self._grid = GridMap(bounds=grid[0], cell_size=grid[1], quad=quad)
        self._margin = margin

    def generate_path(self, sys, obstacles, goal_pos):
        graph = GraphSearch(graph=self._grid, obstacles=obstacles, margin=self._margin)
        path = graph.a_star(sys.get_state()[:2], goal_pos)
        path = graph.reduce_path(path)
        self._global_path = np.array([p.pos for p in path])
        print(self._global_path)
        if self._global_path == []:
            print("Global Path not found.")
            sys.exit(1)
        if False:
            plot_global_map(self._global_path, obstacles)
        return self._global_path

    def logging(self, logger):
        logger._paths.append(self._global_path)


class ThetaStarPathGenerator:
    def __init__(self, grid, quad, margin):
        self._global_path = None
        self._grid = GridMap(bounds=grid[0], cell_size=grid[1], quad=False)
        self._margin = margin

    def generate_path(self, sys, obstacles, goal_pos):
        graph = GraphSearch(graph=self._grid, obstacles=obstacles, margin=self._margin)
        path = graph.theta_star(sys.get_state()[:2], goal_pos)
        self._global_path = np.array([p.pos for p in path])
        print(self._global_path)
        if self._global_path == []:
            print("Global Path not found.")
            sys.exit(1)
        if True:
            plot_global_map(self._global_path, obstacles)
        return self._global_path

    def logging(self, logger):
        logger._paths.append(self._global_path)
