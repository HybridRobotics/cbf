import heapq as hq
import math

import numpy as np

from models.geometry_utils import *


# TODO: Generalize to 3D?
class Node:
    def __init__(self, pos, parent=None, g_cost=math.inf, f_cost=math.inf):
        self.pos = pos
        self.parent = parent
        self.g_cost = g_cost
        self.f_cost = f_cost

    def __eq__(self, other):
        return all(self.pos == other.pos)

    def __le__(self, other):
        if self.pos[0] == other.pos[0]:
            return self.pos[1] <= other.pos[1]
        else:
            return self.pos[0] <= other.pos[0]

    def __lt__(self, other):
        if self.pos[0] == other.pos[0]:
            return self.pos[1] < other.pos[1]
        else:
            return self.pos[0] < other.pos[0]


# TODO: Generalize to 3D
class GridMap:
    # cell_size > 0; don't make cell_size too small
    def __init__(self, bounds=((0.0, 0.0), (10.0, 10.0)), cell_size=0.1, quad=True):
        self.bounds = bounds
        self.cell_size = cell_size
        self.quad = quad
        self.Nx = math.ceil((bounds[1][0] - bounds[0][0]) / cell_size)
        self.Ny = math.ceil((bounds[1][1] - bounds[0][1]) / cell_size)

        pos = lambda i, j: np.array([bounds[0][0] + (i + 0.5) * cell_size, bounds[0][1] + (j + 0.5) * cell_size])
        self.grid = [[Node(pos(i, j)) for j in range(self.Ny)] for i in range(self.Nx)]

    # pos should be within bounds
    def set_node(self, pos, parent, g_cost, f_cost):
        i_x = math.floor((pos[0] - self.bounds[0][0]) / self.cell_size)
        i_y = math.floor((pos[1] - self.bounds[0][1]) / self.cell_size)
        self.grid[i_x][i_y].parent = parent
        self.grid[i_x][i_y].g_cost = g_cost
        self.grid[i_x][i_y].f_cost = f_cost
        return self.grid[i_x][i_y]

    # pos should be within bounds
    def get_node(self, pos):
        i_x = math.floor((pos[0] - self.bounds[0][0]) / self.cell_size)
        i_y = math.floor((pos[1] - self.bounds[0][1]) / self.cell_size)
        return self.grid[i_x][i_y]

    def get_neighbours(self, node):
        i_x = math.floor((node.pos[0] - self.bounds[0][0]) / self.cell_size)
        i_y = math.floor((node.pos[1] - self.bounds[0][1]) / self.cell_size)
        neighbours = []
        for i in range(i_x - 1, i_x + 2):
            for j in range(i_y - 1, i_y + 2):
                if i == i_x and j == i_y:
                    continue
                if self.quad:
                    if 0 <= i <= self.Nx - 1 and 0 <= j <= self.Ny - 1 and abs(i - i_x) + abs(j - i_y) <= 1:
                        neighbours.append(self.grid[i][j])
                else:
                    if 0 <= i <= self.Nx - 1 and 0 <= j <= self.Ny - 1:
                        neighbours.append(self.grid[i][j])
        return neighbours


class GraphSearch:
    def __init__(self, graph, obstacles, margin):
        self.graph = graph
        self.obstacles = obstacles
        self.margin = margin

    def a_star(self, start_pos, goal_pos):
        h_cost = lambda pos: np.linalg.norm(goal_pos - pos)
        edge_cost = lambda n1, n2: np.linalg.norm(n1.pos - n2.pos)

        openSet = []
        start = self.graph.set_node(start_pos, None, 0.0, h_cost(start_pos))
        goal = self.graph.get_node(goal_pos)

        hq.heappush(openSet, (start.f_cost, start))

        while len(openSet) > 0:
            current = openSet[0][1]
            if current == goal:
                return self.reconstruct_path(current)

            hq.heappop(openSet)

            for n in self.graph.get_neighbours(current):
                if self.check_collision(n.pos):
                    continue
                g_score = current.g_cost + edge_cost(current, n)
                if g_score < n.g_cost:
                    n_ = self.graph.set_node(n.pos, current, g_score, g_score + h_cost(n.pos))
                    if not n in (x[1] for x in openSet):
                        hq.heappush(openSet, (n_.f_cost, n_))
        return []

    def theta_star(self, start_pos, goal_pos):
        h_cost = lambda pos: np.linalg.norm(goal_pos - pos)
        edge_cost = lambda n1, n2: np.linalg.norm(n1.pos - n2.pos)

        openSet = []
        start = self.graph.set_node(start_pos, None, 0.0, h_cost(start_pos))
        goal = self.graph.get_node(goal_pos)

        hq.heappush(openSet, (start.f_cost, start))

        while len(openSet) > 0:
            current = openSet[0][1]
            if current == goal:
                return self.reconstruct_path(current)

            hq.heappop(openSet)

            for n in self.graph.get_neighbours(current):
                if self.check_collision(n.pos):
                    continue
                if (not current.parent is None) and self.line_of_sight(current.parent, n):
                    g_score = current.parent.g_cost + edge_cost(current.parent, n)
                    if g_score < n.g_cost:
                        n_ = self.graph.set_node(n.pos, current.parent, g_score, g_score + h_cost(n.pos))
                        # delete n from min-heap
                        for i in range(len(openSet)):
                            if openSet[i][1] == n:
                                openSet[i] = openSet[-1]
                                openSet.pop()
                                if i < len(openSet):
                                    hq._siftup(openSet, i)
                                    hq._siftdown(openSet, 0, i)
                                break
                        hq.heappush(openSet, (n_.f_cost, n_))
                else:
                    g_score = current.g_cost + edge_cost(current, n)
                    if g_score < n.g_cost:
                        n_ = self.graph.set_node(n.pos, current, g_score, g_score + h_cost(n.pos))
                        # delete n from min-heap
                        for i in range(len(openSet)):
                            if openSet[i][1] == n:
                                openSet[i] = openSet[-1]
                                openSet.pop()
                                if i < len(openSet):
                                    hq._siftup(openSet, i)
                                    hq._siftdown(openSet, 0, i)
                                break
                        hq.heappush(openSet, (n_.f_cost, n_))
        return []

    # TODO: optimize
    def line_of_sight(self, n1, n2):
        e = self.graph.cell_size
        div = np.linalg.norm(n2.pos - n1.pos) / e
        for i in range(1, math.floor(div) + 1):
            if self.check_collision((n2.pos * i + n1.pos * (div - i)) / div):
                return False
        return True

    def check_collision(self, pos):
        for o in self.obstacles:
            A, b = o.get_convex_rep()
            b = b.reshape((len(b),))
            if all(A @ pos - b - self.margin * np.linalg.norm(A, axis=1) <= 0):
                return True
        return False

    def reconstruct_path(self, node):
        path = [node]
        while not node.parent is None:
            node = node.parent
            path.append(node)
        return [path[len(path) - i - 1] for i in range(len(path))]

    def reduce_path(self, path):
        red_path = []
        if len(path) > 1:
            for i in range(1, len(path)):
                if (not path[i].parent.parent is None) and self.line_of_sight(path[i], path[i].parent.parent):
                    path[i].parent = path[i].parent.parent
                else:
                    red_path.append(path[i].parent)
        red_path.append(path[-1])
        return red_path
