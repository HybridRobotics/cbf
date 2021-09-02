import math

import matplotlib.pyplot as plt

from planning.path_generator.astar import *

# # Node test
pos = np.array([1.0, 1.0])
node_1 = Node(pos=pos)
node_2 = Node(pos=2 * pos, parent=node_1, f_cost=1.0)
print(node_1.parent)
print(node_2.parent.pos)
print(node_2.g_cost)
print(node_1 == node_2, "\n")


# # GridMap test
grid_1 = GridMap()
grid_2 = GridMap(bounds=((1.0, 1.0), (2.0, 1.3)), cell_size=0.1, quad=False)
print(grid_1.Nx, " ", grid_1.Ny)
print(grid_2.Nx, " ", grid_2.Ny)
print(len(grid_2.grid[0]))
grid_2_pos = [[grid_2.grid[i][j].pos for i in range(grid_2.Nx)] for j in range(grid_2.Ny)]
print(grid_2_pos)
grid_2_node = grid_2.set_node(
    grid_2_pos[0][0], parent=grid_2.grid[grid_2.Nx - 1][grid_2.Ny - 1], g_cost=1.0, f_cost=1.0
)
print(grid_2.get_node(grid_2_pos[0][0]) == grid_2_node)
print([n.pos for n in grid_1.get_neighbours(grid_1.grid[0][0])])
print([n.pos for n in grid_1.get_neighbours(grid_1.grid[grid_1.Nx // 2][grid_1.Ny // 2])])
print([n.pos for n in grid_1.get_neighbours(grid_1.grid[grid_1.Nx - 1][grid_1.Ny - 1])])
print([n.pos for n in grid_2.get_neighbours(grid_2.grid[0][0])])
print([n.pos for n in grid_2.get_neighbours(grid_2.grid[grid_2.Nx // 2][grid_2.Ny // 2])])
print([n.pos for n in grid_2.get_neighbours(grid_2.grid[grid_2.Nx - 1][grid_2.Ny - 1])], "\n")


# # GraphSearch test
obstacles = []
mat_A = np.array([[2.0, 0.0], [0.0, 2.0]])
vec_b = np.array([10.0, 10.0])
obstacles.append(PolytopeRegion(mat_A, vec_b))
graph = GraphSearch(graph=grid_1, obstacles=obstacles, margin=1.0)
print(graph.check_collision(grid_1.grid[0][0].pos))
print(graph.check_collision(grid_1.grid[grid_1.Nx - 1][grid_1.Ny - 1].pos))
start_pos = np.array([7.5, 0.0])
goal_pos = np.array([0.0, 7.5])

# astar test with quad directional neighbours
path = graph.a_star(start_pos, goal_pos)
path_arr = np.array([p.pos for p in path])
plt.plot(path_arr[:, 0], path_arr[:, 1])
plt.show()

# astar test with octo directional neighbours
grid_1 = GridMap(quad=False)
graph = GraphSearch(graph=grid_1, obstacles=obstacles, margin=1.0)
path = graph.a_star(start_pos, goal_pos)
path_arr = np.array([p.pos for p in path])
plt.plot(path_arr[:, 0], path_arr[:, 1])
plt.show()

# reduce_path test
reduced_path = graph.reduce_path(path)
reduced_path_arr = np.array([p.pos for p in reduced_path])
plt.plot(reduced_path_arr[:, 0], reduced_path_arr[:, 1])
plt.show()

# theta_star test
grid_1 = GridMap(quad=False)
graph = GraphSearch(graph=grid_1, obstacles=obstacles, margin=1.0)

path = graph.theta_star(start_pos, goal_pos)
path_arr = np.array([p.pos for p in path])
plt.plot(path_arr[:, 0], path_arr[:, 1])
plt.show()


# # maze environment test
def plot_map(obstacles, path_arr):
    fig, ax = plt.subplots()
    for o in obstacles:
        patch = o.get_plot_patch()
        ax.add_patch(patch)
    ax.plot(path_arr[:, 0], path_arr[:, 1])
    plt.show()


s = 0.2  # scale of environment
start = np.array([0.5 * s, 5.5 * s])
goal = np.array([12.5 * s, 0.5 * s])
bounds = ((0.0 * s, 0.0 * s), (13.0 * s, 6.0 * s))
cell_size = 0.25 * s
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

# astar test with quad directional neighbours
grid = GridMap(bounds=bounds, cell_size=cell_size, quad=True)
graph = GraphSearch(graph=grid, obstacles=obstacles, margin=0.05)
print(graph.check_collision(start))
path = graph.a_star(start, goal)
path_arr = np.array([p.pos for p in path])
plot_map(obstacles, path_arr)

# astar test with octo directional neighbours
grid = GridMap(bounds=bounds, cell_size=cell_size, quad=False)
graph = GraphSearch(graph=grid, obstacles=obstacles, margin=0.05)
path = graph.a_star(start, goal)
path_arr = np.array([p.pos for p in path])
plot_map(obstacles, path_arr)

# reduce_path test
reduced_path = graph.reduce_path(path)
reduced_path_arr = np.array([p.pos for p in reduced_path])
plot_map(obstacles, reduced_path_arr)

# theta_star test
grid = GridMap(bounds=bounds, cell_size=cell_size, quad=False)
graph = GraphSearch(graph=grid, obstacles=obstacles, margin=0.05)
path = graph.theta_star(start, goal)
path_arr = np.array([p.pos for p in path])
plot_map(obstacles, path_arr)
print(path_arr)
