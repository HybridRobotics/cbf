import matplotlib.pyplot as plt

from planning.path_generator.opt_planner import *

# # optimization_planner test
obstacles = []
mat_A = np.array([[2.0, 0.0], [0.0, 2.0]])
vec_b = np.array([10.0, 10.0])
obstacles.append(PolytopeRegion(mat_A, vec_b))
start_pos = np.array([7.5, 0.0])
goal_pos = np.array([0.0, 7.5])
optim = OptimizationPlanner(obstacles, margin=1.0)
path = optim.optimize(start_pos, goal_pos, 10)
print(path)
plt.plot(path[0, :], path[1, :])
plt.show()


# # maze environment test
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

optim = OptimizationPlanner(obstacles, margin=0.05)
try:
    path = optim.optimize(start, goal, 40)
    print(path)
    fig, ax = plt.subplots()
    for o in obstacles:
        patch = o.get_plot_patch()
        ax.add_patch(patch)
    ax.plot(path_arr[0, :], path_arr[1, :])
    plt.show()
except:
    print("Global path not found using optimization")
