from models.geometry_utils import *

# # PolytopeRegion test
obs_1 = RectangleRegion(0.0, 1.0, 0.90, 1.0)
A, b = obs_1.get_convex_rep()
obs_2 = PolytopeRegion(A, b)
obs_3 = PolytopeRegion.convex_hull(obs_2.points)
print(obs_3.mat_A)
print(obs_3.vec_b)
print(all([all(obs_3.points[:, i] == obs_2.points[:, i]) for i in range(2)]))
obs_4 = PolytopeRegion.convex_hull(np.array([[3, 0], [4, 1], [5, 3], [3.5, 5], [1, 2]]))
print(obs_4.mat_A)
print(obs_4.vec_b)
fig, ax = plt.subplots()
patch_2 = obs_2.get_plot_patch()
patch_4 = obs_4.get_plot_patch()
ax.add_patch(patch_2)
ax.add_patch(patch_4)
plt.xlim([-5, 10])
plt.ylim([-5, 10])
plt.show()


# # dual variable test in distance functions
# point_to_region
obs_1 = RectangleRegion(0.0, 1.0, 0.90, 1.0)
A, b = obs_1.get_convex_rep()
point = np.array([[0.0], [0.0]])
dist_1, dual_1 = get_dist_point_to_region(point, A, b)
print(A)
print(dist_1)
print(dual_1)
# before normalization of dual in distance function
# print(np.sqrt(-0.25*dual_1@A@A.T@dual_1 + dual_1@(A@point - b)))
# after normalization of dual in distance function
print(dual_1 @ (A @ point - b))
print(np.linalg.norm(dual_1 @ A), "\n")

# region_to_region
obs_2 = RectangleRegion(0.0, 1.0, -0.90, 0.0)
C, d = obs_2.get_convex_rep()
dist_2, dual_2, dual_3 = get_dist_region_to_region(A, b, C, d)
print(C)
print(dist_2)
print(dual_2, " ", dual_3)
# before normalization of dual in distance function
# print(np.sqrt(-0.25*dual_2@A@A.T@dual_2 - dual_2@b - dual_3@d))
# after normalization of dual in distance function
print(-dual_2 @ b - dual_3 @ d)
print(np.linalg.norm(dual_2 @ A), " ", np.linalg.norm(dual_3 @ C), "\n")
