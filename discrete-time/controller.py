import numpy as np
import sympy as sp
import casadi as ca
import math, datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import utils


class DubinCarDyn:
    @staticmethod
    def forward_dynamics(x, u, timestep):
        """Return updated state in a form of `np.ndnumpy`
        """
        x_next = np.ndarray(shape=(4,), dtype=float)
        x_next[0] = x[0] + x[2] * math.cos(x[3]) * timestep
        x_next[1] = x[1] + x[2] * math.sin(x[3]) * timestep
        x_next[2] = x[2] + u[0] * timestep
        x_next[3] = x[3] + u[1] * timestep
        return x_next

    @staticmethod
    def forward_dynamics_opt(timestep):
        """Return updated state in a form of `ca.SX`
        """
        x_symbol = ca.SX.sym("x", 4)
        u_symbol = ca.SX.sym("u", 2)
        x_symbol_next = x_symbol[0] + x_symbol[2] * ca.cos(x_symbol[3]) * timestep
        y_symbol_next = x_symbol[1] + x_symbol[2] * ca.sin(x_symbol[3]) * timestep
        v_symbol_next = x_symbol[2] + u_symbol[0] * timestep
        theta_symbol_next = x_symbol[3] + u_symbol[1] * timestep
        state_symbol_next = ca.vertcat(x_symbol_next, y_symbol_next, v_symbol_next, theta_symbol_next)
        return ca.Function("dubin_car_dyn", [x_symbol, u_symbol], [state_symbol_next])


class DubinCarGeo:
    def __init__(self, length, width):
        self.__length = length
        self.__width = width
        self.__region = utils.RectangleRegion(-length / 2, length / 2, -width / 2, width / 2)

    def get_params(self):
        return self.__length, self.__width

    def get_convex_rep(self):
        return self.__region.get_convex_rep()


class DualityController:
    def __init__(
        self,
        sys_timestep,
        sim_timestep,
        ctrl_timestep,
        vehicle_length,
        vehicle_width,
        num_horizon_opt,
        num_horizon_cbf,
        gamma,
        dist_margin,
    ):
        # System
        self._state = None
        self._input = None
        self._time = 0.0
        self.__sys_timestep = sys_timestep
        self.__sim_timestep = sim_timestep
        # Planner
        self._planner = None
        # System model
        self.__ctrl_timestep = ctrl_timestep
        self.__dynamics_model = DubinCarDyn.forward_dynamics_opt(ctrl_timestep)
        self.__geometry_model = DubinCarGeo(vehicle_length, vehicle_width)
        # Optimal control with obstacle avoidance
        self.__num_horizon_opt = num_horizon_opt
        self.__num_horizon_cbf = num_horizon_cbf
        self._obstacles = []
        self._gamma = gamma
        self._dist_margin = dist_margin
        # Logging
        self._state_log = []
        self._input_log = []
        self._local_path_log = []
        self._openloop_state_log = []

    def set_state(self, x):
        self._state = x

    def set_input(self, u):
        self._input = u

    def set_planner(self, planner):
        self._planner = planner

    def add_obstacle(self, obs):
        self._obstacles.append(obs)

    def set_obstacles(self, obs_list):
        for obs in obs_list:
            self.add_obstacle(obs)

    def set_obstacle_avoidance_policy(self, policy):
        self._obstacle_avoidance_policy = policy

    def dynamics(self):
        # Logging
        self._state_log.append(self._state)
        self._input_log.append(self._input)
        # Simulate the dynamical system
        count = 0
        while self.__sim_timestep * count < self.__sys_timestep:
            self._state = DubinCarDyn.forward_dynamics(self._state, self._input, self.__sim_timestep)
            count += 1
        self._time += self.__sys_timestep

    def sim(self, simulation_time):
        while self._time < simulation_time:
            # calculate optimal control input
            self.calc_input()
            # simulate the system
            self.dynamics()
        self._state_log.append(self._state)

    def get_translation(self):
        return np.array([[self._state[0]], [self._state[1]]])

    def get_rotation(self):
        return np.array(
            [
                [math.cos(self._state[3]), math.sin(self._state[3])],
                [-math.sin(self._state[3]), math.cos(self._state[3])],
            ]
        )

    def calc_input(self):
        opti = ca.Opti()
        # variables and cost
        x = opti.variable(4, self.__num_horizon_opt + 1)
        u = opti.variable(2, self.__num_horizon_opt)
        cost = 0
        # hyperparameters
        mat_Q = np.diag([100.0, 100.0, 0.0, 1.0])
        mat_R = np.diag([0.0, 0.0])
        pomega = 1.0
        amin, amax = -1.0, 1.0
        omegamin, omegamax = -2.0, 2.0
        # initial condition
        opti.subject_to(x[:, 0] == self._state)
        # get reference trajectory from local planner
        local_path = self._planner.local_path(self._state[0:2])
        # input constraints / dynamics constraints / stage cost
        for i in range(self.__num_horizon_opt):
            # input constraints
            opti.subject_to(u[0, i] <= amax)
            opti.subject_to(amin <= u[0, i])
            opti.subject_to(u[1, i] <= omegamax)
            opti.subject_to(omegamin <= u[1, i])
            # dynamics constraints
            opti.subject_to(x[:, i + 1] == self.__dynamics_model(x[:, i], u[:, i]))
            # state stage cost
            x_diff = x[:, i] - local_path[i, :]
            cost += ca.mtimes(x_diff.T, ca.mtimes(mat_Q, x_diff))
            # input stage cost
            cost += ca.mtimes(u[:, i].T, ca.mtimes(mat_R, u[:, i]))
        # obstacle avoidance
        if self._obstacles != None:
            for obs in self._obstacles:
                # get current value of cbf
                mat_A, vec_b = obs.get_convex_rep()
                if self._obstacle_avoidance_policy == "point2region":
                    # get current value of cbf
                    mat_A, vec_b = obs.get_convex_rep()
                    cbf_curr = utils.get_dist_point_to_region(self._state[0:2], mat_A, vec_b)
                    # duality-cbf constraints
                    lamb = opti.variable(mat_A.shape[0], self.__num_horizon_cbf)
                    omega = opti.variable(self.__num_horizon_cbf, 1)
                    for i in range(self.__num_horizon_cbf):
                        opti.subject_to(lamb[:, i] >= 0)
                        opti.subject_to(
                            ca.mtimes((ca.mtimes(mat_A, x[0:2, i + 1]) - vec_b).T, lamb[:, i])
                            >= omega[i] * self._gamma ** (i + 1) * cbf_curr + self._dist_margin
                        )
                        temp = ca.mtimes(mat_A.T, lamb[:, i])
                        opti.subject_to(ca.mtimes(temp.T, temp) <= 1)
                        opti.subject_to(omega[i] >= 0)
                        cost += pomega * (omega[i] - 1) ** 2
                if self._obstacle_avoidance_policy == "region2region":
                    robot_G, robot_g = self.__geometry_model.get_convex_rep()
                    # get current value of cbf
                    cbf_curr = utils.get_dist_region_to_region(
                        mat_A,
                        vec_b,
                        np.dot(robot_G, self.get_rotation().T),
                        np.dot(np.dot(robot_G, self.get_rotation().T), self.get_translation()),
                    )
                    # duality-cbf constraints
                    lamb = opti.variable(mat_A.shape[0], self.__num_horizon_cbf)
                    mu = opti.variable(vec_b.shape[0], self.__num_horizon_cbf)
                    omega = opti.variable(self.__num_horizon_cbf, 1)
                    for i in range(self.__num_horizon_cbf):
                        robot_R = ca.hcat(
                            [
                                ca.vcat([ca.cos(x[2, i + 1]), ca.sin(x[2, i + 1])]),
                                ca.vcat([-ca.sin(x[2, i + 1]), ca.cos(x[2, i + 1])]),
                            ]
                        )
                        robot_T = x[0:2, i + 1]
                        opti.subject_to(lamb[:, i] >= 0)
                        opti.subject_to(mu[:, i] >= 0)
                        opti.subject_to(
                            -ca.mtimes(robot_g.T, mu[:, i])
                            + ca.mtimes((ca.mtimes(mat_A, robot_T) - vec_b).T, lamb[:, i])
                            >= omega[i] * self._gamma ** (i + 1) * cbf_curr + self._dist_margin
                        )
                        opti.subject_to(
                            ca.mtimes(robot_G.T, mu[:, i]) + ca.mtimes(ca.mtimes(robot_R.T, mat_A.T), lamb[:, i]) == 0
                        )
                        temp = ca.mtimes(mat_A.T, lamb[:, i])
                        opti.subject_to(ca.mtimes(temp.T, temp) <= 1)
                        opti.subject_to(omega[i] >= 0)
                        cost += pomega * (omega[i] - 1) ** 2
        # solve optimization
        opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", option)
        start_timer = datetime.datetime.now()
        opt_sol = opti.solve()
        end_timer = datetime.datetime.now()
        delta_timer = end_timer - start_timer
        print("solver time: ", delta_timer.total_seconds())
        self._input = opt_sol.value(u[:, 0])
        # Logging
        self._local_path_log.append(local_path)
        self._openloop_state_log.append(opt_sol.value(x).T)

    def plot_world(self):
        print("Generate plotting")
        fig, ax = plt.subplots()
        plt.axis("equal")
        # plot robot's global path
        global_path = self._planner.global_path()
        ax.plot(global_path[:, 0], global_path[:, 1], "bo--", linewidth=1, markersize=4)
        # plot robot's closed-loop trajectory
        closedloop_states = np.vstack(self._state_log)
        ax.plot(closedloop_states[:, 0], closedloop_states[:, 1], "k-", linewidth=2, markersize=4)
        # plot obstacles
        if self._obstacles != None:
            for obs in self._obstacles:
                rec_patch = obs.get_plot_patch()
                ax.add_patch(rec_patch)
        plt.savefig("figures/world.eps", format="eps", dpi=1000, pad_inches=0)

    def plot_states(self):
        fig, ax = plt.subplots()
        tspan = np.linspace(0, self._time, round(self._time / self.__sys_timestep) + 1)
        # plot robot's closed-loop states
        states = np.vstack(self._state_log)
        ax.plot(tspan, states[:, 2])
        plt.savefig("figures/states-profile.eps", format="eps", dpi=1000, pad_inches=0)

    def animate_world(self):
        print("Generate animation")
        fig, ax = plt.subplots()
        # plot robot's global path
        global_path = self._planner.global_path()
        ax.plot(global_path[:, 0], global_path[:, 1], "bo--", linewidth=1, markersize=4)
        # plot robot's closed-loop trajectory
        closedloop_states = np.vstack(self._state_log)
        ax.plot(closedloop_states[:, 0], closedloop_states[:, 1], "k-", linewidth=2, markersize=4)
        # plot obstacles
        if self._obstacles != None:
            for obs in self._obstacles:
                rec_patch = obs.get_plot_patch()
                ax.add_patch(rec_patch)
        tpsan = np.linspace(0, self._time, round(self._time / self.__sys_timestep) + 1)
        ### Initialize data for animation
        frames = len(self._local_path_log)
        # initialize local reference trajectory
        local_path = self._local_path_log[0]
        (line_localpath,) = ax.plot(local_path[:, 0], local_path[:, 1])
        # initialize open-loop trajectory
        openloop_state = self._openloop_state_log[0]
        (line_openloop_state,) = ax.plot(openloop_state[:, 0], openloop_state[:, 1])
        if self._obstacle_avoidance_policy == "region2region":
            # initialize vehicle
            polygon_points = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])
            vehicle_polygon = patches.Polygon(
                polygon_points, alpha=1.0, closed=True, fc="None", ec="tab:brown", zorder=10, linewidth=2
            )
            ax.add_patch(vehicle_polygon)

        def update(index):
            # update local reference trajectory
            localpath = self._local_path_log[index]
            line_localpath.set_data(localpath[:, 0], localpath[:, 1])
            # update open-loop trajectory
            openloop_state = self._openloop_state_log[index]
            line_openloop_state.set_data(openloop_state[:, 0], openloop_state[:, 1])
            if self._obstacle_avoidance_policy == "region2region":
                # update vehicle
                length, width = self.__geometry_model.get_params()
                x, y, theta = closedloop_states[index, 0], closedloop_states[index, 1], closedloop_states[index, 3]
                vehicle_points = np.array(
                    [
                        [
                            x + length / 2 * np.cos(theta) - width / 2 * np.sin(theta),
                            y + length / 2 * np.sin(theta) + width / 2 * np.cos(theta),
                        ],
                        [
                            x + length / 2 * np.cos(theta) + width / 2 * np.sin(theta),
                            y + length / 2 * np.sin(theta) - width / 2 * np.cos(theta),
                        ],
                        [
                            x - length / 2 * np.cos(theta) + width / 2 * np.sin(theta),
                            y - length / 2 * np.sin(theta) - width / 2 * np.cos(theta),
                        ],
                        [
                            x - length / 2 * np.cos(theta) - width / 2 * np.sin(theta),
                            y - length / 2 * np.sin(theta) + width / 2 * np.cos(theta),
                        ],
                    ]
                )
                vehicle_polygon.set_xy(vehicle_points)

        anim = animation.FuncAnimation(fig, update, frames=frames, interval=100)
        anim.save("animation/world.gif", dpi=200, writer=animation.PillowWriter(fps=30))

