from utils import *
import numpy as np
import casadi as ca
import datetime


class NmpcDcbfOptimizerParam:
    def __init__(self):
        self.horizon = 4
        self.mat_Q = np.diag([100.0, 100.0, 1.0, 1.0])
        self.mat_R = np.diag([0.0, 0.0])
        self.mat_Rold = np.diag([1.0, 1.0])
        self.mat_dR = np.diag([1.0, 1.0])
        self.gamma = 0.9
        self.pomega = 1.0
        self.margin_dist = 0.01


class NmpcDbcfOptimizer:
    def __init__(self, variables: dict, costs: dict, dynamics_opt):
        self.opti = None
        self.variables = variables
        self.costs = costs
        self.dynamics_opt = dynamics_opt

    def set_state(self, state):
        self.state = state

    def initialize_variables(self, param):
        self.variables["x"] = self.opti.variable(4, param.horizon + 1)
        self.variables["u"] = self.opti.variable(2, param.horizon)

    def add_initial_condition_constraint(self):
        self.opti.subject_to(self.variables["x"][:, 0] == self.state._x)

    def add_input_constraint(self, param):
        # TODO: wrap params
        amin, amax = -1.0, 1.0
        omegamin, omegamax = -1.0, 1.0
        for i in range(param.horizon):
            # input constraints
            self.opti.subject_to(self.variables["u"][0, i] <= amax)
            self.opti.subject_to(amin <= self.variables["u"][0, i])
            self.opti.subject_to(self.variables["u"][1, i] <= omegamax)
            self.opti.subject_to(omegamin <= self.variables["u"][1, i])

    def add_dynamics_constraint(self, param):
        for i in range(param.horizon):
            self.opti.subject_to(
                self.variables["x"][:, i + 1] == self.dynamics_opt(self.variables["x"][:, i], self.variables["u"][:, i])
            )

    def add_reference_trajectory_tracking_cost(self, param, reference_trajectory):
        self.costs["reference_trajectory_tracking"] = 0
        for i in range(param.horizon):
            x_diff = self.variables["x"][:, i] - reference_trajectory[i, :]
            self.costs["reference_trajectory_tracking"] += ca.mtimes(x_diff.T, ca.mtimes(param.mat_Q, x_diff))

    def add_input_stage_cost(self, param):
        self.costs["input_stage"] = 0
        for i in range(param.horizon):
            self.costs["input_stage"] += ca.mtimes(
                self.variables["u"][:, i].T, ca.mtimes(param.mat_R, self.variables["u"][:, i])
            )

    def add_prev_input_cost(self, param):
        self.costs["prev_input"] = 0
        self.costs["prev_input"] += ca.mtimes(
            (self.variables["u"][:, 0] - self.state._u).T,
            ca.mtimes(param.mat_Rold, (self.variables["u"][:, 0] - self.state._u)),
        )

    def add_input_smoothness_cost(self, param):
        self.costs["input_smoothness"] = 0
        for i in range(param.horizon - 1):
            self.costs["input_smoothness"] += ca.mtimes(
                (self.variables["u"][:, i + 1] - self.variables["u"][:, i]).T,
                ca.mtimes(param.mat_dR, (self.variables["u"][:, i + 1] - self.variables["u"][:, i])),
            )

    def add_point_to_convex_constraint(self, param, obs_geo, safe_dist):
        # get current value of cbf
        mat_A, vec_b = obs_geo.get_convex_rep()
        cbf_curr, lamb_curr = get_dist_point_to_region(self.state._x[0:2], mat_A, vec_b)
        # filter obstacle if it's still far away
        if cbf_curr > safe_dist:
            return
        # duality-cbf constraints
        lamb = self.opti.variable(mat_A.shape[0], param.horizon)
        omega = self.opti.variable(param.horizon, 1)
        for i in range(param.horizon):
            self.opti.subject_to(lamb[:, i] >= 0)
            self.opti.subject_to(
                ca.mtimes((ca.mtimes(mat_A, self.variables["x"][0:2, i + 1]) - vec_b).T, lamb[:, i])
                >= omega[i] * param.gamma ** (i + 1) * cbf_curr + param.margin_dist
            )
            temp = ca.mtimes(mat_A.T, lamb[:, i])
            self.opti.subject_to(ca.mtimes(temp.T, temp) <= 1)
            self.opti.subject_to(omega[i] >= 0)
            self.costs["decay_rate_relaxing"] += param.pomega * (omega[i] - 1) ** 2
            # warm start
            self.opti.set_initial(lamb[:, i], lamb_curr)
            self.opti.set_initial(omega[i], 0.1)

    def add_convex_to_convex_constraint(self, param, robot_geo, obs_geo, safe_dist):
        mat_A, vec_b = obs_geo.get_convex_rep()
        robot_G, robot_g = robot_geo.convex_rep()
        # get current value of cbf
        cbf_curr, lamb_curr, mu_curr = get_dist_region_to_region(
            mat_A,
            vec_b,
            np.dot(robot_G, self.state.rotation().T),
            np.dot(np.dot(robot_G, self.state.rotation().T), self.state.translation()) + robot_g,
        )
        # filter obstacle if it's still far away
        if cbf_curr > safe_dist:
            return
        # duality-cbf constraints
        lamb = self.opti.variable(mat_A.shape[0], param.horizon)
        mu = self.opti.variable(robot_G.shape[0], param.horizon)
        omega = self.opti.variable(param.horizon, 1)
        for i in range(param.horizon):
            robot_R = ca.hcat(
                [
                    ca.vcat(
                        [
                            ca.cos(self.variables["x"][3, i + 1]),
                            ca.sin(self.variables["x"][3, i + 1]),
                        ]
                    ),
                    ca.vcat(
                        [
                            -ca.sin(self.variables["x"][3, i + 1]),
                            ca.cos(self.variables["x"][3, i + 1]),
                        ]
                    ),
                ]
            )
            robot_T = self.variables["x"][0:2, i + 1]
            self.opti.subject_to(lamb[:, i] >= 0)
            self.opti.subject_to(mu[:, i] >= 0)
            self.opti.subject_to(
                -ca.mtimes(robot_g.T, mu[:, i]) + ca.mtimes((ca.mtimes(mat_A, robot_T) - vec_b).T, lamb[:, i])
                >= omega[i] * param.gamma ** (i + 1) * cbf_curr + param.margin_dist
            )
            self.opti.subject_to(
                ca.mtimes(robot_G.T, mu[:, i]) + ca.mtimes(ca.mtimes(robot_R.T, mat_A.T), lamb[:, i]) == 0
            )
            temp = ca.mtimes(mat_A.T, lamb[:, i])
            self.opti.subject_to(ca.mtimes(temp.T, temp) <= 1)
            self.opti.subject_to(omega[i] >= 0)
            self.costs["decay_rate_relaxing"] += param.pomega * (omega[i] - 1) ** 2
            # warm start
            self.opti.set_initial(lamb[:, i], lamb_curr)
            self.opti.set_initial(mu[:, i], mu_curr)
            self.opti.set_initial(omega[i], 0.1)

    def add_obstacle_avoidance_constraint(self, param, system, obstacles_geo):
        self.costs["decay_rate_relaxing"] = 0
        # TODO: wrap params
        # TODO: move safe dist inside attribute `system`
        safe_dist = system._dynamics.safe_dist(system._state._x, 0.1, -1.0, 1.0, param.margin_dist)
        for obs_geo in obstacles_geo:
            # TODO: need to add case for `add_point_convex_constraint()`
            if isinstance(system._geometry.rep(), RectangleRegion):
                self.add_convex_to_convex_constraint(param, system._geometry, obs_geo, safe_dist)
                # self.add_point_to_convex_constraint(param, obs_geo, safe_dist
            else:
                raise NotImplementedError()

    def add_warm_start(self, param, system):
        # TODO: wrap params
        x_ws, u_ws = system._dynamics.nominal_safe_controller(self.state._x, 0.1, -1.0, 1.0)
        for i in range(param.horizon):
            self.opti.set_initial(self.variables["x"][:, i + 1], x_ws)
            self.opti.set_initial(self.variables["u"][:, i], u_ws)

    def setup(self, param, system, reference_trajectory, obstacles):
        self.set_state(system._state)
        self.opti = ca.Opti()
        self.initialize_variables(param)
        self.add_initial_condition_constraint()
        self.add_input_constraint(param)
        self.add_dynamics_constraint(param)
        self.add_reference_trajectory_tracking_cost(param, reference_trajectory)
        self.add_input_stage_cost(param)
        self.add_prev_input_cost(param)
        self.add_input_smoothness_cost(param)
        self.add_obstacle_avoidance_constraint(param, system, obstacles)
        self.add_warm_start(param, system)

    def solve_nlp(self):
        cost = 0
        for cost_name in self.costs:
            cost += self.costs[cost_name]
        self.opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        start_timer = datetime.datetime.now()
        self.opti.solver("ipopt", option)
        opt_sol = self.opti.solve()
        end_timer = datetime.datetime.now()
        delta_timer = end_timer - start_timer
        print("solver time: ", delta_timer.total_seconds())
        return opt_sol


class NmpcDcbfController:
    # TODO: Refactor this class to inheritate from a general optimizer
    def __init__(self, dynamics=None):
        self._param = NmpcDcbfOptimizerParam()
        self._optimizer = NmpcDbcfOptimizer({}, {}, dynamics.forward_dynamics_opt(0.1))

    def generate_control_input(self, system, global_path, local_trajectory, obstacles):
        self._optimizer.setup(self._param, system, local_trajectory, obstacles)
        self._opt_sol = self._optimizer.solve_nlp()
        return self._opt_sol.value(self._optimizer.variables["u"][:, 0])

    def logging(self, logger):
        logger._xtrajs.append(self._opt_sol.value(self._optimizer.variables["x"]).T)
        logger._utrajs.append(self._opt_sol.value(self._optimizer.variables["u"]).T)