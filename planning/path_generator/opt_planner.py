import math

import casadi as ca
import numpy as np

from models.geometry_utils import *


class OptimizationPlanner(object):
    def __init__(self, obstacles, margin):
        self.opt = ca.Opti()
        self.obstacles = obstacles
        self.margin = margin
        self.guess = None

    def optimize(self, start_pos, goal_pos, horizon):
        p = self.opt.variable(2, horizon + 2)
        gamma = self.opt.variable(horizon + 1, 1)

        # set initial and final positions:
        self.opt.subject_to(p[:, 0] == start_pos)
        self.opt.subject_to(p[:, -1] == goal_pos)

        # set cost
        cost = 0.0
        for t in range(1, horizon + 2):
            # x_diff = p[:,t] - goal_pos
            # cost += ca.mtimes(x_diff.T, x_diff)
            x_diff = p[:, t] - p[:, t - 1]
            cost += ca.mtimes(x_diff.T, x_diff)
            # cost += ca.norm_2(x_diff)

        # set obstacle avoidance constraints
        for o in self.obstacles:
            mat_A, vec_b = o.get_convex_rep()
            mu = self.opt.variable(horizon + 1, mat_A.shape[0])
            for t in range(horizon + 1):
                norm = ca.mtimes(mu[t, :], mat_A)
                # separation constraint
                self.opt.subject_to(
                    -ca.mtimes(norm, norm.T) / 4 + ca.mtimes(norm, p[:, t + 1]) - ca.mtimes(mu[t, :], vec_b) - gamma[t]
                    >= self.margin ** 2
                )
                # selection constraint
                self.opt.subject_to(gamma[t] + ca.mtimes(norm, (p[:, t] - p[:, t + 1])) >= 0)
                # non-negativity constraints
                self.opt.subject_to(gamma[t] >= 0)
                self.opt.subject_to(mu[t, :] >= 0)
            # set mu initia; guess
            if self.guess is None:
                self.opt.set_initial(mu, 0.1 * np.ones((horizon + 1, mat_A.shape[0])))
            else:
                self.opt.set_initial(mu, self.guess.mu)

        # set initial guess
        if self.guess is None:
            for t in range(horizon + 1):
                self.opt.set_initial(p[:, t], start_pos + (goal_pos - start_pos) * t / (horizon + 1))
                self.opt.set_initial(gamma[t], 0.0)
            self.opt.set_initial(p[:, -1], goal_pos)
        else:
            self.opt.set_initial(p, self.guess.p)
            self.opt.set_initial(gamma, self.guess.gamma)

        # solve opt
        self.opt.minimize(cost)
        options_p = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        options_s = {"max_iter": 10000}
        self.opt.solver("ipopt", options_p, options_s)
        self.sol = self.opt.solve()

        return self.sol.value(p)

    def warm_start(self, guess):
        self.guess = guess
