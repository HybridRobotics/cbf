from control.dcbf_optimizer import NmpcDbcfOptimizer, NmpcDcbfOptimizerParam


class NmpcDcbfController:
    # TODO: Refactor this class to inheritate from a general optimizer
    def __init__(self, dynamics=None, opt_param=None):
        # TODO: Rename self._param into self._opt_param to avoid ambiguity
        self._param = opt_param
        self._optimizer = NmpcDbcfOptimizer({}, {}, dynamics.forward_dynamics_opt(0.1))

    def generate_control_input(self, system, global_path, local_trajectory, obstacles):
        self._optimizer.setup(self._param, system, local_trajectory, obstacles)
        self._opt_sol = self._optimizer.solve_nlp()
        return self._opt_sol.value(self._optimizer.variables["u"][:, 0])

    def logging(self, logger):
        logger._xtrajs.append(self._opt_sol.value(self._optimizer.variables["x"]).T)
        logger._utrajs.append(self._opt_sol.value(self._optimizer.variables["u"]).T)
