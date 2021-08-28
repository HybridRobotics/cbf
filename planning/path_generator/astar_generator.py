import numpy as np


class AstarPathGenerator:
    # TODO: this class shall be renamed as replace by a search-based planner
    def __init__(self):
        self._global_path = None

    def generate_path(self):
        # TODO: Remove hardcoded global path
        self._global_path = np.array([[0.0, 0.2], [0.5, 0.2], [0.5, 0.8], [1.0, 0.8]])
        return self._global_path

    def logging(self, logger):
        logger.append(self._global_path)
