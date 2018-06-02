from pyfiberamp.dynamic.dynamic_solver_base import DynamicSolverBase
from pyfiberamp.dynamic.fiber_simulation_pybindings import simulate


class DynamicSolverCpp(DynamicSolverBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.solver_func = simulate

    def solve(self, *args):
        return self.solver_func(*args)
