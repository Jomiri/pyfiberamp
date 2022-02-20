import numpy as np


class ConvergenceChecker:
    EPS = 1e-30
    #CONVERGENCE_CHECKING_INTERVAL = 10000

    def __init__(self, checking_interval, max_iterations, tol, stop_at_steady_state, weights):
        self.convergence_checking_interval = checking_interval
        self.max_iterations = max_iterations
        self.tol = tol
        self.stop_at_steady_state = stop_at_steady_state
        self.prev_mean_N2 = 0
        self.current_mean_N2 = 0
        self.weights = weights[:, np.newaxis] / np.sum(weights) * len(weights)

    def has_not_converged(self, N2, n_iteration):
        if n_iteration == 0:
            return True

        if n_iteration == self.max_iterations:
            self.update_mean_N2(N2)
            self.print_status(n_iteration)
            return False

        if n_iteration % self.convergence_checking_interval != 0:
            return True

        self.update_mean_N2(N2)
        self.print_status(n_iteration)

        if not self.stop_at_steady_state:
            return True

        return not self.excitation_change_under_tolerance()

    def update_mean_N2(self, N2):
        self.prev_mean_N2 = self.current_mean_N2
        self.current_mean_N2 = np.mean(N2[:, 1:-1] * self.weights)  # Boundary points with N2=0 excluded

    def print_status(self, n_iteration):
        print("Iteration {:d}, average excitation: {:.5E}".format(n_iteration, self.current_mean_N2))

    def excitation_change_under_tolerance(self):
        return abs(self.current_mean_N2 - self.prev_mean_N2) / (self.prev_mean_N2 + self.EPS) < self.tol