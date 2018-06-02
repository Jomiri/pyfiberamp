

class DynamicBoundaryConditions:
    def __init__(self, P_in_out, reflections, n_forward):
        self.P_in_out = P_in_out
        self.reflections = reflections
        self.n_forward = n_forward

    def apply_input(self, P, idx):
        P[:self.n_forward, 0] = self.P_in_out[:self.n_forward, idx]
        P[self.n_forward:, -1] = self.P_in_out[self.n_forward:, idx]

    def apply_output(self, P, idx):
        self.P_in_out[:self.n_forward, idx] = P[:self.n_forward, -1]
        self.P_in_out[self.n_forward:, idx] = P[self.n_forward:, 0]

    def apply_reflection(self, P):
        for r in self.reflections:
            source_idx, target_idx, R = r
            if source_idx < self.n_forward:
                P[target_idx, -1] += R * P[source_idx, -2]
            else:
                P[target_idx, 0] += R * P[source_idx, 1]

    def correct_output_by_reflection(self):
        for r in self.reflections:
            source_idx, target_idx, R = r
            T = 1 - R
            self.P_in_out[source_idx, :] *= T