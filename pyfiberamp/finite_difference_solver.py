from pyfiberamp.helper_funcs import *
from pyfiberamp.util import SlicedArray


class FiniteDifferenceSolver:
    def __init__(self):
        self.total_time = None
        self.z_nodes = START_NODES
        self.propagation_speed = c / 100000

    def simulate(self, channels, fiber, F, dN2dt):
        total_time = 10 * fiber.spectroscopy.upper_state_lifetime
        dx = fiber.length / (self.z_nodes + 1)
        dt = dx / self.propagation_speed
        N_time = int(round(total_time / dt))

        u = channels.get_propagation_directions()[:, np.newaxis]
        grid_shape = (len(u), self.z_nodes + 2)
        fb_slices = channels.get_forward_and_backward_slices()
        P = SlicedArray(np.zeros(grid_shape), fb_slices)
        forward_slice = fb_slices['forward']
        backward_slice = fb_slices['backward']

        input_powers = SlicedArray(channels.get_input_powers(), fb_slices)
        P[forward_slice, 0] = input_powers.forward
        P[backward_slice, -1] = input_powers.backward

        # first step array

        Pp = np.copy(P)
        N2 = np.zeros(self.z_nodes + 2)
        N2p = np.copy(N2)
        prev_averageN2nt = 1e-30
        dux_forward = SlicedArray(np.zeros(grid_shape), fb_slices)
        dupx_backward = SlicedArray(np.zeros_like(dux_forward), fb_slices)
        speed_dt_dx = self.propagation_speed * dt / dx * u
        speed_dt = self.propagation_speed * dt * u
        speed_dt_2 = self.propagation_speed * dt / 2 * u
        speed_dt_dx_2 = self.propagation_speed * dt / (2 * dx) * u

        # iterate MacCormack method
        for n in range(N_time):
            # Step 1
            dux_forward[:, :-1] = np.diff(P)
            dux_forward[:, -1] = dux_forward[:, -2]
            Pp[...] = P - speed_dt_dx * dux_forward + speed_dt * F(P, N2)
            N2p[...] = N2 + dt * dN2dt(P, N2)

            # refresh boundaries
            Pp[forward_slice, 0] = input_powers.forward
            Pp[backward_slice, -1] = input_powers.backward

            # clamp to min power
            Pp[Pp < SIMULATION_MIN_POWER] = SIMULATION_MIN_POWER
            N2p[N2p < SIMULATION_MIN_POWER] = SIMULATION_MIN_POWER

            # Step 2
            dupx_backward[:, 1:] = np.diff(Pp)
            dupx_backward[:, 0] = dupx_backward[:, 1]
            P[...] = (P + Pp) / 2 - speed_dt_dx_2 * dupx_backward + speed_dt_2 * F(Pp, N2p)
            N2[...] = (N2 + N2p) / 2 + dt / 2 * dN2dt(Pp, N2p)

            # refresh boundaries
            P[forward_slice, 0] = input_powers.forward
            P[backward_slice, -1] = input_powers.backward

            # clamp to min power
            P[P < SIMULATION_MIN_POWER] = SIMULATION_MIN_POWER
            N2[N2 < SIMULATION_MIN_POWER] = SIMULATION_MIN_POWER

            """
            if n%1000 == 0:
                average_N2nt = np.mean(N2) / Nt
                if abs(average_N2nt - prev_averageN2nt) / prev_averageN2nt < 1e-6:
                    break
                prev_averageN2nt = average_N2nt + 1e-30
                print('Iteration N: {:d}, average N2: {:.4f}'.format(n, average_N2nt))
                print(N2)
            """
        return P, N2