
import numpy as np
from pyfiberamp.helper_funcs import *
from pyfiberamp.util import SlicedArray


class DynamicModel:

    def __init__(self, channels, fiber):
        """The model is initialized with the optical channels and the simulated fiber. The parameters are named in the
        same way as in the paper by Giles and Desurvire."""
        self.v = np.array(channels.get_frequencies())[:, np.newaxis]
        self.dv = np.array(channels.get_frequency_bandwidths())[:, np.newaxis]
        self.u = np.array(channels.get_propagation_directions())[:, np.newaxis]
        self.m = np.array(channels.get_number_of_modes())[:, np.newaxis]
        self.a = np.array(channels.get_absorption())[:, np.newaxis]
        self.g = np.array(channels.get_gain())[:, np.newaxis]
        self.loss = np.array(channels.get_background_loss())[:, np.newaxis]
        self.fiber = fiber
        self.channels = channels

    def simulate_steady_state(self):
        # First, we precalculate all the constants
        tau = self.fiber.spectroscopy.upper_state_lifetime
        h_v_dv = h * self.v * self.dv
        g_m_h_v_dv = self.g * self.m * h_v_dv
        a_g = self.a + self.g
        a_l = self.a + self.loss
        a = self.a
        u = self.u
        Nt = self.fiber.ion_number_density
        A = 1 / tau
        h_v_pi_r2_inv = 1 / (h * self.v * np.pi * self.fiber.core_radius**2)


        total_time = 10 * tau
        dx = 0.05
        speed = c / 10000 #1.5
        dt = dx / speed
        N_time = int(round(total_time / dt))
        N_spatial = int(round(self.fiber.length / dx)) - 1


        def F(P, n2):
            return u * ((a_g * n2/Nt - a_l) * P + g_m_h_v_dv * n2 / Nt)


        def dN2dt(P, n2):
            return np.sum(P * h_v_pi_r2_inv * (a - a_g * n2 / Nt), axis=0) - n2 * A

        grid_shape = (len(self.g), N_spatial + 2)
        fb_slices = self.channels.get_forward_and_backward_slices()
        P = SlicedArray(np.zeros(grid_shape), fb_slices)
        forward_slice = fb_slices['forward']
        backward_slice = fb_slices['backward']
        # set boundary condition forward
        input_powers = SlicedArray(self.channels.get_input_powers(), fb_slices)
        P[forward_slice, 0] = input_powers.forward
        P[backward_slice, -1] = input_powers.backward

        # first step array

        Pp = np.copy(P)
        N2 = np.zeros(N_spatial + 2)
        N2p = np.copy(N2)
        prev_averageN2nt = 1e-30
        dux_forward = SlicedArray(np.zeros(grid_shape), fb_slices)
        dupx_backward = SlicedArray(np.zeros_like(dux_forward), fb_slices)
        speed_dt_dx = speed * dt / dx * u
        speed_dt = speed * dt * u
        speed_dt_2 = speed * dt / 2 * u
        speed_dt_dx_2 = speed * dt / (2 * dx) * u

        # iterate MacCormack
        for n in range(N_time):

            # Step 1
            dux_forward[:, :-1] = np.diff(P)
            dux_forward[:, -1] = dux_forward[:,-2]
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
        z = np.linspace(0, self.fiber.length, N_spatial+2)
        return z, P, N2
