import numpy as np
np.seterr(all='raise')
from pyfiberamp.helper_funcs import *
from pyfiberamp.util import SlicedArray
from pyfiberamp.dynamic_simulation.dynamic_model import DynamicModel


CONVERGENCE_CHECKING_INTERVAL = 10
N2NT_EPSILON = 1e-30


class FiniteDifferenceSolver:
    def __init__(self, channels, fiber):
        self.channels = channels
        self.fiber = fiber
        self.model = DynamicModel(self.channels, self.fiber)
        self.solver = self.maccormack_mod
        self.verbose = True

    def grid_shape(self, z_nodes):
        return len(self.channels.get_propagation_directions()), z_nodes

    def multi_step_steady_state_simulation(self, z_nodes, max_times, propagation_speeds, tol):

        assert(len(z_nodes) == len(max_times) == len(propagation_speeds))
        nodes = 2
        P = np.ones(self.grid_shape(nodes)) * SIMULATION_MIN_POWER
        N2 = np.zeros(nodes)
        F, dN2dt = self.model.fiber_response_functions()

        for idx, (nodes, max_time, speed) in enumerate(zip(z_nodes, max_times, propagation_speeds)):
            P = resample_array(P, nodes)
            N2 = resample_array(N2[np.newaxis, :], nodes).flatten()
            P, N2, converged = self.simulate(P=P, N2=N2, F=F, dN2dt=dN2dt, max_time=max_time, propagation_speed=speed, excitation_tol=tol)
            if idx == 0 and not converged:
               print('First step did not converge properly! Add nodes and/or increase the max time.')
            tol = 1e-20

        return P, N2, self.z(nodes)

    def simulate(self, P, N2, F, dN2dt, max_time, propagation_speed, excitation_tol):
        return self.solver(P, N2, F, dN2dt, max_time, propagation_speed, excitation_tol)

    def maccormack_mod(self, P, N2, F, dN2dt, max_time, propagation_speed, excitation_tol):
        dx = self.fiber.length / (P.shape[1] - 1)
        dt = dx / propagation_speed # Courant condition

        u = self.channels.get_propagation_directions()[:, np.newaxis]
        fb_slices = self.channels.get_forward_and_backward_slices()
        forward_slice = fb_slices['forward']
        backward_slice = fb_slices['backward']

        input_powers = SlicedArray(self.channels.get_input_powers(), fb_slices)
        P[forward_slice, 0] = input_powers.forward
        P[backward_slice, -1] = input_powers.backward

        Pp = np.copy(P)
        N2p = np.copy(N2)
        dP = np.zeros_like(P)

        cn_dt = propagation_speed * dt * u
        cn_dt_2 = cn_dt / 2
        cn_dt_dx = cn_dt / dx
        cn_dt_dx_2 = cn_dt_dx / 2

        # iterate MacCormack method
        average_excitation = 0
        time = 0
        converged = False
        while time < max_time and not converged:
            # Step 1: forward differencing for forward propagation and backward differencing for backward propagation
            dif = np.diff(P)
            dP[forward_slice, :-1] = dif[forward_slice, :]
            dP[forward_slice, -1] = dif[forward_slice, -1]
            dP[backward_slice, 1:] = dif[backward_slice, :]
            dP[backward_slice, 0] = dif[backward_slice, 0]
            Pp[...] = P - cn_dt_dx * dP + cn_dt * F(P, N2)
            N2p[...] = N2 + dt * dN2dt(P, N2)

            # refresh boundaries and clamp to positive
            Pp[forward_slice, 0] = input_powers.forward
            Pp[backward_slice, -1] = input_powers.backward
            self.clamp(Pp)

            # Step 2: backward differencing for forward propagation and forward differencing for backward propagation
            dif = np.diff(Pp)
            dP[forward_slice, 1:] = dif[forward_slice, :]
            dP[forward_slice, 0] = dif[forward_slice, 0]
            dP[backward_slice, :-1] = dif[backward_slice, :]
            dP[backward_slice, -1] = dif[backward_slice, -1]
            P[...] = (P + Pp) / 2 - cn_dt_dx_2 * dP + cn_dt_2 * F(Pp, N2p)
            N2[...] = (N2 + N2p) / 2 + dt / 2 * dN2dt(Pp, N2p)

            # refresh boundaries and clamp to positive
            P[forward_slice, 0] = input_powers.forward
            P[backward_slice, -1] = input_powers.backward
            self.clamp(P)

            time += dt

            n_iteration = int(round(time/dt))
            if n_iteration % CONVERGENCE_CHECKING_INTERVAL == 0:
                average_excitation, converged = self.check_convergence(N2, average_excitation, n_iteration, excitation_tol, dt)

        return P, N2, converged

    def check_convergence(self, N2, average_excitation, n_iteration, tol, dt):
        new_average_excitation = np.mean(N2) / self.fiber.ion_number_density
        if self.verbose:
            print('Iteration: {:d}, {:.2f} us, average excitation: {:.4f}'.format(n_iteration, dt*n_iteration*1e6, new_average_excitation))
        if abs(new_average_excitation - average_excitation) / (average_excitation + N2NT_EPSILON) < tol:
            return new_average_excitation, True
        else:
            average_excitation = new_average_excitation
        return average_excitation, False

    @staticmethod
    def laplace_smoothing_factor(Pin):
        s_op = np.zeros_like(Pin)
        s_op[:,1:-1] = Pin[:, :-2] + Pin[:, 2:] - 2 * Pin[:, 1:-1]
        return s_op * 2 / np.pi

    def z(self, nodes):
        return np.linspace(0, self.fiber.length, nodes)

    @staticmethod
    def clamp(P):
        P[P < SIMULATION_MIN_POWER] = SIMULATION_MIN_POWER
        return P

    def simulate_maccormack(self):
        F, dN2dt = self.model.fiber_response_functions()
        P = self.intermediate_P
        dx = self.fiber.length / (self.z_nodes + 1)
        dt = dx / self.propagation_speed
        N_time = int(round(self.max_time / dt))

        u = self.channels.get_propagation_directions()[:, np.newaxis]
        fb_slices = self.channels.get_forward_and_backward_slices()
        forward_slice = fb_slices['forward']
        backward_slice = fb_slices['backward']

        input_powers = SlicedArray(self.channels.get_input_powers(), fb_slices)
        P[forward_slice, 0] = input_powers.forward
        P[backward_slice, -1] = input_powers.backward

        # first step array

        Pp = np.copy(P)
        N2 = self.intermediate_N2
        N2p = np.copy(N2)
        prev_averageN2nt = 1e-30
        dx_forward = SlicedArray(np.zeros_like(P), fb_slices)
        dx_backward = SlicedArray(np.zeros_like(dx_forward), fb_slices)
        speed_dt_dx = self.propagation_speed * dt / dx * u
        speed_dt = self.propagation_speed * dt * u
        speed_dt_2 = self.propagation_speed * dt / 2 * u
        speed_dt_dx_2 = self.propagation_speed * dt / (2 * dx) * u
        P_interpolate = np.copy(P)
        # iterate MacCormack method
        for n in range(0, N_time, 2):
            # FIRST TIME STEP: FORWARD THEN BACKWARD DIFFERENCING
            # Step 1
            dx_forward[:, :-1] = np.diff(P)
            dx_forward[:, -1] = dx_forward[:, -2]
            Pp[...] = P - speed_dt_dx * dx_forward + speed_dt * F(P, N2)
            N2p[...] = N2 + dt * dN2dt(P, N2)

            # refresh boundaries
            Pp[forward_slice, 0] = input_powers.forward
            Pp[backward_slice, -1] = input_powers.backward

            # clamp to min power

            self.clamp(Pp)
            # Step 2
            dx_backward[:, 1:] = np.diff(Pp)
            dx_backward[:, 0] = dx_backward[:, 1]
            P[...] = (P + Pp) / 2 - speed_dt_dx_2 * dx_backward + speed_dt_2 * F(Pp, N2p)
            N2[...] = (N2 + N2p) / 2 + dt / 2 * dN2dt(Pp, N2p)

            # refresh boundaries
            P[forward_slice, 0] = input_powers.forward
            P[backward_slice, -1] = input_powers.backward

            # clamp to min power
            self.clamp(P)
            """
            if n%1000 == 0:
                average_N2nt = np.mean(N2) / self.fiber.ion_number_density
                if abs(average_N2nt - prev_averageN2nt) / prev_averageN2nt < 1e-6:
                    break
                prev_averageN2nt = average_N2nt + 1e-30
                print('Iteration N: {:d}, average N2: {:.4f}'.format(n, average_N2nt))
                print(N2)
            """
        return P, N2

    def simulate_bfecc(self):
        F, dN2dt = self.model.fiber_response_functions()
        P = self.intermediate_P
        dx = self.fiber.length / (self.z_nodes + 1)
        dt = dx / self.propagation_speed
        N_time = int(round(self.max_time / dt))

        u = self.channels.get_propagation_directions()[:, np.newaxis]
        fb_slices = self.channels.get_forward_and_backward_slices()
        forward_slice = fb_slices['forward']
        backward_slice = fb_slices['backward']

        input_powers = SlicedArray(self.channels.get_input_powers(), fb_slices)
        P[forward_slice, 0] = input_powers.forward
        P[backward_slice, -1] = input_powers.backward

        # first step array

        P1 = np.copy(P)
        P0 = np.copy(P)
        N2 = self.intermediate_N2
        N2_1 = np.copy(N2)
        N2_0 = np.copy(N2)
        prev_averageN2nt = 1e-30
        dx_forward = SlicedArray(np.zeros_like(P), fb_slices)
        dx_backward = SlicedArray(np.zeros_like(dx_forward), fb_slices)
        speed_dt_dx = self.propagation_speed * dt / dx * u
        speed_dt = self.propagation_speed * dt * u
        speed_dt_2 = self.propagation_speed * dt / 2 * u
        speed_dt_dx_2 = self.propagation_speed * dt / (2 * dx) * u
        # iterate MacCormack method
        for n in range(0, N_time, 2):
            # Step 1: forward advection
            dx_backward[:, 1:] = np.diff(P)
            dx_backward[:, 0] = dx_backward[:, 1]
            dx_forward[:, :-1] = np.diff(P)
            dx_forward[:, -1] = dx_forward[:, -2]
            dx_backward[backward_slice,:] = dx_forward[backward_slice,:]
            P1[...] = P - speed_dt_dx * dx_backward + speed_dt * F(P, N2)
            N2_1[...] = N2 + dt * dN2dt(P, N2)

            # refresh boundaries
            P1[forward_slice, 0] = input_powers.forward
            P1[backward_slice, -1] = input_powers.backward

            # clamp to min power
            self.clamp(P1)
            # Step 2: reverse advection from estimate
            dx_forward[:, :-1] = np.diff(P1)
            dx_forward[:, -1] = dx_forward[:, -2]
            dx_backward[:, 1:] = np.diff(P1)
            dx_backward[:, 0] = dx_backward[:, 1]
            dx_forward[backward_slice, :] = dx_backward[backward_slice, :]
            P0[...] = P1 + speed_dt_dx * dx_forward - speed_dt * F(P1, N2_1)
            N2_0[...] = N2_1 - dt * dN2dt(P1, N2_1)

            # refresh boundaries
            P0[forward_slice, 0] = input_powers.forward
            P0[backward_slice, -1] = input_powers.backward

            # clamp to min power
            self.clamp(P0)
            # Step 3: error correction
            P[...] = P1 + (P - P0) / 2
            N2[...] = N2_1 + (N2 - N2_0) / 2

            # refresh boundaries
            P[forward_slice, 0] = input_powers.forward
            P[backward_slice, -1] = input_powers.backward

            # clamp to min power
            self.clamp(P)
            """
            if n%1000 == 0:
                average_N2nt = np.mean(N2) / self.fiber.ion_number_density
                if abs(average_N2nt - prev_averageN2nt) / prev_averageN2nt < 1e-6:
                    break
                prev_averageN2nt = average_N2nt + 1e-30
                print('Iteration N: {:d}, average N2: {:.4f}'.format(n, average_N2nt))
                print(N2)
            """
        return P, N2

    def simulate_bfecc_mod(self):
        F, dN2dt = self.model.fiber_response_functions()
        P = self.intermediate_P
        dx = self.fiber.length / (self.z_nodes + 1)
        dt = dx / self.propagation_speed
        N_time = int(round(self.max_time / dt))

        u = self.channels.get_propagation_directions()[:, np.newaxis]
        fb_slices = self.channels.get_forward_and_backward_slices()
        forward_slice = fb_slices['forward']
        backward_slice = fb_slices['backward']

        input_powers = SlicedArray(self.channels.get_input_powers(), fb_slices)
        P[forward_slice, 0] = input_powers.forward
        P[backward_slice, -1] = input_powers.backward

        # first step array

        P1 = np.copy(P)
        P0 = np.copy(P)
        N2 = self.intermediate_N2
        N2_1 = np.copy(N2)
        N2_0 = np.copy(N2)
        prev_averageN2nt = 1e-30
        dx_forward = SlicedArray(np.zeros_like(P), fb_slices)
        dx_backward = SlicedArray(np.zeros_like(dx_forward), fb_slices)
        speed_dt_dx = self.propagation_speed * dt / dx * u
        speed_dt = self.propagation_speed * dt * u
        speed_dt_2 = self.propagation_speed * dt / 2 * u
        speed_dt_dx_2 = self.propagation_speed * dt / (2 * dx) * u
        # iterate MacCormack method
        for n in range(0, N_time, 2):
            # Step 1: forward advection
            dx_backward[:, 1:] = np.diff(P)
            dx_backward[:, 0] = dx_backward[:, 1]
            dx_forward[:, :-1] = np.diff(P)
            dx_forward[:, -1] = dx_forward[:, -2]
            dx_backward[backward_slice,:] = dx_forward[backward_slice,:]
            P1[...] = P - speed_dt_dx * dx_backward + speed_dt * F(P, N2)
            N2_1[...] = N2 + dt * dN2dt(P, N2)

            # refresh boundaries
            P1[forward_slice, 0] = input_powers.forward
            P1[backward_slice, -1] = input_powers.backward

            # clamp to min power
            P1[P1 < SIMULATION_MIN_POWER] = SIMULATION_MIN_POWER

            # Step 2: reverse advection from estimate
            dx_forward[:, :-1] = np.diff(P1)
            dx_forward[:, -1] = dx_forward[:, -2]
            dx_backward[:, 1:] = np.diff(P1)
            dx_backward[:, 0] = dx_backward[:, 1]
            dx_forward[backward_slice, :] = dx_backward[backward_slice, :]
            P0[...] = P1 + speed_dt_dx * dx_forward - speed_dt * F(P1, N2_1)
            N2_0[...] = N2_1 - dt * dN2dt(P1, N2_1)

            # refresh boundaries
            P0[forward_slice, 0] = input_powers.forward
            P0[backward_slice, -1] = input_powers.backward

            # clamp to min power
            P0[P0 < SIMULATION_MIN_POWER] = SIMULATION_MIN_POWER

            # Step 3: error correction
            P[...] = P1 + (P - P0) / 2
            N2[...] = N2_1 + (N2 - N2_0) / 2

            # refresh boundaries
            P[forward_slice, 0] = input_powers.forward
            P[backward_slice, -1] = input_powers.backward

            # clamp to min power
            P[P < SIMULATION_MIN_POWER] = SIMULATION_MIN_POWER  # P_interpolate[P < SIMULATION_MIN_POWER]#SIMULATION_MIN_POWER

            """
            if n%1000 == 0:
                average_N2nt = np.mean(N2) / self.fiber.ion_number_density
                if abs(average_N2nt - prev_averageN2nt) / prev_averageN2nt < 1e-6:
                    break
                prev_averageN2nt = average_N2nt + 1e-30
                print('Iteration N: {:d}, average N2: {:.4f}'.format(n, average_N2nt))
                print(N2)
            """
        return P, N2

