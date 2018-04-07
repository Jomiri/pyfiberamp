import matplotlib.pyplot as plt

from .helper_funcs import *


class SimulationResult:
    def __init__(self, sol):
        self.sol = sol
        self.z = self.sol.x
        self.P = self.sol.y
        self.db_scale = False
        self.slices = {}
        self.wavelengths = None
        self.backward_raman_allowed = True
        self.upper_level_fraction = None
        self.is_passive_fiber = False

    @property
    def forward_signal_wls(self):
        return self.wavelengths[self.slices['forward_signal_slice']]

    @property
    def backward_signal_wls(self):
        return self.wavelengths[self.slices['backward_signal_slice']]

    @property
    def forward_pump_wls(self):
        return self.wavelengths[self.slices['forward_pump_slice']]

    @property
    def backward_pump_wls(self):
        return self.wavelengths[self.slices['backward_pump_slice']]

    @property
    def ase_wls(self):
        return self.wavelengths[self.slices['forward_ase_slice']]

    @property
    def raman_wls(self):
        return self.wavelengths[self.slices['forward_raman_slice']]

    @property
    def forward_signals(self):
        return self.P[self.slices['forward_signal_slice']]

    @property
    def backward_signals(self):
        return self.P[self.slices['backward_signal_slice']]

    @property
    def forward_pumps(self):
        return self.P[self.slices['forward_pump_slice']]

    @property
    def backward_pumps(self):
        return self.P[self.slices['backward_pump_slice']]

    @property
    def forward_ramans(self):
        return self.P[self.slices['forward_raman_slice']]

    @property
    def backward_ramans(self):
        return self.P[self.slices['backward_raman_slice']]

    @property
    def forward_ase(self):
        return self.P[self.slices['forward_ase_slice']]

    @property
    def backward_ase(self):
        return self.P[self.slices['backward_ase_slice']]

    def success(self):
        return self.sol.success

    @property
    def average_excitation(self):
        return np.mean(self.upper_level_fraction)

    def make_result_dict(self):
        keys = ['forward_signal', 'backward_signal',
                'forward_pump', 'backward_pump',
                'forward_ase', 'backward_ase',
                'forward_raman', 'backward_raman']
        slice_names = [key + '_slice' for key in keys]
        result_dict = {}
        for key, slice_name in zip(keys, slice_names):
            slice = self.slices[slice_name]
            power = self.P[slice]
            if len(power) == 0:
                continue
            start_idx, end_idx = 0, -1
            if 'backward' in 'key':
                start_idx, end_idx = end_idx, start_idx
            output_powers = power[:, end_idx]
            input_powers = power[:, start_idx]
            gain = to_db(output_powers / input_powers)
            result_dict[key] = {'input_powers': input_powers,
                                'output_powers': output_powers,
                                'gain': gain}
        return result_dict

    def forward_signal_gains(self):
        signal = self.forward_signals
        gain = signal[:, -1] / signal[:, 0]
        return to_db(gain)

    def forward_pump_absorptions(self):
        co_pump = self.forward_pumps
        absorption = co_pump[:, -1] / co_pump[:, 0]
        return -to_db(absorption)

    def plot_amplifier_result(self):
        self.plot_power_evolution()
        self.plot_ase_spectrum()
        # self.plot_total_power()
        plt.show()

    def plot_power_evolution(self):
        fig, ax = plt.subplots()
        self.plot_forward_signal_evolution(ax)
        self.plot_forward_pump_evolution(ax)
        self.plot_backward_pump_evolution(ax)
        self.plot_forward_raman_evolution(ax)
        self.plot_backward_raman_evolution(ax)
        self.plot_ase_evolution(ax)
        self.finalize_power_plot(ax)

        ax_right = ax.twinx()
        self.plot_excited_ion_fraction(ax_right)
        self.make_power_evolution_legend(ax, ax_right)
        plt.show()

    def plot_ase_spectrum(self):
        if len(self.ase_wls) == 0:
            return
        fig, ax = plt.subplots()
        forward_ase_spectrum = self.forward_ase[:, -1]
        backward_ase_spectrum = self.backward_ase[:, 0]
        ase_wls_nm = self.ase_wls * 1e9
        ase_wl_step = ase_wls_nm[1] - ase_wls_nm[0]
        ax.plot(ase_wls_nm, to_db(forward_ase_spectrum * 1000 / ase_wl_step), label='Forward ASE')
        ax.plot(ase_wls_nm, to_db(backward_ase_spectrum * 1000 / ase_wl_step), label='Backward ASE')
        ax.set_xlabel('Wavelength (nm)', fontsize=18)
        ax.set_ylabel('Power spectral density (dBm/nm)', fontsize=18)
        ax.set_xlim([ase_wls_nm[0], ase_wls_nm[-1]])
        ax.tick_params(which='major', direction='in', left=True, right=True, top=True, bottom=True,
                       width=3, length=10, labelsize=16)
        ax.legend()
        plt.show()

    def plot_forward_signal_evolution(self, ax):
        for i in range(len(self.forward_signal_wls)):
            signal = self.forward_signals[i, :]
            gain = signal[-1] / signal[0]
            gain_db = to_db(gain)
            ax.plot(self.z, self.plotting_transformation(signal),
                    label='Signal {:.2f} nm, gain={:.1f} dB'.format(self.forward_signal_wls[i] * 1e9, gain_db))

    def plot_forward_raman_evolution(self, ax):
        for i in range(len(self.raman_wls)):
            forward_raman = self.forward_ramans[i, :]
            power = forward_raman[-1]
            ax.plot(self.z, self.plotting_transformation(forward_raman),
                    label='Forward Raman {:.2f} nm, power={:.3f} W'.format(self.raman_wls[i] * 1e9, power))

    def plot_backward_raman_evolution(self, ax):
        if self.backward_raman_allowed:
            backward_ramans = self.backward_ramans
            for i in range(len(self.raman_wls)):
                backward_raman = backward_ramans[i, :]
                power = backward_raman[0]
                ax.plot(self.z, self.plotting_transformation(backward_raman),
                        label='Backward Raman {:.2f} nm, power={:.3f} W'.format(self.raman_wls[i] * 1e9, power))

    def plot_forward_pump_evolution(self, ax):
        for i in range(len(self.forward_pump_wls)):
            co_pump = self.forward_pumps[i, :]
            absorption = co_pump[-1] / co_pump[0]
            absorption_db = -to_db(absorption)
            ax.plot(self.z, self.plotting_transformation(co_pump),
                    label='Forward pump {:.2f} nm, absorption={:.1f} dB'.format(self.forward_pump_wls[i] * 1e9,
                                                                           absorption_db))

    def plot_backward_pump_evolution(self, ax):
        for i in range(len(self.backward_pump_wls)):
            counter_pump = self.backward_pumps[i, :]
            absorption = counter_pump[0] / counter_pump[-1]
            absorption_db = -to_db(absorption)
            ax.plot(self.z, self.plotting_transformation(counter_pump),
                    label='Backward pump {:.2f} nm, absorption={:.1f} dB'.format(self.backward_pump_wls[i] * 1e9,
                                                                                absorption_db))

    def plot_ase_evolution(self, ax):
        if len(self.ase_wls) != 0:
            forward_ase = np.sum(self.forward_ase, axis=0)
            forward_ase_power = forward_ase[-1]
            ax.plot(self.z, self.plotting_transformation(forward_ase),
                    label='Forward ASE, power={:.2f} mW'.format(forward_ase_power * 1000))

            backward_ase = np.sum(self.backward_ase, axis=0)
            backward_ase_power = backward_ase[0]
            ax.plot(self.z, self.plotting_transformation(backward_ase),
                    label='Backward ASE, power={:.2f} mW'.format(backward_ase_power * 1000))

    @staticmethod
    def make_power_evolution_legend(ax, ax_right):
        ion_line, ion_label = ax_right.get_legend_handles_labels()
        power_lines, power_labels = ax.get_legend_handles_labels()
        ax.legend(power_lines + ion_line, power_labels + ion_label)

    def finalize_power_plot(self, ax):
        ax.tick_params(which='minor', direction='in', left=True, right=False, top=True, bottom=False,
                       width=2, length=6)
        ax.tick_params(which='major', direction='in', left=True, right=False, top=True, bottom=True,
                       width=3, length=10, labelsize=16)

        ax.set_xlabel('Z (m)', fontsize=18)
        ax.set_ylabel('Power ({unit})'.format(unit=self.power_evolution_unit()), fontsize=18)
        ax.set_xlim([self.z[0], self.z[-1]])
        if not self.db_scale:
            ax.set_ylim([0, ax.get_ylim()[1]])
        ax.grid(True)

    def plot_excited_ion_fraction(self, ax):
        if self.is_passive_fiber:
            ax.set_axis_off()
            return
        ax.plot(self.z, self.upper_level_fraction * 100, '--', label='Excited ion fraction')
        ax.set_ylabel('Ions at the upper laser level (%)', fontsize=18)
        ax.yaxis.set_ticks_position('right')
        ax.tick_params(which='major', direction='in', left=False, right=True, top=True, bottom=True,
                       width=3, length=10, labelsize=16)
        ax.set_xlim([self.z[0], self.z[-1]])
        ax.set_ylim([0, 100])

    def plot_signal_intensity(self, effective_area):
        intensity = self.forward_signals[0, :] / effective_area
        plt.plot(self.z, intensity)
        plt.show()

    def plot_total_power(self):
        fig, ax = plt.subplots()
        total_power = np.sum(self.P, axis=0)
        ax.plot(self.z, total_power, label='Total power in fiber')
        ax.set_xlabel('Z (m)', fontsize=18)
        ax.set_ylabel('Total power in fiber (W)', fontsize=18)
        ax.set_ylim([0, ax.get_ylim()[1]])

    def power_evolution_unit(self):
        return 'dBm' if self.db_scale else 'W'

    def plotting_transformation(self, power):
        return to_db(power * 1000) if self.db_scale else power
