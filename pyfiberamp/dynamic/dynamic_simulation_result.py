import matplotlib.pyplot as plt

from pyfiberamp.simulation_result import SimulationResult


class DynamicSimulationResult(SimulationResult):
    def __init__(self, z, t, powers, upper_level_fraction, output_powers, channels, fiber):
        super().__init__(z, powers, upper_level_fraction, channels, fiber, backward_raman_allowed=False)
        self.output_powers = output_powers
        self.t = t
        self.t_us = t * 1e6

    def plot_outputs(self, labels=None, plot_density=1):
        fig, ax = plt.subplots()
        for idx, ch in enumerate(self.channels._all_channels()):
            if labels is not None and ch.label not in labels:
                continue
            self.plot_single_output(ax, ch, idx, plot_density)
        self.finalize_output_power_plot(ax)
        self.make_output_power_legend(ax)
        plt.show()

    def plot_single_output(self, ax, ch, idx, plot_density):
        output = self.output_powers[idx, :]
        ax.plot(self.t_us[::plot_density], self.plotting_transformation(output)[::plot_density],
                label=self.make_output_legend_entry(ch))

    def finalize_output_power_plot(self, ax):
        xlabel = 'Time (microseconds)'
        ylabel = 'Power ({unit})'.format(unit=self.power_evolution_unit())
        xlim = [self.t_us[0], self.t_us[-1]]
        self.finalize_power_plot(ax, xlabel, ylabel, xlim)

    def make_output_power_legend(self, ax):
        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels)

    def make_output_legend_entry(self, channel):
        channel_type = self.channel_type_to_title(channel.channel_type)
        optional_label = ' (' + channel.label + ')' if channel.label != '' else ''
        wavelength = channel.wavelength * 1e9
        return '{channel_type}{label}, {wl:.1f} nm'.format(channel_type=channel_type,
                                                        label=optional_label,
                                                        wl=wavelength)

