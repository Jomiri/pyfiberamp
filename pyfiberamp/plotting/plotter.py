import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from pyfiberamp.fibers import PassiveFiber
from pyfiberamp.helper_funcs import *

default_colors = [
    '#000000',
    '#C10020',
    '#00538A',
    '#007D34',
    '#FFB300',
    '#803E75',
    '#FF6800',
    '#A6BDD7',
    '#CEA262',
    '#817066',
    '#F6768E'
]

default_style = {
        "axes.prop_cycle": cycler(color=default_colors),
        "font.family": "sans-serif",
        "axes.labelsize": 14,
        "font.size": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.top": True,
        "ytick.left": True
}


# Steady state power evolution #
def plot_simulation_result(simulation_result, figsize=DEFAULT_FIGSIZE):
    plot_power_evolution(simulation_result, figsize)
    plot_ase_spectra(simulation_result, figsize)
    plt.show()


def plot_power_evolution(simulation_result, figsize=DEFAULT_FIGSIZE):
    with mpl.rc_context(default_style):
        fig, ax = plt.subplots(figsize=figsize)
        for idx, ch in enumerate(simulation_result.channels.as_iterator()):
            ch_power = simulation_result.powers[idx, :]
            if ch.channel_type != 'ase':
                plot_single_non_ase_channel(ax, simulation_result.z, ch, ch_power, simulation_result.use_db_scale)
        plot_ase_evolution(ax,
                           simulation_result.z,
                           simulation_result.channels,
                           simulation_result.powers,
                           simulation_result.use_db_scale)
        finalize_fiber_power_plot(ax,
                                  simulation_result.z,
                                  simulation_result.use_db_scale)

        ax_right = ax.twinx()
        plot_excited_ion_fraction(ax_right,
                                  simulation_result.z,
                                  simulation_result.fiber,
                                  simulation_result.local_average_excitation)
        make_power_evolution_legend(ax, ax_right)
        plt.show()


def plot_single_non_ase_channel(ax, z, channel, power, use_db_scale):
    wl = channel.wavelength
    if channel.direction == 1:
        start_idx, end_idx = (0, -1)
        dir_label = 'forward'
    else:
        start_idx, end_idx = (-1, 0)
        dir_label = 'backward'
    input_power = power[start_idx]
    output_power = power[end_idx]
    gain_or_absorption = to_db(output_power / input_power)
    ax.plot(z, plotting_transformation(power, use_db_scale),
            label=make_legend_entry(f'{dir_label} {channel.channel_type}',
                                    channel.channel_id,
                                    wl,
                                    gain_or_absorption,
                                    output_power,
                                    use_db_scale))


def plot_ase_evolution(ax, z, channels, powers, use_db_scale):
    if len(channels.forward_ase) != 0:
        forward_ase = np.sum(powers.forward_ase, axis=0)
        forward_ase_power = forward_ase[-1]
        ax.plot(z, plotting_transformation(forward_ase, use_db_scale),
                label='Forward ASE, input_power={:.2f} mW'.format(forward_ase_power * 1000))

    if len(channels.backward_ase) != 0:
        backward_ase = np.sum(powers.backward_ase, axis=0)
        backward_ase_power = backward_ase[0]
        ax.plot(z, plotting_transformation(backward_ase, use_db_scale),
                    label='Backward ASE, input_power={:.2f} mW'.format(backward_ase_power * 1000))


def make_legend_entry(channel_type, channel_id, wl, gain_or_absorption, output_power, use_db_scale):
    ch_type = channel_type.replace('-', ' ').title()
    channel_id_label = f' {channel_id}' if isinstance(channel_id, str) else ''
    wl_nm = wl*1e9
    label_start = f'{ch_type}{channel_id_label}, {wl_nm:.1f} nm, '
    power_label = make_power_label(output_power, use_db_scale)
    db_label = make_db_label(gain_or_absorption, channel_type)
    return label_start + power_label + db_label


def make_power_label(output_power, use_db_scale):
    if use_db_scale:
        power, unit = (to_dbm(output_power), 'dBm')
    else:
        use_watts = output_power > 1
        power, unit = (output_power, 'W') if use_watts else (output_power * 1000, 'mW')
    return 'output={power:.1f} {unit}, '.format(power=power, unit=unit)


def make_db_label(gain_or_absorption, channel_type):
    include_db = 'pump' in channel_type or 'signal' in channel_type
    if not include_db:
        return ''
    gain_label = 'gain' if gain_or_absorption > 0 else 'absorption'
    return '{g_or_a}={db_value:.1f} dB'.format(g_or_a=gain_label, db_value=abs(gain_or_absorption))


def make_power_evolution_legend(ax, ax_right):
    ion_line, ion_label = ax_right.get_legend_handles_labels()
    power_lines, power_labels = ax.get_legend_handles_labels()
    handles = ion_line + power_lines
    labels = ion_label + power_labels
    if handles:
        ax.legend(handles, labels).get_frame().set_edgecolor('black')


def finalize_fiber_power_plot(ax, z, use_db_scale):
    x_label = 'Position inside fiber [m]'
    y_label = 'Power [{unit}]'.format(unit='dBm' if use_db_scale else 'W')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    xlim = [z[0], z[-1]]
    ax.set_xlim(xlim)
    if not use_db_scale:
        ax.set_ylim([0, ax.get_ylim()[1]])
    ax.grid(True)


def plot_excited_ion_fraction(ax, z, fiber, local_average_excitation):
    if isinstance(fiber, PassiveFiber):
        ax.set_axis_off()
        return
    weights = np.zeros_like(z)
    weights[:-1] = np.diff(z)
    weights[-1] = z[-1] - z[-2]
    ax.plot(z, local_average_excitation * 100, '--',
            label=f'Excited ion fraction, average={np.average(local_average_excitation, weights=weights)*100:.1f} %',
            color='cyan')
    ax.set_ylabel('Ions at the upper laser level [%]')
    ax.yaxis.set_ticks_position('right')
    ax.set_xlim([z[0], z[-1]])
    ax.set_ylim([0, 100])


def plot_ase_spectra(simulation_result, figsize=DEFAULT_FIGSIZE):
    with mpl.rc_context(default_style):
        if len(simulation_result.wavelengths.forward_ase) == 0:
            return
        fig, ax = plt.subplots(figsize=figsize)
        forward_ase_spectrum = simulation_result.powers.forward_ase[:, -1]
        backward_ase_spectrum = simulation_result.powers.backward_ase[:, 0]
        forward_ase_wls_nm = simulation_result.wavelengths.forward_ase * 1e9
        forward_ase_wl_step = np.squeeze(np.array([ch.wavelength_bandwidth
                                                   for ch in simulation_result.channels.forward_ase]))
        backward_ase_wls_nm = simulation_result.wavelengths.backward_ase * 1e9
        backward_ase_wl_step = np.squeeze(np.array([ch.wavelength_bandwidth
                                                    for ch in simulation_result.channels.backward_ase]))
        ax.plot(forward_ase_wls_nm, to_dbm(forward_ase_spectrum / forward_ase_wl_step), label='Forward ASE')
        ax.plot(backward_ase_wls_nm, to_dbm(backward_ase_spectrum / backward_ase_wl_step), label='Backward ASE')
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel('Power spectral density [dBm/nm]')
        ax.set_xlim([min(forward_ase_wls_nm[0], backward_ase_wls_nm[0]),
                     max(forward_ase_wls_nm[-1], backward_ase_wls_nm[-1])])
        ax.legend().get_frame().set_edgecolor('black')
        ax.grid(True)
        plt.show()


def plot_signal_intensity(self, effective_area):
    intensity = self.powers.forward_signal[0, :] / effective_area
    plt.plot(self.z, intensity)
    plt.show()


def plot_total_power(self):
    fig, ax = plt.subplots()
    total_power = np.sum(self.powers, axis=0)
    ax.plot(self.z, total_power, label='Total power in fiber')
    ax.set_xlabel('Z [m]', fontsize=18)
    ax.set_ylabel('Total power in fiber [W]')
    ax.set_ylim([0, ax.get_ylim()[1]])


def plotting_transformation(power, use_db_scale):
    return to_db(power * 1000) if use_db_scale else power





# Dynamic outputs
def plot_dynamic_outputs(simulation_result, channel_ids=None, plot_density=1, figsize=DEFAULT_FIGSIZE):
    with mpl.rc_context(default_style):
        fig, ax = plt.subplots(figsize=figsize)
        for idx, ch in enumerate(simulation_result.channels.as_iterator()):
            if channel_ids is None or ch.channel_id in channel_ids:
                plot_single_output(ax, ch, simulation_result.t,
                                   simulation_result.output_powers[idx, :],
                                   plot_density,
                                   simulation_result.use_db_scale,
                                   ((channel_ids is not None and ch.channel_id in channel_ids)
                                    or isinstance(ch.channel_id, str)))
        ax.set_xlabel('Time [μs]')
        ax.set_ylabel('Power [{unit}]'.format(unit='W' if not simulation_result.use_db_scale else 'dBm'))
        ax.set_xlim(np.array([simulation_result.t[0], simulation_result.t[-1]]) * 1e6)
        ax.legend().get_frame().set_edgecolor('black')
        ax.grid(True)
        plt.show()


def plot_single_output(ax, ch, t, power, plot_density, use_db_scale, id_listed):
    ax.plot(t[::plot_density]*1e6, plotting_transformation(power, use_db_scale)[::plot_density],
            label=make_output_legend_entry(ch, id_listed))


def make_output_legend_entry(channel, id_listed=False):
    channel_type_str = ('forward' if channel.direction == 1 else 'backward') + ' ' + channel.channel_type
    optional_label = ' (' + str(channel.channel_id) + ')' if id_listed else ''
    wavelength = channel.wavelength * 1e9
    return '{channel_type}{label}, {wl:.1f} nm'.format(channel_type=channel_type_str.title(),
                                                       label=optional_label,
                                                       wl=wavelength)


def plot_transverse_ion_excitation(self):
    assert self.fiber.doping_profile.radii is not None, 'Doping profile radii not defined.'
    assert self.fiber.num_ion_populations > 1, 'More than one ion population needed for transverse plotting.'
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    radii = self.fiber.doping_profile.radii
    n2 = self.upper_level_fraction
    n2_plot = np.vstack((n2[::-1, :], n2))
    max_r = radii[-1]
    r_plot = np.linspace(-max_r, max_r, len(radii) * 2)[:, np.newaxis] * np.ones(n2_plot.shape[1])[np.newaxis, :]
    l_plot = self.z[np.newaxis, :] * np.ones(n2_plot.shape[0])[:, np.newaxis]

    # plt.imshow(n2_plot, extent=(-max_r, max_r, 0, self.fiber.length))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(l_plot, r_plot * 1e6, n2_plot, cmap=cm.coolwarm)
    ax.set_xlabel('Distance (m)', fontsize=18)
    ax.set_ylabel('Radius (um)', fontsize=18)
    ax.set_zlabel('Fractional excitation', fontsize=18)
    plt.show()