import os


# Physical quantities or constants
DEFAULT_RAMAN_GAIN = 1e-13
RAMAN_FREQ_SHIFT = 13.2e12
ASE_OUTPUT_POWER_GUESS = 1e-3
RAMAN_GAIN_WL_BANDWIDTH = 5e-9
REP_RATE_LOWER_LIMIT = 1e4  # Hz
NUMBER_OF_ASE_POLARIZATION_MODES = 2  # Two polarization modes
RAMAN_MODES_IN_PM_FIBER = 1
YB_UPPER_STATE_LIFETIME = 1e-3  # s
c = 299_792_458
h = 6.62607e-34
DEFAULT_GROUP_INDEX = 1.45

# Constants for the numerical algorithm
SIMULATION_MIN_POWER = 1e-80
SOLVER_MAX_NODES = 20_000
START_NODES = 20


# Constants for active fiber
CROSS_SECTION_SMOOTHING_FACTOR = 1e-51
SPECTRUM_PLOT_NPOINTS = 1000

default_fig_width = 12
golden_ratio = (5**0.5 - 1) / 2
DEFAULT_FIGSIZE = (default_fig_width, golden_ratio*default_fig_width)


# Default absorption and emission cross section files
this_folder = os.path.dirname(os.path.realpath(__file__))
spectrum_folder = os.path.join(this_folder, 'spectroscopies', 'fiber_spectra')
YB_ABSORPTION_CS_FILE = os.path.join(spectrum_folder, 'ytterbium absorption cross sections.dat')
YB_EMISSION_CS_FILE = os.path.join(spectrum_folder, 'ytterbium emission cross sections.dat')
