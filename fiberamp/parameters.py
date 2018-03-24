import os


# Physical quantities or constants
G_RAMAN = 1e-13
RAMAN_SHIFT_FREQ = 13.2e12
ASE_OUTPUT_POWER_GUESS = 1e-3
RAMAN_GAIN_WL_BANDWIDTH = 5e-9
REP_RATE_LOWER_LIMIT = 1e4  # Hz
MODES_IN_SINGLE_MODE_FIBER = 2  # Two polarization modes
RAMAN_MODES_IN_PM_FIBER = 1
YB_UPPER_STATE_LIFETIME = 1e-3  # s
c = 299792458
h = 6.62607e-34


# Constants for the numerical algorithm
SIMULATION_MIN_POWER = 1e-14
SOLVER_MAX_NODES = 20000


# Default absorption and emission cross section files
spectrum_folder = os.path.join('..', 'fiber_spectra')
YB_ABSORPTION_CS_FILE = os.path.join(spectrum_folder, 'ytterbium absorption cross sections.dat')
YB_EMISSION_CS_FILE = os.path.join(spectrum_folder, 'ytterbium emission cross sections.dat')