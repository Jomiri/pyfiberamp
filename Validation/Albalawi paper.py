from fiberamp import FiberAmplifierSimulation
from fiberamp.fibers import YbDopedDoubleCladFiber


Yb_number_density = 3e25
core_r = 5e-6
background_loss = 0
length = 3
pump_cladding_r = 50e-6
core_to_cladding_ratio = core_r / pump_cladding_r
core_NA = 0.12
npoints = 20
tolerance = 1e-5

fiber = YbDopedDoubleCladFiber(length,
                        core_r, Yb_number_density,
                        background_loss, core_NA, core_to_cladding_ratio)
simulation = FiberAmplifierSimulation(fiber)
simulation.add_cw_signal(wl=1030e-9, power=0.4, mode_field_diameter=2*4.8e-6)
simulation.add_counter_pump(wl=914e-9, power=47.2)
#simulation.include_ase(wl_start=1000e-9, wl_end=1100e-9, n_bins=50)

result = simulation.run(npoints, tol=tolerance)
assert(result.success())
result.db_scale = False
print(result.forward_signals[0, -1])
print(result.backward_pumps[0, 0])
print(result.average_excitation)
result.plot_amplifier_result()