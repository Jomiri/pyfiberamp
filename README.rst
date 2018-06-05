===========================
 Introduction to PyFiberAmp
===========================

In short
============
PyFiberAmp is a rate equation simulation tool for rare-earth-doped fiber amplifiers and fiber lasers partly based on the Giles
model [1]_. PyFiberAmp allows an arbitrary number of user-defined time-varying signals from continuous-wave to nanosecond pulses in both
core-pumped and double-clad fiber amplifiers. It also includes reflective boundary conditions for modeling simple CW, gain-switched and
Q-switched fiber lasers. The time-dynamic simulations are accelerated by a dedicated partial differential equation solver written in C++.
In addition to signal and pump beams, PyFiberAmp also supports amplified spontaneous emission (ASE) and stimulated and spontaneous Raman scattering effects (SRS) [2]_ (SRS only in steady-state simulations).
Documentation (still in progress) is available on `Read the Docs <https://pyfiberamp.readthedocs.io/en/latest/index.html>`_.
Do you have a question, comment or a feature request? Please open a new issue on GitHub
or contact me at pyfiberamp@gmail.com.

A visual example
=================
Few-nanosecond pulses propagating in an Yb-doped fiber amplifier are distorted because of gain saturation.
The Gaussian pulse with its exponential leading edge retains its shape better than the square or saw-tooth
pulses.

.. image:: docs/images/pulses.gif
    :align: center


.. Listaus: 1) esimerkkinotebookit, 2) päivitetyt kuvat readmehen 3) asennustesti tällä ja toisella koneella 4) merge

Download
=========
PyFiberAmp is not yet on PyPI. You can either download the code as zip-file or clone the repository with
::

    $ git clone git://github.com/Jomiri/pyfiberamp.git

and then install the library with
::

    python setup.py install

System requirements
===================
PyFiberAmp depends on the standard scientific Python packages: Numpy, SciPy and Matplotlib and has been
tested on Windows 7 and Windows 10. It should work on other operating systems as well
provided that Python and the required packages are installed. The `Anaconda distribution
<https://www.anaconda.com/download/>`_ contains everything out of the box.

Even though all of PyFiberAmp's functionality is available in interpreted Python code, the use of the separate
C++ extension is recommended for computationally intensive time-dynamic simulations.
The system requirements of the C++ extension are stricter: Windows 7 or 10, Python 3.6 and a fairly modern
CPU with AVX2 instruction support. If you cannot satisfy the requirements but the Python based solver is
too slow for you, open a new issue or send me a message, and I'll try to provide you with a compatible C++ version.

Example
========
The simple example below demonstrates a core-pumped Yb-doped fiber amplifier. All units are in SI.
::

    from pyfiberamp.steady_state import SteadyStateSimulation
    from pyfiberamp.fibers import YbDopedFiber

    yb_number_density = 2e25  # m^-3
    core_radius = 3e-6  # m
    length = 2.5  # m
    core_na = 0.12

    fiber = YbDopedFiber(length=length,
                        core_radius=core_radius,
                        ion_number_density=yb_number_density,
                        background_loss=0,
                        core_na=core_na)
    simulation = SteadyStateSimulation()
    simulation.fiber = fiber
    simulation.add_cw_signal(wl=1035e-9, power=2e-3)
    simulation.add_forward_pump(wl=976e-9, power=300e-3)
    simulation.add_ase(wl_start=1000e-9, wl_end=1080e-9, n_bins=80)

    result = simulation.run(tol=1e-5)
    result.plot_amplifier_result()

The script plots the power evolution in the amplifier and the amplified spontaneous emission (ASE) spectra. The
co-propagating pump is absorbed in the first ~1.8 m while the signal experiences gain. When the pump has been depleted,
the signal starts to be reabsorbed by the fiber. This design is not perfect 1) because the fiber is too long and
2) because the input signal is too weak to saturate the gain at the start of the fiber.

.. image:: docs/images/readme_power_evolution.png
    :align: center
    :width: 769px
    :height: 543px

.. image:: docs/images/readme_ase_spectra.png
    :align: center
    :width: 769px
    :height: 543px

For more usage examples, please see the Jupyter notebooks in the examples folder. Many more examples will be added in the
future.

Fiber data
==========
PyFiberAmp comes with spectroscopic data (absorption and emission cross sections) for Yb-doped germanosilicate fibers
[3]_ and supports importing spectra for other dopants/glass-compositions.

Theory basics
==============
For a quick view on the theory, see the `pyfiberamp theory.pdf
<https://github.com/Jomiri/pyfiberamp/blob/master/pyfiberamp%20theory.pdf>`_ file. A more complete description can be found in the
references.

License
========
PyFiberAmp is licensed under the MIT license. The C++ extension depends on the `pybind11
<https://github.com/pybind/pybind11>`_  and `Armadillo <http://arma.sourceforge.net/>`_ projects. See the license file
for their respective licenses.

References
===========
.. [1] C.R. Giles and E. Desurvire, "Modeling erbium-doped fiber amplifiers," in Journal of Lightwave Technology, vol. 9, no. 2, pp. 271-283, Feb 1991. doi: 10.1109/50.65886
.. [2] R.G. Smith, "Optical Power Handling Capacity of Low Loss Optical Fibers as Determined by Stimulated Raman and Brillouin Scattering," Appl. Opt. 11, 2489-2494 (1972)
.. [3] R. Paschotta, J. Nilsson, A. C. Tropper and D. C. Hanna, "Ytterbium-doped fiber amplifiers," in IEEE Journal of Quantum Electronics, vol. 33, no. 7, pp. 1049-1056, Jul 1997. doi: 10.1109/3.594865
