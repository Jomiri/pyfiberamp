================
 PyFiberAmp
================

Introduction
============
PyFiberAmp is a rate equation simulation module for rare earth fiber amplifiers. It uses the Giles model [1]_.
Additionally, stimulated and spontaneous Raman scattering effects in both active and passive fibers can be modeled
with a similar basic rate equation model [2]_.

Installation
============

Examples
========
For usage examples, please see the Examples Jupyter notebook.

Theory basics
==============

Giles model
-----------
The original Giles model [1]_ describes an active fiber as *a two-level system* using four spectroscopic parameters:

1. Absorption spectrum :math:`\alpha` (dB/m vs. wavelength)
2. Gain spectrum :math:`g^{*}` (dB/m vs. wavelength)
3. Saturation parameter :math:`\zeta` (1/(ms))
4. Linear loss :math:`l` (dB/m)

The absorption and gain spectra are defined as

.. math:: \alpha (\lambda) = \sigma_a (\lambda) \Gamma (\lambda) n_t
.. math:: g^* (\lambda) = \sigma_e (\lambda) \Gamma (\lambda) n_t

where :math:`\sigma_a (\lambda)` and :math:`\sigma_e (\lambda)` are the absorption and emission cross sections,
respectively, :math:`\Gamma (\lambda)` is the overlap integral between the optical beam and the dopant,
and :math:`n_t` is the number density of the dopant ions. The absorption spectrum refers to the small signal absorption
with all the dopant ions in the lower manifold. The gain spectrum refers to the small signal gain with all the ions in
the upper manifold.

The saturation parameter is defined as

.. math:: \zeta = \pi b_{eff}^2 n_t/\tau

where :math:`b_{eff}` is the effective doped radius (equals to core radius in a uniformly doped fiber),
and :math:`\tau` is the upper state lifetime of the ions.

The loss :math:`l` can typically be assumed to be constant within the limited bandwidth of simulated wavelengths.

The optical powers (:math:`P_k`, the overline denotes spatial averaging) and the fraction of ions in the excited state
(\dfrac{\overline{n}_2}{\bar{n}_t}) are then described by the following system of equations

.. math:: \dfrac{\overline{n}_2}{\bar{n}_t} = \dfrac{\sum_k{\dfrac{P_k(z) \alpha_k}{h \nu_k \zeta}}}{1 + \sum_k{\dfrac{P_k(z)(\alpha_k + g_k^*)}{h \nu_k \zeta}}}
.. math:: \dfrac{dP_k}{dz} = u_k(\alpha_k + g_k^*)\dfrac{\overline{n}_2}{\bar{n}_t}P_k(z) + u_k g_k^* \dfrac{\overline{n}_2}{\bar{n}_t} m h \nu_k \Delta \nu_k - u_k(\alpha_k + l_k) P_k

where :math:`h \nu_k` is the photon energy at frequency :math:`nu`, :math:`u_k` is the propagation direction of the beam
(+1 is forward and -1 is backward), :math:`m=2` is the number of modes in a single-mode fiber (two polarization modes)
and :math:`\Delta \nu_k` is the frequency bandwidth of the beam (can be assumed zero for signals and non-zeros for
amplified spontaneous emission and Raman).

The Giles model is an 1D boundary value problem. In the simplest case, the boundary values are just the input powers of
the beams. If only forward-propagating beams are considered, the problem is reduced to an easily integrable initial value
problem, but typically both backward-propagating ASE and pump beams must be modeled, so some input powers are know only
at the end of the fiber and some at the beginning. PyFiberAmp uses the solve_bvp solver from SciPy. The solver typically
converges rapidly provided that the initial guess is chosen well.

The Giles model suffers from several limitations, the most important of which are listed below:
1. As a two-level model, it cannot describe complex transitions such as excited state absorption.
2. As a steady-state model, it cannot describe pulse propagation at low repetition rates, when the gain varies from pulse to pulse.
3. As a spatially averaged model, it cannot describe radially varying dopant distributions or power-dependent overlap between the optical beams and the dopant.

Raman model
-----------
Stimulated and spontaneous Raman scattering is simulated with the following rate equations:

.. math:: \dfrac{dP_R^{\pm}}{dz} = \mp(g_R I_s P_R^{\pm} + g_R I_s h \nu_R \Delta \nu_R)

where the first term on the right hand side describes stimulated Raman scattering and the second term describes
spontaneous Raman scattering, :math:`g_R` is the Raman gain coefficient (about :math:`1\times 10^{-13} m/W`),
:math:`I_s` is the peak intensity of the signal, and the subsript R refers to the Raman channel. The derivative of
backward-propagating Raman beam is negative.

The signal is depleted by the amount of power gained by the Raman beams multiplied by the ratio of photon frequencies
(energies) of the signal and Raman beams because the number of photons is conserved by the Raman photons have smaller
energies than the signal photons.

.. math:: \dfrac{dP_s}{dz} = -\dfrac{\nu_s}{\nu_R} \left(\dfrac{dP_R^+}{dz} - \dfrac{dP_R^-}{dz}\right)

In practice, terms on the right hand side of these equations are added to the Giles model rate equations to simulated
Raman generation in active fibers, where the Raman signal interacts with the dopant ions as well.

Fiber data
==========
PyFiberAmp comes with cross section data for ytterbium in germanosilicate glass [3]_ and supports importing
spectroscopic data of other Yb-doped and rare earth fibers. The default ytterbium upper state lifetime is 1.0 ms,
but different values can be used as well.




References
===========
.. [1] C.R. Giles and E. Desurvire, "Modeling erbium-doped fiber amplifiers," in Journal of Lightwave Technology, vol. 9, no. 2, pp. 271-283, Feb 1991. doi: 10.1109/50.65886
.. [2] R.G. Smith, "Optical Power Handling Capacity of Low Loss Optical Fibers as Determined by Stimulated Raman and Brillouin Scattering," Appl. Opt. 11, 2489-2494 (1972)
.. [3] R. Paschotta, J. Nilsson, A. C. Tropper and D. C. Hanna, "Ytterbium-doped fiber amplifiers," in IEEE Journal of Quantum Electronics, vol. 33, no. 7, pp. 1049-1056, Jul 1997. doi: 10.1109/3.594865