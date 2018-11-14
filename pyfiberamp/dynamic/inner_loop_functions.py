"""
These are a handful of pure python functions that can be sped up with numba or
pythran.

These bottleneck functions are called many times in an inner loop for the
dynamic simulation, and therefore are good candidates for speed optimization.
All of these functions have side effects such that the first argument passed to
them is modified in-place.

To use pythran, execute this from the command line:
    $ pythran -o fiber_simulation_pythran_bindings.so inner_loop_functions.py
where the shared-library filename extension is normally .so on linux and .pyd
on windows. Pythran translates the python code to C++11 and compiles it to a
binary extension module. It takes about 80 seconds to compile with g++ 8.2.1 on
a first-generation core-i7 CPU.

"""
import numpy as np


#pythran export dPdZ(float[][], float[][], float[][], float[][], float[][], float, int, int, bool)
def dPdZ(P_hat, N2, a_g_per_Nt, a_l, g_m_h_v_dv_per_Nt, dz, num_ion_populations, n_channels, add):
    """Modifies `P_hat` in-place."""
    out = np.zeros_like(P_hat)
    for i in range(num_ion_populations):
        start = i * n_channels
        for j in range(out.shape[0]):
            for k in range(out.shape[1]):
                out[j, k] += P_hat[j, k] * (a_g_per_Nt[start+j, k] * N2[i, k] - a_l[start+j, k]) + (g_m_h_v_dv_per_Nt[start+j, k] * N2[i, k])
    if add:
        P_hat[:, :] = P_hat + (dz * out)
    else:
        P_hat[:, :] = P_hat - (dz * out)

#pythran export dNdT(float[][], float[][], float[][], float[][], float, float, int, int)
def dNdT(N2, P, a_per_h_v_pi_r2, a_g_per_h_v_pi_r2_Nt, A, dt, num_ion_populations, n_channels):
    """Modifies `N2` in-place."""
    out = np.empty_like(N2)
    for i in range(num_ion_populations):
        start = i * n_channels
        tmp = np.zeros(out.shape[1], dtype=out.dtype)
        for k in range(out.shape[1]):
            for j in range(n_channels):
                tmp[k] += P[j, k] * (a_per_h_v_pi_r2[start+j, k] - a_g_per_h_v_pi_r2_Nt[start+j, k] * N2[i, k])
            out[i, k] = tmp[k] - A * N2[i, k]
    N2[:, :] = N2 + (dt * out)

#pythran export shift_against_propagation_direction_to_from(float[][], float[][], int)
def shift_against_propagation_direction_to_from(P_hat_backward, P_hat_forward, n_forward):
    """Modifies `P_hat_backward` in-place."""
    P_hat_backward[:n_forward, :-1] = P_hat_forward[:n_forward, 1:]
    P_hat_backward[n_forward:, 1:] = P_hat_forward[n_forward:, :-1]

#pythran export shift_to_propagation_direction_to_from(float[][], float[][], int)
def shift_to_propagation_direction_to_from(P_hat_forward, P, n_forward):
    """Modifies `P_hat_foreward` in-place."""
    P_hat_forward[:n_forward, 1:] = P[:n_forward, :-1]
    P_hat_forward[n_forward:, :-1] = P[n_forward:, 1:]

#pythran export min_clamp(float[][], float)
def min_clamp(arr, min_value):
    """Modifies `arr` in-place."""
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] < min_value:
                arr[i, j] = min_value

#pythran export apply_input(float[][], float[][], int, int)
def apply_input(P, P_in_out, idx_iteration, n_forward):
    """Modifies `P` in-place."""
    P[:n_forward, 0] = P_in_out[:n_forward, idx_iteration]
    P[n_forward:, -1] = P_in_out[n_forward:, idx_iteration]

#pythran export apply_output(float[][], float[][], int, int)
def apply_output(P_in_out, P_hat_forward, idx_iteration, n_forward):
    """Modifies `P_in_out` in-place."""
    P_in_out[:n_forward, idx_iteration] = P_hat_forward[:n_forward, -1]
    P_in_out[n_forward:, idx_iteration] = P_hat_forward[n_forward:, 0]

#pythran export apply_reflection(float[][], int[], int[], float[], int)
def apply_reflection(P, source_idx, target_idx, R, n_forward):
    """Modifies `P` in-place."""
    for _source_idx, _target_idx, _R in zip(source_idx, target_idx, R):
        if _source_idx < n_forward:
            P[_target_idx, -1] += _R * P[_source_idx, -2]
        else:
            P[_target_idx, 0] += _R * P[_source_idx, 1]

#pythran export new_P(float[][], float[][], float[][])
def new_P(P, P_hat_forward, P_hat_backward):
    """Modifies `P` in-place."""
    P[:, :] = P_hat_forward + 0.5 * (P - P_hat_backward)
