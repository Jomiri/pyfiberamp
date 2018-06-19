import numpy as np
from scipy.special import j0 as J0
from scipy.special import k0 as K0
from scipy.integrate import quad
import matplotlib.pyplot as plt


def fundamental_mode(beta_factor, r, a, wl, n1, n2):
    k0 = 2*np.pi/wl
    beta = beta_factor*k0*n1
    kappa = np.sqrt((n1*k0)**2 - beta**2)
    gamma = np.sqrt(beta**2 - (n2 * k0)**2)
    if abs(r) < a:
        return J0(kappa*r)
    else:
        return K0(gamma*abs(r))


def fundamental_mode_matching_func(beta_factor, a, wl, n1, n2):
    k0 = 2 * np.pi / wl
    beta = beta_factor * k0 * n1
    kappa = np.sqrt((n1 * k0) ** 2 - beta ** 2)
    gamma = np.sqrt(beta ** 2 - (n2 * k0) ** 2)
    return J0(kappa * a) - K0(gamma * abs(a))


