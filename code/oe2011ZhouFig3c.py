# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:49:14 2021
BDG reflectivity with respect to probe
The model is based on (9)
@author: YeK
Ref: OE 2011 Four-wave mixing analysis of Brillouin
dynamic grating in a polarization-maintaining fiber: theory and experiment
"""

import numpy as np
from numpy import exp, conj, pi, sqrt
from scipy.constants import speed_of_light
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from odeintw import odeintw


def power(A):
    """
    power of the wave

    Parameters
    ----------
    A : complex128
        The power amplitudes of the wave.

    Returns
    -------
    Power of the wave.
    """
    return abs(A * conj(A))


def dadz(z, y, kappa, delta_k, A1, A3):
    """
    four wave mixing coupled equations

    Parameters
    ----------
    z : float64
        Locations at the optical fiber.
    y : complex128
        The power amplitudes of the pump2 and the diffracted wave.
    kappa : float64
        Coupling constant.
    delta_k : float64
        Phase mismatch which is related to the frequency difference.
    A1 : float64
        Power amplitude of the pump1
    A3 : float64
        Power amplitude of the probe

    Returns
    -------
    Left hand side of the coupled equations
    """
    A2, A4 = y
    diff = [
        -kappa * (A2 * power(A1) + A1 * A4 * conj(A3) * exp(-1j * delta_k * z)),
        -kappa * (A4 * power(A3) + A2 * A3 * conj(A1) * exp(1j * delta_k * z))
        ]
    return diff


def bc(y0, yL, kappa, delta_k, A1, A3):
    """
    boundary conditions for the BVP solver

    Parameters
    ----------
    y0 : array
        Power amplitudes of the pump2 and the diffracted wave at z=0
    yl : array
        Power amplitudes of the pump2 and the diffracted wave at z=L.
    kappa : float64
        Coupling constant.
    delta_k : float64
        Phase mismatch which is related to the frequency difference.

    Returns
    -------
    The residuals of the boundary condition, the values of which should be zero.

    """
    A20, A40 = y0
    A2L, A4L = yL
    """
    The boundary conditions are A10 = sqrt(0.2), A2L = sqrt(0.01), A30 = sqrt(0.1), A4L = 0
    """
    return [A2L - sqrt(0.011), A4L]


def plotPower(z_span, data, style=['science', 'no-latex'], ax=None, lab=''):
    """
    Plot the power of the pump2 and the diffracted wave along the optical fiber

    Parameters
    ----------
    z_span : 1dArray
        Locations at the optical fiber.
    data : 2dArray
        Power amplitudes of the pump2 and the diffracted wave along the optical fiber.
    style : string, optional
        matplotlib subplot style. The default is ['science', 'no-latex'].
    ax : int, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    Power of the pump2 and the diffracted wave in one figure.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, dpi=200, figsize=(6, 3))

    p2 = power(data[:, 0])  # power of the pump2
    p4 = power(data[:, 1])  # power of the diffracted wave

    ax.plot(z_span, abs(p2), label='pump2'+lab)
    ax.plot(z_span, abs(p4), label='diffracted wave')
    ax.legend()
    return ax


if __name__ == '__main__':
    """
    parameters of the optical fiber
    """
    length = 13  # the length of the waveguide, unit: m
    lamb = 1550 * 10**(-9)  # wavelength at 1550 nm
    rho = 2210   # density of the optical fiber, unit kg/m ** (-3)
    r_e = 0.902  # electrostrictive constant of the silicon nitride
    gamma_b = (2 * pi) * 30 * 10**6  # Brillouin bandwidth, unit: rad * Hz
    omega_b = (2 * pi) * 10875 * 10**6  # Brillouin frequency, unit: rad * Hz
    A_eff = 80 * 10**(-12)  # acoustic-optic effective area, unit: m ** 2

    # Coupling constant
    kappa = 8 * pi**3 * r_e**2 / (speed_of_light * rho * lamb**3 * omega_b * gamma_b * A_eff)
    # Phase mismatch which is related to the frequency difference
    delta_k = 0
    # Power amplitudes of the pump1, which are held constant along the fiber
    A1 = sqrt(0.183)
    """
    Sweep the power of probe to obtain fig. 3(c)
    """
    A3_range = sqrt(np.linspace(50, 450, 100) * 10**(-3))
    rfwm_range = np.zeros(len(A3_range))
    for A3, i in zip(A3_range, np.arange(len(A3_range))):
        """
        define the power amplitudes of the pump2 and the diffracted wave and the length span of the
        optical fiber
        """
        z_span = np.linspace(0, length, 1000)
        A2_span = np.zeros(len(z_span))  # power amplitude of the pump2 along the fiber
        A4_span = np.zeros(len(z_span))  # power amplitude of the diffracted wave along the fiber

        """
        initial value of the pump2 and the diffracted wave at z=0
        """
        init = [
            sqrt(0.01), 0j
            ]

        """
        initial guessing for the BVP solver
        """
        dataStart = odeintw(dadz, init, z_span, args=(kappa, delta_k, A1, A3), tfirst=True)

        """
        BVP solver
        """
        data = solve_bvp(
            lambda z, y: dadz(z, y, kappa=kappa, delta_k=delta_k, A1=A1, A3=A3),
            lambda y0, yL: bc(y0, yL, kappa=kappa, delta_k=delta_k, A1=A1, A3=A3),
            z_span,
            dataStart.T
            )

        # The calculated reflectivity from equation (9)
        A40 = data.y[1, 0]
        rfwm_cal = power(A40)/power(A3)
        rfwm_range[i] = rfwm_cal
    plt.plot(A3_range**2, rfwm_range * 10**(3), label='p1=183mW, p2 = 11mW')
    plt.legend()
