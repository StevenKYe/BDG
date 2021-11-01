# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:49:14 2021
Moving FGD model
@author: YeK
Ref: OE 2011 Four-wave mixing analysis of Brillouin
dynamic grating in a polarization-maintaining fiber: theory and experiment
"""

import numpy as np
from numpy import exp, conj, pi, sqrt, tanh
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


def dadz(z, y, kappa, delta_k, A1, A2):
    """
    moving FBG model

    Parameters
    ----------
    z : float64
        Locations at the optical fiber.
    y : complex128
        The power amplitudes of the probe and the diffracted wave.
    kappa : float64
        Coupling constant.
    delta_k : float64
        Phase mismatch which is related to the frequency difference.
    A1 : float64
        Power amplitude of the pump1
    A2 : float64
        Power amplitude of the pump2
    Returns
    -------
    Left hand side of the coupled equations
    """
    A3, A4 = y
    diff = [
        -kappa * A1 * A4 * conj(A2) * exp(-1j * delta_k * z),
        -kappa * A2 * A3 * conj(A1) * exp(1j * delta_k * z)
        ]
    return diff


def bc(y0, yL, kappa, delta_k, A1, A2):
    """
    boundary conditions for the BVP solver

    Parameters
    ----------
    y0 : array
        Power amplitudes of the probe and the diffracted wave at z=0
    yl : array
        Power amplitudes of the probe and the diffracted wave at z=L.
    kappa : float64
        Coupling constant.
    delta_k : float64
        Phase mismatch which is related to the frequency difference.

    Returns
    -------
    The residuals of the boundary condition, the values of which should be zero.

    """
    A30, A40 = y0
    A3L, A4L = yL
    """
    The boundary conditions are A30 = sqrt(0.1), A4L = 0
    """
    return [A30 - sqrt(0.1), A4L]


def plotPower(z_span, data, style=['science', 'no-latex'], ax=None, lab=''):
    """
    Plot the power of the probe and the diffracted wave along the optical fiber

    Parameters
    ----------
    z_span : 1dArray
        Locations at the optical fiber.
    data : 2dArray
        Power amplitudes of the probe and the diffracted wave.
    style : string, optional
        matplotlib subplot style. The default is ['science', 'no-latex'].
    ax : int, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    Power of the probe and the diffracted wave.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, dpi=200, figsize=(6, 3))

    p3 = power(data[:, 0])  # power of the probe
    p4 = power(data[:, 1])  # power of the diffracted wave

    ax.plot(z_span, abs(p3), label='probe' + lab)
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

    #  Coupling constant
    kappa = 8 * pi**3 * r_e**2 / (speed_of_light * rho * lamb**3 * omega_b * gamma_b * A_eff)
    #  Phase mismatch which is related to the frequency difference
    delta_k = 0
    #  Power amplitudes of the pump1 and pump2, which are held constant along the fiber
    A1 = sqrt(0.2)
    A2 = sqrt(0.01)

    """
    Define the power amplitudes of the four waves and the length span of the optical fiber
    """
    z_span = np.linspace(0, length, 1000)
    A3_span = np.zeros(len(z_span))  # power amplitude of the probe along the fiber
    A4_span = np.zeros(len(z_span))  # power amplitude of the diffracted wave along the fiber

    """
    initial guessing of the probe, and the diffracted wave at z=0
    """
    init = [
        sqrt(0.1) + 0j, 0j
        ]

    """
    initial guessing for the BVP solver
    """
    dataStart = odeintw(dadz, init, z_span, args=(kappa, delta_k, A1, A2), tfirst=True)

    """
    BVP solver
    """
    data = solve_bvp(
        lambda z, y: dadz(z, y, kappa=kappa, delta_k=delta_k, A1=A1, A2=A2),
        lambda y0, yL: bc(y0, yL, kappa=kappa, delta_k=delta_k, A1=A1, A2=A2),
        z_span,
        dataStart.T
        )
    plotPower(data.x, data.y.T, lab=' bvp')

    #  The estimated reflectivity from equation (8)
    rfbg_est = tanh(kappa*abs(A1)*abs(A2)*length)**2

    """
    The calculated reflectivity from equation (6)
    (data.y[1,0])**2 is the power of the diffracted wave at z=0
    (data.y[0,0])**2 is the power of the probe at z=0
    """
    rfbg_cal = (data.y[1, 0])**2/(data.y[0, 0])**2

    print('The calculated reflectivity from equation (6) is {:.6f}'.format(abs(rfbg_cal)))
    print('The estimated reflectivity from equation (8) is {:.6f}'.format(rfbg_est))
