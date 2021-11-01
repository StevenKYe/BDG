# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:49:14 2021
Coupled equations for all the four optical waves
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


def eta(Omega, r_e, omega1, rho, omega_b, gamma_b, A_eff):
    """
    The frequency related coupling coefficient, the variable is Omega

    Parameters
    ----------
    Omega : float64
        Variable; The actual frequecny difference between the two pumps (also that between the probe and
                                                                         the diffracted wave).
    r_e : float64
        Constant; The electrostrictive constant.
    omega1 : float64
        Constant; Frequency of the pump1
    rho : float64
        Constant; Density of the core of the optical fiber.
    omega_b : float64
        Constant; Brillouin frequency of the optical fiber.
    gamma_b : float64
        Constant; Brillouin linewidth of the optical fiber.
    A_eff : float64
        Constant; Acousto-optic effective area.

    Returns
    -------
    Coupling coefficient at certain frequency.
    """
    return r_e**2 * omega1**3 / (rho * speed_of_light**4 * A_eff * (omega_b**2 - Omega**2 - 1j
                                                                    * gamma_b * Omega))


def delta_omega(neff1, neff2, lamb):
    """
    The required angle frequency splitting between the pump1 and the probe for phase matching

    Parameters
    ----------
    neff1 : float64
        Effective index along the slow axes of the opticall fiber.
    neff2 : float64
        Effective index along the fast axes of the optical fiber.
    lamb : float64
        Wavelength of the pump1.

    Returns
    -------
    The required angle frequency spliting between the pump1 and the probe for phase matching
    """
    mean_n = (neff1 + neff2) / 2  # mean effective index
    delta_n = abs(neff1 - neff2)  # effective index variance along the optical fiber
    freq = speed_of_light / lamb  # frequency of the pump1, unit: rad*Hz
    return 2 * pi * delta_n * freq / mean_n


def delta_k(neff1, neff2, omega1, omega3):
    """
    Phase mismatch of the BDG when the frequencies of the pump1 and the probe are
    omega1 and omega3 seperately

    Parameters
    ----------
    neff1 : float64
        Effective index along the slow axes of the opticall fiber.
    neff2 : float64
        Effective index along the fast axes of the optical fiber.
    omega1 : float64
        Angle frequency of the pump1.
    omega3 : float64
        Angle frequency of the probe.

    Returns
    -------
    Phase mismatch of the BDG when the frequencies of the pump1 and the probe are
    omega1 and omega3 seperately
    """
    q1 = 2 * omega1 * neff1 / speed_of_light  # wavevector from pump1 and pump2
    q2 = 2 * omega3 * neff2 / speed_of_light  # Wavevector from probe and diffracted wave
    return q1 - q2


def dadz(z, y, eta1, eta2, delta_k):
    """
    four wave mixing coupled equations

    Parameters
    ----------
    z : float64
        Locations at the optical fiber.
    y : complex128
        The power amplitudes of the four waves.
    eta1 : float64
        Coupling coefficient between the two pumps, which is related to the frequency difference between
        the two pumps, the Brillouin frequency, and the Brillouin linewidth.
    eta2 : float64
        Coupling coefficient between the probe and the diffracted wave, which is related to the frequency
        difference between the two pumps, the Brillouin frequency, and the Brillouin linewidth.
    delta_k : float64
        Phase mismatch which is related to the frequency difference.

    Returns
    -------
    Left hand side of the coupled equations
    """
    A1, A2, A3, A4 = y
    diff = [
        1j * eta1 * (A1 * power(A2) + A2 * A3 * conj(A4) * exp(1j * delta_k * z)),
        -1j * conj(eta1) * (A2 * power(A1) + A1 * A4 * conj(A3) * exp(-1j * delta_k * z)),
        1j * eta2 * (A3 * power(A4) + A1 * A4 * conj(A2) * exp(-1j * delta_k * z)),
        -1j * conj(eta2) * (A4 * power(A3) + A2 * A3 * conj(A1) * exp(1j * delta_k * z))
        ]
    return diff


def bc(y0, yL, eta1, eta2, delta_k):
    """
    boundary conditions for the BVP solver

    Parameters
    ----------
    y0 : array
        Power amplitudes of the four waves at z=0
    yl : array
        Power amplitudes of the four waves at z=L.
    eta1 : float64
        Coupling coefficient between the two pumps, which is related to the frequency difference between
        the two pumps, the Brillouin frequency, and the Brillouin linewidth.
    eta2 : float64
        Coupling coefficient between the probe and the diffracted wave, which is related to the frequency
        difference between the two pumps, the Brillouin frequency, and the Brillouin linewidth.
    delta_k : float64
        Phase mismatch which is related to the frequency difference.

    Returns
    -------
    The residuals of the boundary condition, the values of which should be zero.

    """
    A10, A20, A30, A40 = y0
    A1L, A2L, A3L, A4L = yL
    """
    The boundary conditions are A10 = sqrt(0.2), A2L = sqrt(0.01), A30 = sqrt(0.1), A4L = 0
    """
    return [A10 - sqrt(0.2), A2L - sqrt(0.01), A30 - sqrt(0.1), A4L]


def plotPower(z_span, data, style=['science', 'no-latex'], ax=None, lab=''):
    """
    Plot the power of the four waves along the optical fiber

    Parameters
    ----------
    z_span : 1dArray
        Locations at the optical fiber.
    data : 2dArray
        Power amplitudes of the four waves along the optical fiber.
    style : string, optional
        matplotlib subplot style. The default is ['science', 'no-latex'].
    ax : int, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    Power of the four waves in one figure.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, dpi=200, figsize=(6, 3))

    p1 = power(data[:, 0])  # power of the pump1
    p2 = power(data[:, 1])  # power of the pump2
    p3 = power(data[:, 2])  # power of the probe
    p4 = power(data[:, 3])  # power of the diffracted wave

    ax.plot(z_span, abs(p1), label='pump1' + lab)
    ax.plot(z_span, abs(p2), label='pump2')
    ax.plot(z_span, abs(p3), label='probe')
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

    delta_k0 = 0   # Phase matching condition
    omega1 = 2 * pi * speed_of_light / lamb  # angle frequency of the pump1
    omega3 = omega1 + 2 * pi * (50 * 10**9)  # angle frequency of the probe

    #  actual frequency difference between the two pumps
    Omega_span = np.linspace(-100, 100, 200) * 2 * pi * 10**6 + omega_b

    rfwm_span = np.zeros(len(Omega_span))  # calculated reflectivity at different Omega
    for Omega, i in zip(Omega_span, np.arange(len(Omega_span))):

        #  Coupling coefficient between the two pumps
        eta1 = eta(Omega, r_e, omega1, rho, omega_b, gamma_b, A_eff)

        #  Coupling coefficient between the probe and the diffracted wave
        eta2 = eta(Omega, r_e, omega3, rho, omega_b, gamma_b, A_eff)

        """
        define the power amplitudes of the four waves and the length span of the optical fiber
        """
        z_span = np.linspace(0, length, 1000)
        A1_span = np.zeros(len(z_span))  # power amplitude of the pump1 along the fiber
        A2_span = np.zeros(len(z_span))  # power amplitude of the pump2 along the fiber
        A3_span = np.zeros(len(z_span))  # power amplitude of the probe along the fiber
        A4_span = np.zeros(len(z_span))  # power amplitude of the diffracted wave along the fiber

        """
        initial guessing of the pump1, pump2, probe, and the diffracted wave at z=0
        """
        init = [
            sqrt(0.2) + 0j, sqrt(0.01), sqrt(0.1) + 0j, 0j
            ]

        """
        initial guessing for the BVP solver
        """
        dataStart = odeintw(dadz, init, z_span, args=(eta1, eta2, delta_k0), tfirst=True)

        """
        BVP solver
        """
        data = solve_bvp(
            lambda z, y: dadz(z, y, eta1=eta1, eta2=eta2, delta_k=delta_k0),
            lambda y0, yL: bc(y0, yL, eta1=eta1, eta2=eta2, delta_k=delta_k0),
            z_span,
            dataStart.T
            )

        #  The calcualted power amplitude of the probe
        A30 = data.y[2, 0]

        #  The calculated power amplitude of the diffracted wave
        A40 = data.y[3, 0]

        """
        Calculate the reflectivity
        """
        rfwm_cal = power(A40)/power(A30)
        rfwm_span[i] = rfwm_cal

        """
        Calculation progress
        """
        print('Progress: {}%'.format(100 * i / len(Omega_span)))

    plt.plot((Omega_span - omega_b) / (2 * pi), rfwm_span)
