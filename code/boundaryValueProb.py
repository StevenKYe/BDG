# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 09:33:30 2021

@author: YeK
"""

import numpy as np
from scipy.integrate import solve_bvp, odeint
import matplotlib.pyplot as plt


def pend(t, y, b, c):
    theta, omega = y
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return dydt


def bc(y0, y1, b, c):
    # Values at t=0:
    theta0, omega0 = y0

    # Values at t=100:  
    theta1, omega1 = y1

    # These return values are what we want to be 0:
    return [omega0, theta1]


b = 0.02
c = 0.08

t = np.linspace(0, 100, 201)

# Use the solution to the initial value problem as the initial guess
# for the BVP solver. (This is probably not necessary!  Other, simpler
# guesses might also work.)
ystart = odeint(pend, [1, 0], t, args=(b, c,), tfirst=True)


result = solve_bvp(lambda t, y: pend(t, y, b=b, c=c),
                   lambda y0, y1: bc(y0, y1, b=b, c=c),
                   t, ystart.T)


plt.figure(figsize=(6, 3), dpi=200)
plt.plot(result.x, result.y[0], label=r'$\theta(t)$')
plt.plot(result.x, result.y[1], '--', label=r'$\omega(t)$')
plt.xlabel('t')
plt.grid()
plt.legend()
plt.tight_layout()

plt.show()