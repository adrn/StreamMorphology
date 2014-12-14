# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from astropy import log as logger
import astropy.units as u
import matplotlib.pyplot as plt

# Project
import gary.dynamics as gd
import gary.integrate as gi
import gary.potential as gp
from gary.units import galactic
from ..initialconditions import loop_grid, box_grid

def test_call():
    E = -0.15

    potential = gp.LeeSutoTriaxialNFWPotential(v_c=0.22, r_s=30.,
                                               a=1., b=1., c=0.8, units=galactic)

    w0 = loop_grid(E=E, potential=potential)

    ix = np.random.randint(len(w0), size=100)
    t,w = potential.integrate_orbit(w0[ix], dt=1, nsteps=60000,
                                    Integrator=gi.DOPRI853Integrator)

    for i in range(100):
        E = potential.total_energy(w[:,i,:3].copy(), w[:,i,3:].copy())
        dE = np.abs(E[1:] - E[0])
        print(dE.max())

        if np.any(dE > 1E-9):
            fig = gd.plot_orbits(w, ix=i, linestyle='none', alpha=0.1)

            plt.figure()
            plt.semilogy(dE)
            plt.show()
            raise ValueError("Energy conservation failed!")
