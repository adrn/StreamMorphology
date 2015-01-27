# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

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
from ..initialconditions import tube_grid, box_grid

def test_tube():
    potential = gp.LeeSutoTriaxialNFWPotential(v_c=0.22, r_s=30.,
                                               a=1., b=1., c=0.8, units=galactic)

    for E in np.linspace(-0.12, -0.2, 5):
        w0 = tube_grid(E=E, potential=potential, dx=1., dz=1.)

        # TODO: check that energy of all orbits is same as input
        Es = potential.total_energy(w0[:,:3], w0[:,3:])
        np.testing.assert_allclose(Es, E)

def test_call_box():
    E = -0.15

    potential = gp.LeeSutoTriaxialNFWPotential(v_c=0.22, r_s=30.,
                                               a=1., b=1., c=0.8, units=galactic)

    w0 = box_grid(E=E, potential=potential, Ntotal=16)

    for i in range(len(w0)):
        try:
            t,w = potential.integrate_orbit(w0[i], dt=1., nsteps=60000,
                                            Integrator=gi.DOPRI853Integrator,
                                            Integrator_kwargs=dict(nsteps=128))

            fig = gd.plot_orbits(w, linestyle='none', alpha=0.1)

            E = potential.total_energy(w[:,0,:3].copy(), w[:,0,3:].copy())
            dE = np.abs(E[1:] - E[0])
            print(i,dE.max())

            # plt.figure()
            # plt.semilogy(dE)
            # plt.show()

        except RuntimeError:
            print("Failed.")
