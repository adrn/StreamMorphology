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

potential = gp.LeeSutoTriaxialNFWPotential(v_c=0.22, r_s=30.,
                                           a=1., b=1., c=0.8, units=galactic)

def test_tube():
    for E in np.linspace(-0.12, -0.2, 5):
        w0 = tube_grid(E=E, potential=potential, dx=1., dz=1.)
        Es = potential.total_energy(w0[:,:3], w0[:,3:])
        np.testing.assert_allclose(Es, E)

def test_box():
    for E in np.linspace(-0.12, -0.2, 5):
        w0 = box_grid(E=E, potential=potential, approx_num=16)
        Es = potential.total_energy(w0[:,:3], w0[:,3:])
        np.testing.assert_allclose(Es, E)
