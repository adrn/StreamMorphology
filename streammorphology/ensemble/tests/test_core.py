# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys
import logging

# Third-party
import matplotlib.pyplot as plt
import numpy as np
from astropy import log as logger
import gary.dynamics as gd

# Project
from ..core import align_ensemble, compute_align_matrix
from ...potential import potential_registry

logger.setLevel(logging.DEBUG)

plot_path = "plots/tests/TODO"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

potential = potential_registry['triaxial-NFW']

def test_align_orbit():
    # start with an orbit that circulates x
    w0 = [1., 0., 30., 0., 0.15, -0.1]
    t,w = potential.integrate_orbit(w0, dt=1., nsteps=25000)
    w = w[:,0]

    # fig = gd.plot_orbits(w, marker=None)
    # plt.show()

    R = compute_align_matrix(w[-1])
    new_x = np.array(R.dot(w[:,:3].T).T)
    new_v = np.array(R.dot(w[:,3:].T).T)
    new_L = np.cross(new_x[-1], new_v[-1])[0]

    a = np.array([0., 0., np.linalg.norm(new_L)])
    b = new_L
    assert np.allclose(a,b)

    a = np.array([np.linalg.norm(new_x[-1]), 0., 0.])
    assert np.allclose(a,new_x[-1])

def test_align_many_orbits():
    # start with an orbit that circulates x
    w0 = [1., 0., 30., 0., 0.15, -0.1]
    t,w = potential.integrate_orbit(w0, dt=1., nsteps=25000)
    w = w[:,0]

    # fig = gd.plot_orbits(w, marker=None)
    # plt.show()

    for i in np.random.randint(len(w), size=100):
        print(i)
        R = compute_align_matrix(w[i])
        new_x = R.dot(w[:i+1,:3].T).T
        new_v = R.dot(w[:i+1,3:].T).T
        new_L = np.cross(new_x[-1], new_v[-1])[0]

        a = np.array([0., 0., np.linalg.norm(new_L)])
        b = new_L
        assert np.allclose(a,b)

        a = np.array([np.linalg.norm(new_x[-1]), 0., 0.])
        assert np.allclose(a,new_x[-1])
