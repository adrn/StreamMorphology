# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
import gary.potential as gp
from gary.units import galactic

# Project
from ..fast_ensemble import ensemble_integrate

def test_integrate():
    potential = gp.LogarithmicPotential(v_c=1., r_h=0.1, q1=1., q2=1., q3=1., units=galactic)

    norbits = 100
    w0 = np.zeros((norbits, 6))
    w0[:,:] = np.array([1.,0.,0.,0.,0.9,0.])[None]
    w = ensemble_integrate(potential.c_instance, w0, dt0=0.5, nsteps=10000, t0=0.)

    for i in range(1,norbits):
        assert np.all(w[0] == w[i])
