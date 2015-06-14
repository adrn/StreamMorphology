# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys
import logging

# Third-party
import numpy as np
import gary.dynamics as gd
import gary.potential as gp
from gary.units import galactic

# Project
from ..fast_mle import mle

def test_mle():
    potential = gp.LogarithmicPotential(v_c=1., r_h=0.1, q1=1., q2=1., q3=0.8, units=galactic)

    w0 = np.array([1., 0., 1., 0., 0.75, 0.])
    l,t,w = gd.fast_lyapunov_max(w0, potential, dt=0.1, nsteps=100000)
    l2,t2,w2 = mle(w[0].copy(), potential, dt=0.1, nsteps=100000)

    print(l[-2] - l2)
    print(t[-2] - t2)
    print(w[-2] - w2.reshape(3,6))
