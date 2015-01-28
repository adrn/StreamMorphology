# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import numpy as np
import matplotlib.pyplot as plt

# Project
from ..core import ptp_periods, estimate_max_period, estimate_dt_nsteps

plot_path = "output/tests/freqmap"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

def test_ptp_periods():

    t = np.linspace(0.,10.,250)
    for T in np.linspace(0.1,2.,10):
        x = np.cos(2*np.pi*t/T)
        T_pred = ptp_periods(t, x)
        np.testing.assert_allclose(T, T_pred, rtol=1E-2)

    # check for failure if period is too long
    T = 5.
    x = np.cos(2*np.pi*t/T)
    T_pred = ptp_periods(t, x)
    assert np.isnan(T_pred)

def test_estimate_max_period():
    pass

def test_estimate_dt_nsteps():
    pass
