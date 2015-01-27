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

