# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

# Project
import gary.potential as gp
from gary.units import galactic
from ..mpi_util import worker
from ..mmap_util import dtype, read_allfreqs

plot_path = "output/tests/freqmap"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

def test_worker():

    tmp_w0 = "/tmp/w0.npy"
    tmp_pot = "/tmp/potential.yml"
    tmp_allfreqs = "/tmp/allfreqs.dat"

    # create a test initial conditions and potential
    potential = gp.IsochronePotential(m=1E11, b=2.5, units=galactic)
    w0 = np.array([[10., 0., 0.075, 0., 0.12, 0.02]])
    gp.save(potential, tmp_pot)

    if os.path.exists(tmp_w0):
        os.remove(tmp_w0)

    if os.path.exists(tmp_allfreqs):
        os.remove(tmp_allfreqs)

    np.save(tmp_w0, w0)

    # create a test allfreqs file
    allfreqs = np.memmap(tmp_allfreqs, mode='w+',
                         shape=(len(w0),), dtype=dtype)

    task = dict()
    task['index'] = 0
    task['w0_filename'] = tmp_w0
    task['potential_filename'] = tmp_pot
    task['cache_filename'] = tmp_allfreqs
    worker(task)

    af = read_allfreqs(tmp_allfreqs, norbits=1)
    print(af)
