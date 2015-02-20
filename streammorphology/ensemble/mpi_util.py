# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import gary.potential as gp
import numpy as np

# Project
from .. import ETOL

__all__ = ['worker']

def worker(task):

    # unpack input argument dictionary
    index = task['index']
    w0_filename = task['w0_filename']
    filename = task['cache_filename']
    potential = gp.load(task['potential_filename'])

    # if these aren't set, we'll need to estimate them
    dt = task.get('dt',None)
    nsteps = task.get('nsteps',None)

    # if these aren't set, assume defaults
    nperiods = task.get('nperiods',200)
    nsteps_per_period = task.get('nsteps_per_period',500)

    # read out just this initial condition
    w0 = np.load(w0_filename)
    shape = (len(w0),2)
    all_kld = np.memmap(filename, mode='r', shape=shape, dtype='float64')
