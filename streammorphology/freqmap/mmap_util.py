# coding: utf-8

""" Utilities for keeping track of big memmap'd freqmap files """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
from collections import OrderedDict

# Third-party
import numpy as np
from astropy.utils import isiterable

# define indices of columns -- need this for the memmap'd file
colmap = OrderedDict(freqs=(0,1,2), dE_max=3, success=4, is_tube=5, dt=6, nsteps=7, max_amp_freq_ix=8)
l = np.concatenate([[x] if not isiterable(x) else list(x) for x in colmap.values()]).max()+1
mmap_shape = (2, l)

__all__ = ['read_allfreqs', 'mmap_shape']

def read_allfreqs(f, norbits=None):
    """
    Read the numpy memmap'd file containing results from a frequency
    mapping. This function returns a numpy structured array with named
    columns and proper data types.

    Parameters
    ----------
    f : str
        The path to a file containing the frequency mapping results.
    norbits : int (optional)
        Number of orbits, e.g., the length of the first axis. Needed to
        properly read in the memmap file. If not specified, will attempt
        to read this from a initial conditions file (``w0.npy``) located
        in the same directory as the allfreqs file. This could produce
        insane results if the files don't match!
    """
    if norbits is None:
        w0_filename = os.path.join(os.path.split(f)[0], 'w0.npy')
        w0 = np.load(w0_filename)
        norbits = len(w0)

    allfreqs_shape = (norbits,) + mmap_shape

    # first get the memmap array
    allfreqs = np.memmap(f, mode='r', shape=allfreqs_shape, dtype='float64').copy()

    # replace NAN nsteps with 0
    allfreqs[np.isnan(allfreqs[:,0,colmap['nsteps']]),0,colmap['nsteps']] = 0
    allfreqs[np.isnan(allfreqs[:,0,colmap['max_amp_freq_ix']]),0,colmap['max_amp_freq_ix']] = 0
    dtype = [('freqs','f8',(2,3)), ('dE_max','f8'), ('success','b1'),
             ('is_tube','b1'), ('dt','f8'), ('nsteps','i8'), ('max_amp_freq_ix','i8')]
    data = [(allfreqs[i,:,:3],) + tuple(allfreqs[i,0,3:]) for i in range(norbits)]
    return np.array(data, dtype=dtype)
