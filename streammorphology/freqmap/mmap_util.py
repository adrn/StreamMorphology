# coding: utf-8

""" Utilities for keeping track of big memmap'd freqmap files """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import numpy as np

__all__ = ['read', 'dtype', 'error_codes']

# define indices of columns -- need this for the memmap'd file
dtype = [('freqs','f8',(2,3)), ('dE_max','f8'), ('success','b1'),
         ('is_tube','b1'), ('dt','f8'), ('nsteps','i8'),
         ('max_amp_freq_ix','i8'), ('error_code','i8')]

error_codes = {1: "Failed to integrate orbit or estimate dt, nsteps.",
               2: "Energy conservation criteria not met.",
               3: "NAFF failed on find_fundamental_frequencies()."}

def read(f, norbits=None):
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

    f = os.path.abspath(f)
    if norbits is None:
        w0_filename = os.path.join(os.path.split(f)[0], 'w0.npy')
        w0 = np.load(w0_filename)
        norbits = len(w0)

    # first get the memmap array
    allfreqs = np.memmap(f, mode='r', shape=(norbits,), dtype=dtype)

    return allfreqs
