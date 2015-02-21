# coding: utf-8

""" Utilities for keeping track of big memmap'd files """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import numpy as np

__all__ = ['read_alllyap', 'dtype']

# define indices of columns -- need this for the memmap'd file
dtype = [('lyap_exp','f8'), ('status','i8')]

def read_alllyap(f, norbits=None):
    """
    Read the numpy memmap'd file containing results from a Lyapunov
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
