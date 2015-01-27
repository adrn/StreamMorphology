# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy import log as logger
import numpy as np
from scipy.signal import argrelmax, argrelmin

# Project
import gary.dynamics as gd
import gary.integrate as gi

__all__ = ['ptp_periods', 'orbit_to_freqs']

def ptp_periods(t, *coords):
    """
    Estimate the dominant periods of given coordinates by
    computing the mean peak-to-peak time.

    Parameters
    ----------
    t : array_like
        Array of times. Must have same shape as individual coordinates.
    *coords
        Positional arguments allow specifying coordinates to compute the
        periods of. For example, these could simply be x,y,z, or
        could be R,phi,z for cylindrical coordinates.

    """

    freqs = []
    for x in coords:
        # first compute mean peak-to-peak
        ix = argrelmax(x, mode='wrap')[0]
        f_max = np.mean(t[ix[1:]] - t[ix[:-1]])

        # now compute mean trough-to-trough
        ix = argrelmin(x, mode='wrap')[0]
        f_min = np.mean(t[ix[1:]] - t[ix[:-1]])

        # then take the mean of these two
        freqs.append(np.mean([f_max, f_min]))

    return np.array(freqs)

def estimate_max_period(t, w):
    """
    Given an array of times and an orbit, estimate the longest period
    in the orbit. We will then use this to figure out how long to
    integrate for when frequency mapping.

    Parameters
    ----------
    t : array_like
        Array of times.
    w : array_like
        Single orbit -- should have shape (len(t), 6)

    """

    # circulation
    circ = gd.classify_orbit(w)

    if np.any(circ):  # TUBE ORBIT - pass R,φ,z
        # flip coords
        new_w = gd.align_circulation_with_z(w, circ[0])[:,0]

        # convert to cylindrical
        R = np.sqrt(new_w[:,0]**2 + new_w[:,1]**2)
        phi = np.arctan2(new_w[:,1], new_w[:,0])
        z = new_w[:,2]

        T = ptp_periods(t, R, phi, z)

    else:  # BOX ORBIT - pass x,y,z
        T = ptp_periods(t, *w[:,:3].T)

    return T

def orbit_to_freqs(t, w, force_box=False, **kwargs):
    """
    Compute the fundamental frequencies of an orbit, ``w``. If not forced, this
    function tries to figure out whether the input orbit is a tube or box orbit and
    then uses the appropriate set of coordinates (Poincaré polar coordinates for tube,
    ordinary Cartesian for box). Any extra keyword arguments (``kwargs``) are passed
    to `NAFF.find_fundamental_frequencies`.

    Parameters
    ----------
    t : array_like
        Array of times.
    w : array_like
        The orbit to analyze. Should have shape (len(t),6).
    force_box : bool (optional)
        Force the routine to assume the orbit is a box orbit. Default is ``False``.
    **kwargs
        Any extra keyword arguments are passed to `NAFF.find_fundamental_frequencies`.

    """

    if w.ndim == 3:
        # remove extra length-1 dimension (assumed to be axis=1)
        w = w[:,0]

    # now get other frequencies
    if force_box:
        is_tube = False
    else:
        circ = gd.classify_orbit(w)
        is_tube = np.any(circ)

    naff = gd.NAFF(t)

    if is_tube:
        # need to flip coordinates until circulation is around z axis
        new_ws = gd.align_circulation_with_z(w, circ[0])
        # TODO: does the above always return a 3D array?

        fs = gd.poincare_polar(new_ws[:,0])
        try:
            logger.info('Solving for Rφz frequencies...')
            fRphiz,d,ixes = naff.find_fundamental_frequencies(fs, **kwargs)
        except:
            fRphiz = np.ones(3)*np.nan

        freqs = fRphiz

    else:
        # first get x,y,z frequencies
        logger.info('Solving for XYZ frequencies...')
        fs = [(w[:,j] + 1j*w[:,j+3]) for j in range(3)]
        try:
            fxyz,d,ixes = naff.find_fundamental_frequencies(fs, **kwargs)
        except:
            fxyz = np.ones(3)*np.nan

        freqs = fxyz

    return freqs, is_tube

def estimate_dt_nsteps(potential, w0, nperiods=100):
    # integrate orbit
    t,ws = potential.integrate_orbit(w0, dt=2., nsteps=20000,
                                     Integrator=gi.DOPRI853Integrator)

    # estimate the maximum period
    max_T = estimate_max_period(t, ws).max()

    # arbitrarily choose the timestep...
    try:
        # 1000 steps per period
        dt = round(max_T / 1000, 2)

        # integrate for 250 times the max period
        nsteps = int(round(250 * max_T / dt, -4))
    except ValueError:
        dt = 0.5
        nsteps = 100000

    if dt == 0:
        dt = 0.01

    return dt, nsteps

