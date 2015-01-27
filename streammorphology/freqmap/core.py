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

__all__ = ['ptp_periods', 'ws_to_freqs']

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
    Given an array of times and orbits, estimate the longest period
    in the orbit. We will then use this to figure out how long to
    integrate for when frequency mapping.

    Parameters
    ----------
    t : array_like
        Array of times.
    w : array_like
        Orbit(s).

    """

    # if only a single orbit,
    if w.ndim < 3:
        w = w[:,np.newaxis]

    norbits = w.shape[1]
    periods = []
    for i in range(norbits):
        circ = gd.classify_orbit(w[:,i])

        if np.any(circ):  # TUBE ORBIT - pass R,φ,z
            # flip coords
            new_w = gd.align_circulation_with_z(w[:,i], circ[0])[:,0]

            # convert to cylindrical
            R = np.sqrt(new_w[:,0]**2 + new_w[:,1]**2)
            phi = np.arctan2(new_w[:,1], new_w[:,0])
            z = new_w[:,2]

            T = ptp_periods(t, R, phi, z)

        else:  # BOX ORBIT - pass x,y,z
            T = ptp_periods(t, *w[:,i,:3].T)

        periods.append(T)

    return np.array(periods)

def ws_to_freqs(naff, ws, nintvec=15):
    # now get other frequencies
    circ = gd.classify_orbit(ws)
    is_tube = np.any(circ)

    if is_tube:
        fxyz = np.ones(3)*np.nan
        # need to flip coordinates until circulation is around z axis
        new_ws = gd.align_circulation_with_z(ws[:,0], circ[0])

        fs = gd.poincare_polar(new_ws[:,0])
        try:
            logger.info('Solving for Rφz frequencies...')
            fRphiz,d,ixes = naff.find_fundamental_frequencies(fs, nintvec=nintvec)
        except:
            fRphiz = np.ones(3)*np.nan

    else:
        fRphiz = np.ones(3)*np.nan

        # first get x,y,z frequencies
        logger.info('Solving for XYZ frequencies...')
        fs = [(ws[:,0,j] + 1j*ws[:,0,j+3]) for j in range(3)]
        try:
            fxyz,d,ixes = naff.find_fundamental_frequencies(fs, nintvec=nintvec)
        except:
            fxyz = np.ones(3)*np.nan

    return np.append(fxyz, fRphiz), is_tube

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

