# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from scipy.signal import argrelmax, argrelmin

# Project
import gary.dynamics as gd
import gary.integrate as gi

__all__ = ['ptp_periods', 'estimate_periods', 'estimate_dt_nsteps']

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
        # find peaks
        max_ix = argrelmax(x, mode='wrap')[0]
        max_ix = max_ix[(max_ix != 0) & (max_ix != (len(x)-1))]

        # find troughs
        min_ix = argrelmin(x, mode='wrap')[0]
        min_ix = min_ix[(min_ix != 0) & (min_ix != (len(x)-1))]

        # neglect minor oscillations
        tol = 1E-2
        if abs(np.mean(x[max_ix]) - np.mean(x[min_ix])) < tol:
            freqs.append(np.nan)
            continue

        # first compute mean peak-to-peak

        if len(max_ix) > 0:
            f_max = np.mean(t[max_ix[1:]] - t[max_ix[:-1]])
        else:
            f_max = np.nan

        # now compute mean trough-to-trough
        if len(min_ix) > 0:
            f_min = np.mean(t[min_ix[1:]] - t[min_ix[:-1]])
        else:
            f_min = np.nan

        # then take the mean of these two
        freqs.append(np.mean([f_max, f_min]))

    return np.array(freqs)

def estimate_periods(t, w, min=False):
    """
    Given an array of times and an orbit, estimate the periods of
    present in the orbit. We will then use this to figure out how
    long to integrate for when, e.g., frequency mapping.

    Parameters
    ----------
    t : array_like
        Array of times.
    w : array_like
        Single orbit -- should have shape (len(t), 6)

    Returns
    -------
    periods : :class:`numpy.ndarray`
        The periods found from the time series.

    """

    if w.ndim == 3:
        w = w[:,0]

    # circulation
    circ = gd.classify_orbit(w)

    if np.any(circ):  # TUBE ORBIT - pass R,φ,z
        # flip coords
        new_w = gd.align_circulation_with_z(w, circ)

        # convert to cylindrical
        R = np.sqrt(new_w[:,0]**2 + new_w[:,1]**2)
        phi = np.arctan2(new_w[:,1], new_w[:,0])
        z = new_w[:,2]

        T = ptp_periods(t, R, phi, z)

    else:  # BOX ORBIT - pass x,y,z
        T = ptp_periods(t, *w[:,:3].T)

    if np.any(np.isfinite(T)):
        return np.array(T[np.isfinite(T)])

    else:
        return np.array([np.nan]*w.shape[-1])

def estimate_dt_nsteps(potential, w0, nperiods, nsteps_per_period, return_periods=False):
    """
    Estimate the timestep and number of steps to integrate for given a potential
    and set of initial conditions.

    Parameters
    ----------
    potential : :class:`~gary.potential.Potential`
    w0 : array_like
    nperiods : int
        Number of (max) periods to integrate.
    nsteps_per_period : int
        Number of steps to take per (max) orbital period.
    return_periods : bool (optional)
        Also return the estimated periods for the orbit.

    """

    # integrate orbit
    t,w = potential.integrate_orbit(w0, dt=2., nsteps=25000,
                                    Integrator=gi.DOPRI853Integrator)

    # estimate the maximum period
    T = estimate_periods(t, w)

    # timestep from number of steps per period
    dt = float(T.min()) / float(nsteps_per_period)
    nsteps = int(round(nperiods * T.max() / dt, -4))

    if dt == 0.:
        raise ValueError("Timestep is zero!")

    if return_periods:
        return dt, nsteps, T
    else:
        return dt, nsteps
