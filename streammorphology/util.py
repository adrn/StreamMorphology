# coding: utf-8

""" General utilities. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
import gary.dynamics as gd

__all__ = ['_validate_nd_array', 'estimate_dt_nsteps']

def _validate_nd_array(x, expected_ndim):
    # ensure we have a 1D array of initial conditions
    x = np.array(x)
    if x.ndim != expected_ndim:
        raise ValueError("Input array (or iterable) must be {0}D, not {1}D"
                         .format(expected_ndim, x.ndim))
    return x

def estimate_dt_nsteps(w0, potential, nperiods, nsteps_per_period, return_periods=False):
    """
    Estimate the timestep and number of steps to integrate for given a potential
    and set of initial conditions.

    Parameters
    ----------
    w0 : array_like
    potential : :class:`~gary.potential.Potential`
    nperiods : int
        Number of (max) periods to integrate.
    nsteps_per_period : int
        Number of steps to take per (max) orbital period.
    return_periods : bool (optional)
        Also return the estimated periods for the orbit.

    """

    # integrate orbit
    t,w = potential.integrate_orbit(w0, dt=0.2, nsteps=50000)

    # if loop, align circulation with Z and take R period
    loop = gd.classify_orbit(w[:,0])
    if np.any(loop):
        w = gd.align_circulation_with_z(w[:,0], loop)

        # convert to cylindrical coordinates
        R = np.sqrt(w[:,0]**2 + w[:,1]**2)
        phi = np.arctan2(w[:,1], w[:,0])
        z = w[:,2]

        T = np.array([gd.peak_to_peak_period(t, f) for f in [R, phi, z]])

    else:
        T = np.array([gd.peak_to_peak_period(t, f) for f in w.T[:3,0]])

    # timestep from number of steps per period
    dt = float(T.max()) / float(nsteps_per_period)
    nsteps = int(round(nperiods * T.max() / dt))

    if dt == 0.:
        raise ValueError("Timestep is zero!")

    if return_periods:
        return dt, nsteps, T
    else:
        return dt, nsteps
