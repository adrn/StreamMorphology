# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np

# Project
import gary.dynamics as gd
import gary.integrate as gi

__all__ = ['estimate_periods', 'estimate_dt_nsteps']

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

        T = np.array([gd.peak_to_peak_period(t, f) for f in [R, phi, z]])

    else:  # BOX ORBIT - pass x,y,z
        T = np.array([gd.peak_to_peak_period(t, f) for f in w[:,:3].T])

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
