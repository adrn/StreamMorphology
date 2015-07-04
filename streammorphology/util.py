# coding: utf-8

""" General utilities. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
import gary.dynamics as gd
import gary.integrate as gi

__all__ = ['_validate_nd_array', 'estimate_dt_nsteps']

def _validate_nd_array(x, expected_ndim):
    # ensure we have a 1D array of initial conditions
    x = np.array(x)
    if x.ndim != expected_ndim:
        raise ValueError("Input array (or iterable) must be {0}D, not {1}D"
                         .format(expected_ndim, x.ndim))
    return x

def _autodetermine_initial_dt(w0, potential, dE_threshold=1E-9):
    if w0.ndim > 1:
        raise ValueError("Only one set of initial conditions may be passed in at a time. (w0.ndim == 1)")

    dts = np.logspace(-3, 1, 8)[::-1]
    _base_nsteps = 1000

    for dt in dts:
        nsteps = int(round(_base_nsteps / dt))
        t,w = potential.integrate_orbit(w0, dt=dt, nsteps=nsteps, Integrator=gi.DOPRI853Integrator)
        Ei = potential.total_energy(w[0,0,:3], w[0,0,3:])
        Ef = potential.total_energy(w[-1,0,:3], w[-1,0,3:])
        dE = np.abs((Ef - Ei) / Ei)

        if dE < dE_threshold:
            break

    return dt

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
    dt = _autodetermine_initial_dt(w0, potential, dE_threshold=1E-9)
    nsteps = int(round(10000 / dt))
    t,w = potential.integrate_orbit(w0, dt=dt, nsteps=nsteps)

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
    Tmax = T.max()
    if np.isnan(Tmax):
        T = T[np.isfinite(T)]
        Tmax = T.max()

    dt = float(Tmax) / float(nsteps_per_period)
    nsteps = int(round(nperiods * Tmax / dt))

    if dt == 0.:
        raise ValueError("Timestep is zero!")

    if return_periods:
        return dt, nsteps, T
    else:
        return dt, nsteps
