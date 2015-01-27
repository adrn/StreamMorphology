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
from .mmap_util import colmap, mmap_shape

__all__ = ['ws_to_freqs', 'worker']

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
    if w.ndim < 3:
        w = w[:,np.newaxis]

    norbits = w.shape[1]
    periods = []
    for i in range(norbits):
        loop = gd.classify_orbit(w[:,i])
        if np.any(loop):
            # flip coords
            new_w = gd.align_circulation_with_z(w[:,i], loop[0])[:,0]

            # convert to cylindrical
            R = np.sqrt(new_w[:,0]**2 + new_w[:,1]**2)
            phi = np.arctan2(new_w[:,1], new_w[:,0])
            z = new_w[:,2]

            T = 2*np.pi / ptp_freqs(t, R, phi, z)
        else:
            T = 2*np.pi / ptp_freqs(t, *w[:,i,:3].T)

        periods.append(T)

    return np.array(periods)

def ws_to_freqs(naff, ws, nintvec=15):
    # now get other frequencies
    loop = gd.classify_orbit(ws)
    is_loop = np.any(loop)

    if is_loop:
        fxyz = np.ones(3)*np.nan
        # need to flip coordinates until circulation is around z axis
        new_ws = gd.align_circulation_with_z(ws[:,0], loop[0])

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

    return np.append(fxyz, fRphiz), is_loop

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

def worker(task):
    # unpack input argument dictionary
    index = task['index']
    w0_filename = task['w0_filename']
    allfreqs_filename = task['allfreqs_filename']
    potential = task['potential']
    dt = task.get('dt',None)
    nsteps = task.get('nsteps',None)

    # read out just this initial condition
    w0 = np.load(w0_filename)
    allfreqs_shape = (len(w0),) + mmap_shape
    allfreqs = np.memmap(allfreqs_filename, mode='r', shape=allfreqs_shape, dtype='float64')

    # short-circuit if this orbit is already done
    if allfreqs[index,0,colmap['success']] == 1.:
        return

    # temporary array for results
    tmp = np.zeros(mmap_shape)

    # automatically estimate dt, nsteps
    if dt is None or nsteps is None:
        try:
            dt, nsteps = estimate_dt_nsteps(potential, w0[index].copy())
        except RuntimeError:
            logger.warning("Failed to integrate orbit when estimating dt,nsteps")
            allfreqs = np.memmap(allfreqs_filename, mode='r+', shape=allfreqs_shape, dtype='float64')
            tmp[:,:] = np.nan
            allfreqs[index] = tmp
            allfreqs.flush()
            return

    logger.info("Orbit {}: initial dt={}, nsteps={}".format(index, dt, nsteps))

    dEmax = 1.
    maxiter = 3  # maximum number of times to refine integration step
    for i in range(maxiter+1):
        # integrate orbit
        try:
            t,ws = potential.integrate_orbit(w0[index].copy(), dt=dt, nsteps=nsteps,
                                             Integrator=gi.DOPRI853Integrator,
                                             Integrator_kwargs=dict(nsteps=8192,atol=1E-14,rtol=1E-9))
        except RuntimeError:
            # ODE integration failed
            logger.warning("Orbit integration failed. Shrinking timestep to "
                           "dt={}".format(dt))
            dt /= 2.
            continue

        logger.debug('Orbit integrated')

        # check energy conservation for the orbit
        E = potential.total_energy(ws[:,0,:3].copy(), ws[:,0,3:].copy())
        dE = np.abs(E[1:] - E[0])
        dEmax = dE.max() / np.abs(E[0])

        if dEmax < 1E-9:
            break

        nsteps *= 2
        logger.debug("Refining orbit {} to: dt,nsteps=({},{}). Max. dE={}"
                     .format(index, dt, nsteps, dEmax))

    if dEmax > 1E-9:
        allfreqs = np.memmap(allfreqs_filename, mode='r+', shape=allfreqs_shape, dtype='float64')
        tmp[:,:] = np.nan
        allfreqs[index] = tmp
        allfreqs.flush()
        return

    # start finding the frequencies -- do first half then second half
    naff = gd.NAFF(t[:nsteps//2+1])
    freqs1,is_loop = ws_to_freqs(naff, ws[:nsteps//2+1])
    freqs2,is_loop = ws_to_freqs(naff, ws[nsteps//2:])

    # save to output array
    tmp[0,:6] = freqs1
    tmp[1,:6] = freqs2

    tmp[:,colmap['dEmax']] = dEmax
    tmp[:,colmap['loop']] = float(is_loop)
    tmp[:,colmap['dt']] = float(dt)
    tmp[:,colmap['nsteps']] = nsteps
    tmp[:,colmap['success']] = 1.

    allfreqs = np.memmap(allfreqs_filename, mode='r+', shape=allfreqs_shape, dtype='float64')
    allfreqs[index] = tmp
    allfreqs.flush()

