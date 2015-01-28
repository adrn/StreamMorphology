# coding: utf-8

""" Utilities for running frequency mapping with MPI (map) """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy import log as logger

# Project
import gary.dynamics as gd
import gary.integrate as gi
from .mmap_util import colmap, mmap_shape
from .core import estimate_dt_nsteps

__all__ = ['worker']

ETOL = 1E-9

def worker(task):

    # unpack input argument dictionary
    index = task['index']
    w0_filename = task['w0_filename']
    allfreqs_filename = task['allfreqs_filename']
    potential = task['potential']

    # if these aren't set, we'll need to estimate them
    dt = task.get('dt',None)
    nsteps = task.get('nsteps',None)

    # if these aren't set, assume defaults
    nperiods = task.get('nperiods',200)
    nsteps_per_period = task.get('nsteps_per_period',100)

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
            dt, nsteps = estimate_dt_nsteps(potential, w0[index].copy(),
                                            nperiods, nsteps_per_period)
        except RuntimeError:
            logger.warning("Failed to integrate orbit when estimating dt,nsteps")
            allfreqs = np.memmap(allfreqs_filename, mode='r+', shape=allfreqs_shape, dtype='float64')
            tmp[:,:] = np.nan
            allfreqs[index] = tmp
            allfreqs.flush()
            return

    logger.info("Orbit {}: initial dt={}, nsteps={}".format(index, dt, nsteps))

    maxiter = 3  # maximum number of times to refine integration step
    for i in range(maxiter+1):
        # integrate orbit
        try:
            t,ws = potential.integrate_orbit(w0[index].copy(), dt=dt, nsteps=nsteps,
                                             Integrator=gi.DOPRI853Integrator,
                                             Integrator_kwargs=dict(nsteps=8192,atol=1E-13))
        except RuntimeError:
            # ODE integration failed
            logger.warning("Orbit integration failed. Shrinking timestep to "
                           "dt={}".format(dt))
            dt /= 2.
            nsteps *= 2
            continue

        logger.debug('Orbit integrated')

        # check energy conservation for the orbit
        E = potential.total_energy(ws[:,0,:3].copy(), ws[:,0,3:].copy())
        dE = np.abs(E[1:] - E[0])
        dEmax = dE.max() / np.abs(E[0])

        if dEmax < ETOL:
            break

        nsteps *= 2
        dt /= 2.
        logger.debug("Refining orbit {} to: dt,nsteps=({},{}). Max. dE={}"
                     .format(index, dt, nsteps, dEmax))

    if dEmax > ETOL:
        allfreqs = np.memmap(allfreqs_filename, mode='r+', shape=allfreqs_shape, dtype='float64')
        tmp[:,:] = np.nan
        allfreqs[index] = tmp
        allfreqs.flush()
        return

    # start finding the frequencies -- do first half then second half
    freqs1,is_tube = gd.orbit_to_freqs(t[:nsteps//2+1], ws[:nsteps//2+1])
    freqs2,is_tube = gd.orbit_to_freqs(t[:nsteps//2+1], ws[nsteps//2:])

    # save to output array
    tmp[0,:3] = freqs1
    tmp[1,:3] = freqs2

    tmp[:,colmap['dE_max']] = dEmax
    tmp[:,colmap['is_tube']] = float(is_tube)
    tmp[:,colmap['dt']] = float(dt)
    tmp[:,colmap['nsteps']] = nsteps
    tmp[:,colmap['success']] = 1.

    allfreqs = np.memmap(allfreqs_filename, mode='r+', shape=allfreqs_shape, dtype='float64')
    allfreqs[index] = tmp
    allfreqs.flush()
