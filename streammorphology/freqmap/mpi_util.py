# coding: utf-8

""" Utilities for running frequency mapping with MPI (map) """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
from collections import OrderedDict

# Third-party
import numpy as np
from astropy.utils import isiterable

# Project
# ...

__all__ = ['worker']

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
    freqs1,is_tube = ws_to_freqs(naff, ws[:nsteps//2+1])
    freqs2,is_tube = ws_to_freqs(naff, ws[nsteps//2:])

    # save to output array
    tmp[0,:6] = freqs1
    tmp[1,:6] = freqs2

    tmp[:,colmap['dEmax']] = dEmax
    tmp[:,colmap['loop']] = float(is_tube)
    tmp[:,colmap['dt']] = float(dt)
    tmp[:,colmap['nsteps']] = nsteps
    tmp[:,colmap['success']] = 1.

    allfreqs = np.memmap(allfreqs_filename, mode='r+', shape=allfreqs_shape, dtype='float64')
    allfreqs[index] = tmp
    allfreqs.flush()

