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
import gary.potential as gp
from .core import estimate_dt_nsteps

__all__ = ['worker']

ETOL = 1E-7
def worker(task):

    # unpack input argument dictionary
    index = task['index']
    w0_filename = task['w0_filename']
    alllyap_filename = task['alllyap_filename']
    potential = gp.load(task['potential_filename'])

    # if these aren't set, we'll need to estimate them
    dt = task.get('dt',None)
    nsteps = task.get('nsteps',None)

    # if these aren't set, assume defaults
    nperiods = task.get('nperiods',250)
    nsteps_per_period = task.get('nsteps_per_period',250)

    # read out just this initial condition
    w0 = np.load(w0_filename)
    alllyap_shape = (len(w0),2)
    alllyap = np.memmap(alllyap_filename, mode='r', shape=alllyap_shape, dtype='float64')

    # short-circuit if this orbit is already done
    if alllyap[index,1] == 1.:
        return

    # automatically estimate dt, nsteps
    if dt is None or nsteps is None:
        try:
            dt, nsteps = estimate_dt_nsteps(potential, w0[index].copy(),
                                            nperiods, nsteps_per_period)
        except RuntimeError:
            logger.warning("Failed to integrate orbit when estimating dt,nsteps")
            alllyap = np.memmap(alllyap_filename, mode='r+', shape=alllyap_shape, dtype='float64')
            alllyap[index] = np.array([np.nan, 0.])
            alllyap.flush()
            return

    logger.info("Orbit {}: initial dt={}, nsteps={}".format(index, dt, nsteps))

    def F_max(t,w):
        x,y,z,px,py,pz = w.T
        term1 = np.array([px, py, pz]).T
        term2 = potential.acceleration(w[...,:3])
        return np.hstack((term1,term2))
    integrator = gi.DOPRI853Integrator(F_max)

    maxiter = 3  # maximum number of times to refine integration step
    for i in range(maxiter+1):
        # integrate orbit
        logger.debug("Iteration {} -- integrating orbit...".format(i+1))
        try:
            lyap = gd.lyapunov_max(w0[index].copy(), integrator, dt=dt, nsteps=nsteps)

        except RuntimeError:  # ODE integration failed
            dt /= 2.
            nsteps *= 2
            logger.warning("Orbit integration failed. Shrinking timestep to "
                           "dt={}".format(dt))
            continue

        LEs,ts,ws = lyap
        logger.debug('Orbit integrated successfully, checking energy conservation...')

        # check energy conservation for the orbit
        E = potential.total_energy(ws[:,:3].copy(), ws[:,3:].copy())
        dE = np.abs(E[1:] - E[0])
        dEmax = dE.max() / np.abs(E[0])

        logger.debug('max(∆E) = {0:.2e}'.format(dEmax))
        if dEmax < ETOL:
            break

        nsteps *= 2
        dt /= 2.
        logger.debug("Refining orbit {} to: dt,nsteps=({},{}). Max. dE={}"
                     .format(index, dt, nsteps, dEmax))

    if dEmax > ETOL:
        alllyap = np.memmap(alllyap_filename, mode='r+', shape=alllyap_shape, dtype='float64')
        alllyap[index] = np.array([np.nan, 0.])
        alllyap.flush()
        return

    alllyap = np.memmap(alllyap_filename, mode='r+', shape=alllyap_shape, dtype='float64')
    alllyap[index] = np.array([LEs[-1].max(), 1.])
    alllyap.flush()

    return
