# coding: utf-8

""" Utilities for running frequency mapping with MPI (map) """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy import log as logger
import gary.dynamics as gd
import gary.integrate as gi
import gary.potential as gp

# project
from .. import ETOL
from ..freqmap import estimate_dt_nsteps
from .mmap_util import dtype

__all__ = ['worker', 'parser_arguments']

parser_arguments = list()

# list of [args, kwargs]
parser_arguments.append([('--nperiods',), dict(dest='nperiods', default=250, type=int,
                                               help='Number of periods to integrate for.')])
parser_arguments.append([('--nsteps_per_period',), dict(dest='nsteps_per_period', default=250, type=int,
                                                        help='Number of steps to take per min. period.')])

def worker(task):

    # unpack input argument dictionary
    index = task['index']
    w0_filename = task['w0_filename']
    alllyap_filename = task['cache_filename']
    potential = gp.load(task['potential_filename'])

    # if these aren't set, we'll need to estimate them
    dt = task.get('dt',None)
    nsteps = task.get('nsteps',None)

    # if these aren't set, assume defaults
    nperiods = task['nperiods']
    nsteps_per_period = task['nsteps_per_period']

    # read out just this initial condition
    w0 = np.load(w0_filename)
    norbits = len(w0)
    all_lyap = np.memmap(alllyap_filename, mode='r',
                         shape=(norbits,), dtype=dtype)

    # short-circuit if this orbit is already done
    if all_lyap['status'][index] == 1:
        return

    # automatically estimate dt, nsteps
    if dt is None or nsteps is None:
        try:
            dt, nsteps = estimate_dt_nsteps(potential, w0[index].copy(),
                                            nperiods, nsteps_per_period)
        except RuntimeError:
            logger.warning("Failed to integrate orbit when estimating dt,nsteps")
            alllyap = np.memmap(alllyap_filename, mode='r+',
                                shape=(norbits,), dtype=dtype)
            alllyap['status'][index] = 2
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
        alllyap = np.memmap(alllyap_filename, mode='r+',
                            shape=(norbits,), dtype=dtype)
        alllyap['status'][index] = 2
        alllyap.flush()
        return

    alllyap = np.memmap(alllyap_filename, mode='r+',
                        shape=(norbits,), dtype=dtype)
    alllyap['lyap_exp'][index] = LEs[-1].max()
    alllyap['status'][index] = 1
    alllyap.flush()

    return
