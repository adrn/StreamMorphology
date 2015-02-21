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
from .mmap_util import dtype
from .core import estimate_dt_nsteps
from .. import ETOL

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
    allfreqs_filename = task['cache_filename']
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
    allfreqs = np.memmap(allfreqs_filename, mode='r',
                         shape=(norbits,), dtype=dtype)

    # short-circuit if this orbit is already done
    if allfreqs['success'][index]:
        return

    # automatically estimate dt, nsteps
    if dt is None or nsteps is None:
        try:
            dt, nsteps = estimate_dt_nsteps(potential, w0[index].copy(),
                                            nperiods, nsteps_per_period)
        except RuntimeError:
            logger.warning("Failed to integrate orbit when estimating dt,nsteps")
            allfreqs = np.memmap(allfreqs_filename, mode='r+',
                                 shape=(norbits,), dtype=dtype)
            allfreqs['freqs'][index] = np.nan
            allfreqs['success'][index] = False
            allfreqs.flush()
            return

    logger.info("Orbit {}: initial dt={}, nsteps={}".format(index, dt, nsteps))

    # TODO: do I instead want to refine the internal number of steps?
    maxiter = 3  # maximum number of times to refine integration step
    for i in range(maxiter+1):
        # integrate orbit
        logger.debug("Iteration {} -- integrating orbit...".format(i+1))
        try:
            t,ws = potential.integrate_orbit(w0[index].copy(), dt=dt, nsteps=nsteps,
                                             Integrator=gi.DOPRI853Integrator,
                                             Integrator_kwargs=dict(nsteps=4096,atol=1E-8))
        except RuntimeError:  # ODE integration failed
            dt /= 2.
            nsteps *= 2
            logger.warning("Orbit integration failed. Shrinking timestep to "
                           "dt={}".format(dt))
            continue

        logger.debug('Orbit integrated successfully, checking energy conservation...')

        # check energy conservation for the orbit
        E = potential.total_energy(ws[:,0,:3].copy(), ws[:,0,3:].copy())
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
        allfreqs = np.memmap(allfreqs_filename, mode='r+',
                             shape=(norbits,), dtype=dtype)
        allfreqs['freqs'][index] = np.nan
        allfreqs['success'][index] = False
        allfreqs.flush()
        return

    # start finding the frequencies -- do first half then second half
    freqs1,d1,ixs1,is_tube = gd.orbit_to_freqs(t[:nsteps//2+1], ws[:nsteps//2+1])
    freqs2,d2,ixs2,is_tube = gd.orbit_to_freqs(t[:nsteps//2+1], ws[nsteps//2:])

    max_amp_freq_ix = d1['|A|'][ixs1].argmax()

    # save to output array
    allfreqs = np.memmap(allfreqs_filename, mode='r+',
                         shape=(norbits,), dtype=dtype)
    allfreqs['freqs'][index][0] = freqs1
    allfreqs['freqs'][index][1] = freqs1
    allfreqs['dE_max'][index] = dEmax
    allfreqs['is_tube'][index] = float(is_tube)
    allfreqs['dt'][index] = float(dt)
    allfreqs['nsteps'][index] = nsteps
    allfreqs['max_amp_freq_ix'][index] = max_amp_freq_ix
    allfreqs['success'][index] = True
    allfreqs.flush()
