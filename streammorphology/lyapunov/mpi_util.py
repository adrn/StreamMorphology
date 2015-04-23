# coding: utf-8

""" Utilities for running frequency mapping with MPI (map) """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy import log as logger
import gary.dynamics as gd
import gary.potential as gp

# project
from .. import ETOL
from ..freqmap import estimate_dt_nsteps
from .mmap_util import dtype

__all__ = ['worker', 'parser_arguments']

parser_arguments = list()

# list of [args, kwargs]
parser_arguments.append([('--nperiods',), dict(dest='nperiods', default=1000, type=int,
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
    if all_lyap['success'][index]:
        return

    # container for return
    result = dict()
    result['mmap_filename'] = alllyap_filename
    result['norbits'] = norbits
    result['index'] = index
    result['dtype'] = dtype

    # automatically estimate dt, nsteps
    if dt is None or nsteps is None:
        try:
            dt, nsteps = estimate_dt_nsteps(potential, w0[index].copy(),
                                            nperiods, nsteps_per_period)
        except RuntimeError:
            logger.warning("Failed to integrate orbit when estimating dt,nsteps")
            result['lyap_exp'] = np.nan
            result['success'] = False
            result['error_code'] = 1
            return result

    logger.info("Orbit {}: initial dt={}, nsteps={}".format(index, dt, nsteps))

    # integrate orbit
    logger.debug("Integrating orbit / computing Lyapunov exponent...")
    try:
        # lyap = gd.lyapunov_max(w0[index].copy(), integrator, dt=dt, nsteps=nsteps)
        lyap = gd.fast_lyapunov_max(w0[index].copy(), potential, dt=dt, nsteps=nsteps)

    except RuntimeError:  # ODE integration failed
        logger.warning("Orbit integration failed.")
        dEmax = 1E10
    else:
        logger.debug('Orbit integrated successfully, checking energy conservation...')

        # unpack lyap
        LEs,ts,ws = lyap

        # check energy conservation for the orbit
        E = potential.total_energy(ws[:,0,:3].copy(), ws[:,0,3:].copy())
        dE = np.abs(E[1:] - E[0])
        dEmax = dE.max() / np.abs(E[0])
        logger.debug('max(∆E) = {0:.2e}'.format(dEmax))

    if dEmax > ETOL:
        result['lyap_exp'] = np.nan
        result['success'] = False
        result['error_code'] = 2
        return result

    result['lyap_exp'] = np.mean(LEs[-500:])
    result['success'] = True
    result['error_code'] = 0

    return result
