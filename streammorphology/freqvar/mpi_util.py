# coding: utf-8

""" Utilities for running frequency mapping with MPI (map) """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy import log as logger

# Project
import gary.coordinates as gc
import gary.dynamics as gd
import gary.integrate as gi
import gary.potential as gp
from gary.util import rolling_window

from ..freqmap import estimate_dt_nsteps
from .mmap_util import dtype
from .. import ETOL

__all__ = ['worker', 'parser_arguments']

parser_arguments = list()

# list of [args, kwargs]
parser_arguments.append([('--nperiods',), dict(dest='nperiods', default=1000, type=int,
                                               help='Number of periods to integrate for.')])
parser_arguments.append([('--nsteps_per_period',), dict(dest='nsteps_per_period', default=256, type=int,
                                                        help='Number of steps to take per min. period.')])
parser_arguments.append([('--hammingp',), dict(dest='hammingp', default=4, type=int,
                                               help='Power of Hamming filter.')])
parser_arguments.append([('--window_width',), dict(dest='window_width', default=50, type=int,
                                                   help='Width of the window in num. of periods.')])
parser_arguments.append([('--window_stride',), dict(dest='window_stride', default=1, type=float,
                                                    help='Stride for sliding the window in num. of periods.')])

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

    # the order of the Hamming filter
    p = task['hammingp']

    # read out just this initial condition
    w0 = np.load(w0_filename)
    norbits = len(w0)
    data = np.memmap(allfreqs_filename, mode='r',
                     shape=(norbits,), dtype=dtype)

    # short-circuit if this orbit is already done
    if data['success'][index]:
        logger.debug("Orbit {0} already successfully completed.".format(index))
        return None

    # container for return
    result = dict()
    result['mmap_filename'] = allfreqs_filename
    result['norbits'] = norbits
    result['index'] = index
    result['dtype'] = dtype

    # automatically estimate dt, nsteps
    if dt is None or nsteps is None:
        try:
            dt, nsteps = estimate_dt_nsteps(w0[index].copy(), potential,
                                            nperiods, nsteps_per_period)
        except RuntimeError:
            logger.warning("Failed to integrate orbit when estimating dt,nsteps")
            result['freqs'] = np.nan*data['freqs'][index]
            result['success'] = False
            result['error_code'] = 1
            return result

    logger.info("Orbit {}: dt={}, nsteps={}".format(index, dt, nsteps))

    # integrate orbit
    logger.debug("Integrating orbit...")
    try:
        t,ws = potential.integrate_orbit(w0[index].copy(), dt=dt, nsteps=nsteps,
                                         Integrator=gi.DOPRI853Integrator,
                                         Integrator_kwargs=dict(atol=1E-10))
    except RuntimeError:  # ODE integration failed
        logger.warning("Orbit integration failed.")
        dEmax = 1E10
    else:
        logger.debug('Orbit integrated successfully, checking energy conservation...')

        # check energy conservation for the orbit
        E = potential.total_energy(ws[:,0,:3].copy(), ws[:,0,3:].copy())
        dE = np.abs(E[1:] - E[0])
        dEmax = dE.max() / np.abs(E[0])
        logger.debug('max(∆E) = {0:.2e}'.format(dEmax))

    if dEmax > ETOL:
        logger.warning("Failed due to energy conservation check.")
        result['freqs'] = np.nan*data['freqs'][index]
        result['success'] = False
        result['error_code'] = 2
        return result

    # windowing properties - convert from period to steps
    window_width = int(task['window_width'] * nsteps_per_period)
    window_stride = int(task['window_stride'] * nsteps_per_period)

    # classify orbit full orbit
    circ = gd.classify_orbit(ws[:,0])
    is_tube = np.any(circ)

    logger.debug("NAFFing the windows:")

    allfreqs = []
    for (i1,i2),ww in rolling_window(ws[:,0], window_size=window_width, stride=window_stride, return_idx=True):
        if i2 >= nsteps:
            break

        logger.debug("Window: {0}:{1}".format(i1,i2))
        if is_tube:
            # need to flip coordinates until circulation is around z axis
            new_ws = gd.align_circulation_with_z(ww, circ)
            new_ws = gc.cartesian_to_poincare_polar(new_ws)
        else:
            new_ws = ww

        fs = [(new_ws[:,j] + 1j*new_ws[:,j+3]) for j in range(3)]
        naff = gd.NAFF(t[i1:i2], p=p)

        try:
            freqs,d,ixs = naff.find_fundamental_frequencies(fs, nintvec=5)
        except:
            result['freqs'] = np.nan*data['freqs'][index]
            result['success'] = False
            result['error_code'] = 3
            return result

        allfreqs.append(freqs.tolist())
    allfreqs = np.array(allfreqs)

    result['freqs'] = np.mean(allfreqs, axis=0)
    result['freq_std'] = np.std(allfreqs, axis=0)
    logger.debug("Frequencies: {0}".format(result['freqs']))
    logger.debug("Freq std dev: {0}".format(result['freq_std']))
    logger.debug("Dimensionless freq spread: {0}".format(result['freq_std']/result['freqs']))

    result['dE_max'] = dEmax
    result['dt'] = float(dt)
    result['nsteps'] = nsteps
    result['max_amp_freq_ix'] = d['|A|'][ixs].argmax()
    result['success'] = True
    result['error_code'] = 0
    return result
