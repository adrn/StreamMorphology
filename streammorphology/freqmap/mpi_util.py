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

    # the order of the Hamming filter
    p = task.get('p', 2)

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
            allfreqs['error_code'][index] = 0
            allfreqs.flush()
            return

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
        allfreqs = np.memmap(allfreqs_filename, mode='r+',
                             shape=(norbits,), dtype=dtype)
        allfreqs['freqs'][index] = np.nan
        allfreqs['success'][index] = False
        allfreqs['error_code'][index] = 1
        allfreqs.flush()
        return

    # start finding the frequencies -- do first half then second half
    naff1 = gd.NAFF(t[:nsteps//2+1], p=p)
    naff2 = gd.NAFF(t[nsteps//2:], p=p)

    # classify orbit full orbit
    circ = gd.classify_orbit(ws)
    is_tube = np.any(circ)

    # define slices for first and second parts
    sl1 = slice(None,nsteps//2+1)
    sl2 = slice(nsteps//2,None)

    if is_tube:
        # need to flip coordinates until circulation is around z axis
        new_ws = gd.align_circulation_with_z(ws, circ)
        new_ws = gc.poincare_polar(new_ws)
        fs1 = [(new_ws[sl1,j] + 1j*new_ws[sl1,j+3]) for j in range(3)]
        fs2 = [(new_ws[sl2,j] + 1j*new_ws[sl2,j+3]) for j in range(3)]
    else:  # box
        fs1 = [(ws[sl1,0,j] + 1j*ws[sl1,0,j+3]) for j in range(3)]
        fs2 = [(ws[sl2,0,j] + 1j*ws[sl2,0,j+3]) for j in range(3)]

    # freqs1,d1,ixs1,is_tube = gd.orbit_to_freqs(t[:nsteps//2+1], ws[:nsteps//2+1])
    # freqs2,d2,ixs2,is_tube = gd.orbit_to_freqs(t[:nsteps//2+1], ws[nsteps//2:])
    try:
        freqs1,d1,ixs1 = naff1.find_fundamental_frequencies(fs1, nintvec=8)
        freqs2,d2,ixs2 = naff2.find_fundamental_frequencies(fs2, nintvec=8)
    except:
        allfreqs = np.memmap(allfreqs_filename, mode='r+',
                             shape=(norbits,), dtype=dtype)
        allfreqs['freqs'][index] = np.nan
        allfreqs['success'][index] = False
        allfreqs['error_code'][index] = 2
        allfreqs.flush()
        return

    max_amp_freq_ix = d1['|A|'][ixs1].argmax()

    # save to output array
    allfreqs = np.memmap(allfreqs_filename, mode='r+',
                         shape=(norbits,), dtype=dtype)
    allfreqs['freqs'][index][0] = freqs1
    allfreqs['freqs'][index][1] = freqs2
    allfreqs['dE_max'][index] = dEmax
    allfreqs['is_tube'][index] = float(is_tube)
    allfreqs['dt'][index] = float(dt)
    allfreqs['nsteps'][index] = nsteps
    allfreqs['max_amp_freq_ix'][index] = max_amp_freq_ix
    allfreqs['success'][index] = True
    allfreqs.flush()
