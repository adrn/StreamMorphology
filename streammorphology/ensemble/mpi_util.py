# coding: utf-8

#TODO: Modify ensemble mapping to use adaptive bandwidth, only compute at end of integration

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import sys

# Third-party
from astropy import log as logger
import gary.potential as gp
import numpy as np
from sklearn.neighbors import KernelDensity

# Project
from .. import ETOL
from .mmap_util import dtype
from .core import create_ensemble, do_the_kld, default_metrics

# TODO: replace prepare_parent_orbit

__all__ = ['worker', 'parser_arguments']

parser_arguments = list()

# list of [args, kwargs]
parser_arguments.append([('--nensemble',),
                         dict(dest='nensemble', required=True,
                              type=int, help='Number of orbits per ensemble.')])
parser_arguments.append([('--mscale',),
                         dict(dest='mscale', default=1E4,
                              type=float, help='Mass scale of ensemble.')])
parser_arguments.append([('--bandwidth',),
                         dict(dest='kde_bandwidth', default=10.,
                              type=float, help='KDE kernel bandwidth.')])
parser_arguments.append([('--nkld',),
                         dict(dest='nkld', default=256, type=int,
                              help='Number of times to evaluate the KLD over the integration '
                                   'of the ensemble.')])
parser_arguments.append([('--nperiods',),
                         dict(dest='nperiods', default=256, type=int,
                              help='Number of periods to integrate for, in units of the parent '
                                   'orbit periods.')])
parser_arguments.append([('--nsteps_per_period',),
                         dict(dest='nsteps_per_period', default=250, type=int,
                              help='Number of steps to take per min. period.')])

def worker(task):

    # unpack input argument dictionary
    index = task['index']
    w0_filename = task['w0_filename']
    filename = task['cache_filename']
    potential = gp.load(task['potential_filename'])
    nensemble = task['nensemble']

    # mass scale
    mscale = task['mscale']

    # KDE kernel bandwidth
    bw = task['kde_bandwidth']

    # number of times to compute the KLD
    nkld = int(task['nkld'])

    # number of periods to integrate the ensemble
    nperiods = task['nperiods']
    nsteps_per_period = task['nsteps_per_period']

    # read out just this initial condition
    w0 = np.load(w0_filename)
    norbits = len(w0)
    this_w0 = w0[index].copy()
    all_kld = np.memmap(filename, mode='r',
                        shape=(norbits,), dtype=dtype)

    # short-circuit if this orbit is already done
    if all_kld['success'][index] or all_kld['error_code'][index] != 0:
        return

    # container for return
    result = dict()
    result['mmap_filename'] = filename
    result['norbits'] = norbits
    result['index'] = index
    result['dtype'] = dtype

    try:
        new_w0,dt,nsteps = prepare_parent_orbit(this_w0, potential,
                                                nperiods, nsteps_per_period)
    except RuntimeError:
        logger.warning("Failed to integrate orbit when estimating dt,nsteps")
        result['success'] = False
        result['error_code'] = 1
        return result

    logger.info("Orbit {}: initial dt={}, nsteps={}".format(index, dt, nsteps))

    # create an ensemble of particles around this initial condition
    ball_w0 = create_ensemble(new_w0, potential, N=nensemble, m_scale=mscale)
    logger.debug("Generated ensemble of {0} particles".format(nensemble))

    # get the initial density
    kde = KernelDensity(kernel='epanechnikov', bandwidth=bw)
    kde.fit(ball_w0[:,:3])
    ball_dens0 = np.exp(kde.score_samples(ball_w0[:,:3]))

    try:
        t, metric_d, ball_E = do_the_kld(ball_w0, potential, dt, nsteps,
                                         nkld=nkld, kde_bandwidth=bw,
                                         metrics=default_metrics)
    except:
        logger.warning("Unexpected failure: {0}".format(sys.exc_info()))
        result['success'] = False
        result['error_code'] = 4
        return result

    dE_end = np.abs(ball_E[-1] - ball_E[0])
    if (dE_end > ETOL).sum() > nensemble*0.1: # more than 10% don't meet criteria
        logger.warning("Failed due to energy conservation check.")
        result['success'] = False
        result['error_code'] = 3
        return result

    # threshold defined as 1/10 initial mean density
    thresh10 = ball_dens0.mean() / 10.
    thresh100 = ball_dens0.mean() / 100.

    # time at which mean density falls below threshold
    ix = metric_d['mean'] < thresh10
    try:
        thresh_t_10 = t[ix][0]
    except:
        thresh_t_10 = np.nan

    ix = metric_d['mean'] < thresh100
    try:
        thresh_t_100 = t[ix][0]
    except:
        thresh_t_100 = np.nan

    result['thresh_t_10'] = thresh_t_10
    result['thresh_t_100'] = thresh_t_100
    result['dt'] = dt
    result['nsteps'] = nsteps
    result['dE_max'] = dE_end.max()
    result['success'] = True
    result['error_code'] = 0
    result['metrics_end'] = [metric_d[name][-1] for name in sorted(metric_d.dtype.names)]

    return result
