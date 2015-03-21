# coding: utf-8

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
from .mmap_util import get_dtype
from .core import create_ball, nearest_pericenter, do_the_kld
from ..freqmap import estimate_dt_nsteps

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
parser_arguments.append([('--ndensity_threshold',),
                         dict(dest='ndensity_threshold', default=16, type=int,
                              help='Number of density thresholds.')])

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
    nkld = task['nkld']

    # number of periods to integrate the ensemble
    nperiods = task['nperiods']
    nsteps_per_period = task['nsteps_per_period']

    # number of times to evaluate the number of particles above a density threshold
    ndensity_threshold = task['ndensity_threshold']

    # read out just this initial condition
    w0 = np.load(w0_filename)
    norbits = len(w0)
    this_w0 = w0[index].copy()
    all_kld = np.memmap(filename, mode='r+',
                        shape=(norbits,), dtype=get_dtype(nkld))

    # short-circuit if this orbit is already done
    # TODO: handle status == 0 (not attempted) different from
    #       status >= 2 (previous failure)
    if all_kld['status'][index] == 1:
        return

    try:
        dt,nsteps,T = estimate_dt_nsteps(potential, this_w0, nperiods, nsteps_per_period,
                                         return_periods=True)
    except RuntimeError:
        logger.warning("Failed to integrate orbit when estimating dt,nsteps")
        all_kld['status'][index] = 3  # failed due to integration
        all_kld.flush()
        return

    # find the nearest (in time) pericenter to the given initial condition
    try:
        peri_w0 = nearest_pericenter(this_w0, potential, dt, T.max())
    except:
        logger.warning("Failed to find nearest pericenter.")
        all_kld['status'][index] = 3  # failed due to integration
        all_kld.flush()
        return

    logger.info("Orbit {}: initial dt={}, nsteps={}".format(index, dt, nsteps))

    # create an ensemble of particles around this initial condition
    ball_w0 = create_ball(peri_w0, potential, N=nensemble, m_scale=mscale)
    logger.debug("Generated ensemble of {0} particles".format(nensemble))

    # compute the KLD at specified intervals
    # kde = KernelDensity(kernel='epanechnikov', bandwidth=bw)
    # kde.fit(ball_w0[:,:3])
    # kde_densy = np.exp(kde.score_samples(ball_w0[:,:3]))
    # dens_max = kde_densy.mean()
    # TODO: don't hardcode this in you ass
    dens_max = 10**-3.2

    # compute the density thresholds ranging from the mean density of the initial ball down
    #   to two orders of magnitude lower density
    density_thresholds = 10**np.linspace(np.log10(dens_max), np.log10(dens_max) - 2.,
                                         ndensity_threshold)

    try:
        kld_t, kld, mean_dens = do_the_kld(nkld, ball_w0, potential, dt, nsteps, bw,
                                           density_thresholds)
    except:
        logger.warning("Unexpected failure: {0}".format(sys.exc_info()))
        all_kld['status'][index] = 4  # some kind of catastrophic failure
        all_kld.flush()
        return

    # TODO: compare final E vs. initial E against ETOL?
    # E_end = float(np.squeeze(potential.total_energy(w_i[0,:3], w_i[0,3:])))
    # dE = np.log10(np.abs(E_end - E0))
    # if dE > ETOL:
    #     all_kld['status'][index] = 2  # failed due to energy conservation
    #     all_kld.flush()
    #     return

    # all_kld['frac_above_dens'][index] = frac_above_dens
    all_kld['mean_dens'][index] = mean_dens
    all_kld['kld'][index] = kld
    all_kld['kld_t'][index] = kld_t
    all_kld['dt'][index] = dt
    all_kld['nsteps'][index] = nsteps
    all_kld['dE_max'][index] = 0.  # TODO:
    all_kld['status'][index] = 1  # success!
    all_kld.flush()

    # # fit a power law to the KLDs
    # model = lambda p,t: p[0] * t**p[1]
    # errfunc = lambda p,t,y: (y - model(p,t))
    # p_opt,ier = so.leastsq(errfunc, x0=(0.01,-0.15), args=(KLD_times,KLD))
    # if ier not in [1,2,3,4]:
    #     all_kld[index,0] = np.nan
    #     all_kld[index,1] = 3.  # failed due to leastsq

    # # power-law index
    # k = p_opt[1]
