# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import time as pytime

# Third-party
from astropy import log as logger
import gary.potential as gp
import numpy as np

# Project
from .. import ETOL
from .mmap_util import get_dtype
from .core import create_ball, peri_to_apo, do_the_kld
from ..freqmap import estimate_dt_nsteps

__all__ = ['worker', 'parser_arguments']

parser_arguments = list()

# list of [args, kwargs]
parser_arguments.append([('--nensemble',), dict(dest='nensemble', required=True,
                                                type=int, help='Number of orbits per ensemble.')])
parser_arguments.append([('--mscale',), dict(dest='mscale', default=1E4,
                                             type=float, help='Mass scale of ensemble.')])
parser_arguments.append([('--evln-time',), dict(dest='evolution_time', required=True,
                                                type=float, help='Total time to evolve the ensemble.')])
parser_arguments.append([('--bandwidth',), dict(dest='kde_bandwidth', default=10.,
                                                type=float, help='KDE kernel bandwidth.')])
parser_arguments.append([('--nkld',), dict(dest='nkld', default=256, type=int,
                                           help='Number of times to evaluate the KLD over the '
                                                'integration of the ensemble.')])
parser_arguments.append([('--nperiods',), dict(dest='nperiods', default=500, type=int,
                                               help='Number of periods to integrate for.')])

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

    # total time to evolve the ensembles
    evolution_time = task['evolution_time']

    # if these aren't set, we'll need to estimate them
    dt = task.get('dt',None)
    nsteps = task.get('nsteps',None)
    nsteps_per_period = task.get('nsteps_per_period', 128)

    # read out just this initial condition
    w0 = np.load(w0_filename)
    norbits = len(w0)
    this_w0 = w0[index].copy()
    all_kld = np.memmap(filename, mode='r+',
                        shape=(norbits,), dtype=get_dtype(nkld))

    # short-circuit if this orbit is already done
    if all_kld['status'][index] == 1:
        return

    # TODO: handle status == 0 (not attempted) different from
    #       status >= 2 (previous failure)

    try:
        this_w0, dt, nsteps, apo_ixes = peri_to_apo(w0[index].copy(), potential,
                                                    evolution_time)
    except RuntimeError:
        logger.warning("Failed to integrate orbit when estimating dt,nsteps")
        all_kld['status'][index] = 3  # failed due to integration
        all_kld.flush()
        return

    logger.info("Orbit {}: initial dt={}, nsteps={}".format(index, dt, nsteps))

    ball_w0 = create_ball(this_w0, potential, N=nensemble, m_scale=mscale)
    logger.debug("Generated ensemble of {0} particles".format(nensemble))

    kld_ts, klds = do_the_kld(ball_w0, potential, apo_ixes, dt=dt, kde_bandwidth=bw)

    KLD = np.zeros(256)
    KLD[:len(klds)] = np.array(klds)
    KLD[len(klds):] = np.nan

    KLD_times = np.zeros(256)
    KLD_times[:len(kld_ts)] = np.array(kld_ts)

    # TODO: compare final E vs. initial E against ETOL?
    # E_end = float(np.squeeze(potential.total_energy(w_i[0,:3], w_i[0,3:])))
    # dE = np.log10(np.abs(E_end - E0))
    # if dE > ETOL:
    #     all_kld['status'][index] = 2  # failed due to energy conservation
    #     all_kld.flush()
    #     return

    all_kld['kld'][index] = KLD
    all_kld['kld_t'][index] = KLD_times
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
