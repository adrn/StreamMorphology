# coding: utf-8

""" Class for running ensemble mapping """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy import log as logger
from sklearn.neighbors import KernelDensity

# Project
from .core import create_ensemble# , do_the_kld
from ..experimentrunner import OrbitGridExperiment

__all__ = ['Ensemble']

class Ensemble(OrbitGridExperiment):
    # failure error codes
    error_codes = {
        1: "Failed to integrate orbit or estimate dt, nsteps.",
        2: "Failed to find nearest pericenter.",
        3: "Energy conservation criteria not met.",
        4: "Catastrophic, unexpected, OMG failure."
    }

    cache_dtype = [
        ('thresh_t_10','f8'),
        ('thresh_t_100','f8'),
        ('dt','f8'), # timestep used for integration
        ('nsteps','i8'), # number of steps integrated
        ('dE_max','f8'), # maximum energy difference (compared to initial) during integration
        ('error_code','i8'), # if not successful, why did it fail? see above
        ('success','b1'), # whether computing the frequencies succeeded or not
        ('metrics_end','f8') # TODO
    ]

    _run_kwargs = ['energy_tolerance', 'nperiods', 'nsteps_per_period',
                   'nensemble', 'mscale', 'kde_bandwidth', 'neval']
    config_defaults = dict(
        energy_tolerance=1E-8, # Maximum allowed fractional energy difference
        nperiods=100, # Total number of orbital periods to integrate for
        nsteps_per_period=512, # Number of steps per integration period for integration stepsize
        nensemble=1000, # How many orbits per ensemble
        mscale=1E4, # mass scale of the ensemble
        kde_bandwidth=None, # KDE bandwidth (default=None, uses adaptive)
        neval=128, # Number of times during integration to build KDE
        w0_filename='w0.npy', # Name of the initial conditions file
        cache_filename='ensemble.npy', # Name of the cache file
        potential_filename='potential.yml' # Name of cached potential file
    )

    @classmethod
    def run(cls, w0, potential, **kwargs):

        config = dict()
        for k in cls.config_defaults.keys():
            if k not in kwargs:
                config[k] = cls.config_defaults[k]
            else:
                config[k] = kwargs[k]

        # container for return
        result = dict()

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
