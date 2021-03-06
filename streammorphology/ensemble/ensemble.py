# coding: utf-8

""" Class for running ensemble mapping """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Std lib
import sys

# Third-party
import numpy as np
from astropy import log as logger

# Project
from .core import create_ensemble, prepare_parent_orbit
from .follow_ensemble import follow_ensemble
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

    _run_kwargs = ['energy_tolerance', 'nperiods', 'nsteps_per_period',
                   'nensemble', 'mscale', 'kde_bandwidth', 'neval',
                   'store_all_dens', 'store_all_w', 'min_pericenter']
    config_defaults = dict(
        energy_tolerance=1E-8, # Maximum allowed fractional energy difference
        nperiods=16, # Total number of orbital periods to integrate for
        nsteps_per_period=512, # Number of steps per integration period for integration stepsize
        nensemble=1000, # How many orbits per ensemble
        mscale=1E4, # mass scale of the ensemble
        kde_bandwidth=None, # KDE bandwidth (default=None, uses adaptive)
        neval=128, # Number of times during integration to build KDE
        store_all_dens=False, # Store full distribution of density values for each particle at each eval
        store_all_w=False, # Store all phase-space positions for all ensemble particles at each eval
        min_pericenter=True, # Start the ensembles at minimum pericenter
        w0_filename='w0.npy', # Name of the initial conditions file
        cache_filename='ensemble.npy', # Name of the cache file
        potential_filename='potential.yml' # Name of cached potential file
    )

    @property
    def cache_dtype(self):
        dt = [
            ('dt','f8'), # timestep used for integration
            ('nsteps','i8'), # number of steps integrated
            ('dE_max','f8'), # maximum energy difference (compared to initial) during integration
            ('error_code','i8'), # if not successful, why did it fail? see above
            ('success','b1'), # whether computing the frequencies succeeded or not
            ('mean_dens','f8',self.config.neval), # mean density at the end of integration
            ('mean_log_dens','f8',self.config.neval), # mean density at the end of integration
            ('median_dens','f8',self.config.neval), # mean density at the end of integration
            ('median_log_dens','f8',self.config.neval), # mean density at the end of integration
            ('skew_dens','f8',self.config.neval), # mean density at the end of integration
            ('skew_log_dens','f8',self.config.neval), # mean density at the end of integration
            ('kurtosis_dens','f8',self.config.neval), # mean density at the end of integration
            ('kurtosis_log_dens','f8',self.config.neval), # mean density at the end of integration
            ('t','f8',self.config.neval) # times of each evaluation
        ]
        if self.config.store_all_dens:
            dt.append(('all_dens','f8',(self.config.neval,self.config.nensemble+1)))

        if self.config.store_all_w:
            dt.append(('all_w','f8',(self.config.neval,self.config.nensemble+1,6)))

        return dt

    @classmethod
    def run(cls, w0, potential, **kwargs):
        c = dict()
        for k in cls.config_defaults.keys():
            if k not in kwargs:
                c[k] = cls.config_defaults[k]
            else:
                c[k] = kwargs[k]

        # container for return
        result = dict()

        try:
            new_w0,dt,nsteps = prepare_parent_orbit(w0.copy(), potential,
                                                    c['nperiods'], c['nsteps_per_period'])
        except RuntimeError:
            logger.warning("Failed to integrate orbit when estimating dt,nsteps")
            result['success'] = False
            result['error_code'] = 1
            return result

        logger.debug("Integrating orbit with dt={0}, nsteps={1}".format(dt, nsteps))

        # HACK
        # R0 = np.sqrt(np.sum(new_w0[:3]**2))
        # mscale = (0.1 / R0)**3 * potential.mass_enclosed(new_w0[:3])
        mscale = c['mscale']

        # create an ensemble of particles around this initial condition
        ensemble_w0 = create_ensemble(new_w0, potential, n=c['nensemble'], m_scale=mscale)
        logger.debug("Generated ensemble of {0} particles".format(c['nensemble']))

        try:
            ret = follow_ensemble(ensemble_w0, potential, dt, nsteps,
                                  neval=c['neval'],
                                  kde_bandwidth=c['kde_bandwidth'],
                                  return_all_density=c['store_all_dens'],
                                  return_all_w=c['store_all_w'])
        except:
            import traceback
            t,v,tb = sys.exc_info()
            logger.warning("Unexpected failure: {0}".format(v))
            logger.debug(traceback.format_tb(tb))
            result['success'] = False
            result['error_code'] = 4
            return result

        t = ret['t']
        data = ret['data']
        ball_E = ret['energy']

        dE_end = np.abs(ball_E[-1] - ball_E[0])
        if (dE_end > c['energy_tolerance']).sum() > c['nensemble']*0.1: # more than 10% don't meet criteria
            logger.warning("Failed due to energy conservation check.")
            result['success'] = False
            result['error_code'] = 3
            return result

        result['dt'] = dt
        result['nsteps'] = nsteps
        result['dE_max'] = dE_end.max()
        result['success'] = True
        result['error_code'] = 0

        # moments
        result['mean_dens'] = data['mean']
        result['mean_log_dens'] = data['mean_log']
        result['median_dens'] = data['median']
        result['median_log_dens'] = data['median_log']
        result['skew_dens'] = data['skew']
        result['skew_log_dens'] = data['skew_log']
        result['kurtosis_dens'] = data['kurtosis']
        result['kurtosis_log_dens'] = data['kurtosis_log']

        result['t'] = t
        if c['store_all_dens']:
            result['all_dens'] = ret['all_density']

        if c['store_all_w']:
            result['all_w'] = ret['all_w']

        return result
