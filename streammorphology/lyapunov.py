# coding: utf-8

""" Class for running Lyapunov mapping """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy import log as logger

# Project
from .extern.fast_mle import mle
from .util import estimate_dt_nsteps
from .experimentrunner import OrbitGridExperiment

__all__ = ['Lyapmap']

class Lyapmap(OrbitGridExperiment):
    # failure error codes
    error_codes = {
        1: "Failed to integrate orbit or estimate dt, nsteps.",
        2: "Energy conservation criteria not met."
    }

    cache_dtype = [
        ('lyap_exp','f8'), # MLE estimate
        ('success','b1'), # whether computing the frequencies succeeded or not
        ('error_code','i8'), # if not successful, why did it fail? see below
        # ('lyap_exp_end','f8',(1024,)), # last 1024 timesteps of FTMLE estimate
        ('dE_max','f8'), # maximum energy difference (compared to initial) during integration
    ]

    _run_kwargs = ['nperiods', 'nsteps_per_period', 'noffset_orbits', 'energy_tolerance']
    config_defaults = dict(
        energy_tolerance=1E-7, # Maximum allowed fractional energy difference
        nperiods=1000, # Total number of orbital periods to integrate for
        nsteps_per_period=512, # Number of steps per integration period for integration stepsize
        noffset_orbits=2, # Number of offset orbits to integrate and average.
        w0_filename='w0.npy', # Name of the initial conditions file
        cache_filename='lyapmap.npy', # Name of the cache file
        potential_filename='potential.yml' # Name of cached potential file
    )

    @classmethod
    def run(cls, w0, potential,
            nperiods=config_defaults['nperiods'],
            nsteps_per_period=config_defaults['nsteps_per_period'],
            noffset_orbits=config_defaults['noffset_orbits'],
            energy_tolerance=config_defaults['energy_tolerance']):
        # return dict
        result = dict()

        # get timestep and nsteps for integration
        try:
            dt, nsteps = estimate_dt_nsteps(w0.copy(), potential,
                                            nperiods,
                                            nsteps_per_period)
        except RuntimeError:
            logger.warning("Failed to integrate orbit when estimating dt,nsteps")
            result['lyap_exp'] = np.nan
            result['success'] = False
            result['error_code'] = 1
            return result

        # integrate orbit
        logger.debug("Integrating orbit with dt={0}, nsteps={1}".format(dt, nsteps))
        try:
            LEs,t,w = mle(w0.copy(), potential, dt=dt, nsteps=nsteps,
                          noffset_orbits=noffset_orbits)
        except RuntimeError: # ODE integration failed
            logger.warning("Orbit integration failed.")
            dEmax = 1E10
        else:
            logger.debug('Orbit integrated successfully, checking energy conservation...')

            # check energy conservation for the orbit
            E0 = potential.total_energy(w0[:3].copy(), w0[3:].copy())[0]
            E1 = potential.total_energy(w[0,:3].copy(), w[0,3:].copy())[0]
            dEmax = np.abs((E1-E0)/E0)
            logger.debug('max(∆E) = {0:.2e}'.format(dEmax))

        if dEmax > energy_tolerance:
            result['lyap_exp'] = np.nan
            result['success'] = False
            result['error_code'] = 2
            return result

        # le_end = np.mean(LEs[-16384::16], axis=1)
        # le_end.resize(1024)
        result['lyap_exp'] = np.mean(LEs)
        # result['lyap_exp_end'] = le_end
        result['success'] = True
        result['error_code'] = 0
        result['dE_max'] = dEmax

        return result
