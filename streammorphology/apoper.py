# coding: utf-8

""" Class for running frequency mapping """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy import log as logger
import gary.integrate as gi
import gary.dynamics as gd
from scipy.signal import argrelmin, argrelmax

# Project
from .experimentrunner import OrbitGridExperiment

__all__ = ['ApoPer']

class ApoPer(OrbitGridExperiment):
    # failure error codes
    error_codes = {
        1: "Failed to integrate orbit or estimate dt, nsteps.",
        2: "Energy conservation criteria not met."
    }

    _run_kwargs = ['nperiods', 'nsteps_per_period', 'hamming_p', 'energy_tolerance']
    config_defaults = dict(
        energy_tolerance=1E-7, # Maximum allowed fractional energy difference
        nperiods=16, # Total number of orbital periods to integrate for
        nsteps_per_period=1024, # Number of steps per integration period for integration stepsize
        w0_filename='w0.npy', # Name of the initial conditions file
        cache_filename='apoper.npy', # Name of the cache file
        potential_filename='potential.yml' # Name of cached potential file
    )

    @property
    def cache_dtype(self):
        dtype = [
            ('dE_max','f8'), # maximum energy difference (compared to initial) during integration
            ('success','b1'), # whether computing the frequencies succeeded or not
            ('dt','f8'), # timestep used for integration
            ('nsteps','i8'), # number of steps integrated
            ('error_code','i8'), # if not successful, why did it fail? see below
            ('pericenters','f8',(self.config.nperiods+2,)),
            ('apocenters','f8',(self.config.nperiods+2,))
        ]
        return dtype

    @classmethod
    def run(cls, w0, potential, **kwargs):
        c = dict()
        for k in cls.config_defaults.keys():
            if k not in kwargs:
                c[k] = cls.config_defaults[k]
            else:
                c[k] = kwargs[k]

        # return dict
        result = dict()

        # get timestep and nsteps for integration
        try:
            # integrate orbit
            t,w = potential.integrate_orbit(w0.copy(), dt=0.2, nsteps=50000)

            # radial oscillations
            r = np.sqrt(np.sum(w[:,0,:3]**2, axis=-1))
            T = gd.peak_to_peak_period(t, r)

            # timestep from number of steps per period
            dt = float(T) / float(c['nsteps_per_period'])
            nsteps = int(round(c['nperiods'] * T / dt))

        except RuntimeError:
            logger.warning("Failed to integrate orbit when estimating dt,nsteps")
            result['success'] = False
            result['error_code'] = 1
            return result

        # integrate orbit
        logger.debug("Integrating orbit with dt={0}, nsteps={1}".format(dt, nsteps))
        try:
            t,ws = potential.integrate_orbit(w0.copy(), dt=dt, nsteps=nsteps,
                                             Integrator=gi.DOPRI853Integrator,
                                             Integrator_kwargs=dict(atol=1E-11))
        except RuntimeError: # ODE integration failed
            logger.warning("Orbit integration failed.")
            dEmax = 1E10
        else:
            logger.debug('Orbit integrated successfully, checking energy conservation...')

            # check energy conservation for the orbit
            E = potential.total_energy(ws[:,0,:3].copy(), ws[:,0,3:].copy())
            dE = np.abs(E[1:] - E[0])
            dEmax = dE.max() / np.abs(E[0])
            logger.debug('max(∆E) = {0:.2e}'.format(dEmax))

        if dEmax > c['energy_tolerance']:
            logger.warning("Failed due to energy conservation check.")
            result['freqs'] = np.ones((2,3))*np.nan
            result['success'] = False
            result['error_code'] = 2
            return result

        # find apos, peris
        r = np.sqrt(np.sum(ws[:,0,:3]**2, axis=-1))
        pc = r[argrelmin(r)[0]]
        ac = r[argrelmax(r)[0]]
        pc.resize(c['nperiods']+2)
        ac.resize(c['nperiods']+2)

        result['dE_max'] = dEmax
        result['dt'] = float(dt)
        result['nsteps'] = nsteps
        result['success'] = True
        result['error_code'] = 0
        result['pericenters'] = pc
        result['apocenters'] = ac

        return result
