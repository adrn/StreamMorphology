# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy import log as logger
import gary.integrate as gi
import gary.coordinates as gc
import gary.dynamics as gd
from superfreq import SuperFreq

# Project
from .util import estimate_dt_nsteps
from .ensemble import create_ensemble
from .experimentrunner import OrbitGridExperiment

__all__ = ['EnsembleFreqVariance']

class EnsembleFreqVariance(OrbitGridExperiment):
    # failure error codes
    error_codes = {
        1: "Failed to integrate orbit or estimate dt, nsteps.",
        2: "Energy conservation criteria not met.",
        3: "SuperFreq failed on find_fundamental_frequencies()."
    }

    _run_kwargs = ['nperiods', 'energy_tolerance', 'nsteps_per_period',
                   'hamming_p', 'nensemble', 'nintvec', 'force_cartesian']
    config_defaults = dict(
        nperiods=50, # total number of periods to integrate for
        energy_tolerance=1E-8, # Maximum allowed fractional energy difference
        nsteps_per_period=512, # Number of steps per integration period for integration stepsize
        mscale=1E4,
        hamming_p=4, # Exponent to use for Hamming filter
        nensemble=128, # How many orbits per ensemble
        nintvec=15, # maximum number of integer vectors to use in SuperFreq
        force_cartesian=False, # Do frequency analysis on cartesian coordinates
        w0_filename='w0.npy', # Name of the initial conditions file
        cache_filename='ensemblefreqvariance.npy', # Name of the cache file
        potential_filename='potential.yml' # Name of cached potential file
    )

    @property
    def cache_dtype(self):
        dt = [
            ('dt','f8'), # timestep used for integration
            ('nsteps','i8'), # number of steps integrated
            ('freqs','f8',(self.config.nensemble+1,3)), # three fundamental frequencies computed in windows
            ('amps','f8',(self.config.nensemble+1,3)), # amplitudes of frequencies in time series
            ('dE_max','f8'), # maximum energy difference (compared to initial) during integration
            ('success','b1'), # whether computing the frequencies succeeded or not
            ('error_code','i8') # if not successful, why did it fail? see below
        ]
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
            # new_w0,dt,nsteps = prepare_parent_orbit(w0=w0.copy(),
            #                                         potential=potential,
            #                                         nperiods=c['nperiods'],
            #                                         nsteps_per_period=c['nsteps_per_period'])
            dt,nsteps = estimate_dt_nsteps(w0=w0.copy(),
                                           potential=potential,
                                           nperiods=c['nperiods'],
                                           nsteps_per_period=c['nsteps_per_period'])
            new_w0 = w0.copy()
        except RuntimeError:
            logger.warning("Failed to integrate orbit when estimating dt,nsteps")
            result['success'] = False
            result['error_code'] = 1
            return result

        logger.debug("Integrating orbit with dt={0}, nsteps={1}".format(dt, nsteps))

        # create an ensemble of particles around this initial condition
        ensemble_w0 = create_ensemble(new_w0, potential, n=c['nensemble'], m_scale=c['mscale'])
        logger.debug("Generated ensemble of {0} particles".format(c['nensemble']))

        logger.debug("Integrating ensemble with dt={0}, nsteps={1}".format(dt, nsteps))
        try:
            t,ws = potential.integrate_orbit(ensemble_w0, dt=dt, nsteps=nsteps,
                                             Integrator=gi.DOPRI853Integrator,
                                             Integrator_kwargs=dict(atol=1E-11))
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

        if dEmax > c['energy_tolerance']:
            logger.warning("Failed due to energy conservation check.")
            result['freqs'] = np.nan
            result['success'] = False
            result['error_code'] = 2
            return result

        # classify parent orbit
        circ = gd.classify_orbit(ws[:,0])
        is_tube = np.any(circ)

        logger.debug("Running SuperFreq on each orbit:")

        allfreqs = []
        allamps = []
        for i in range(c['nensemble']+1):

            logger.debug("Orbit {0}".format(i))
            ww = ws[:,i]
            if is_tube and not c['force_cartesian']:
                # need to flip coordinates until circulation is around z axis
                new_ws = gd.align_circulation_with_z(ww, circ)
                new_ws = gc.cartesian_to_poincare_polar(new_ws)
            else:
                new_ws = ww

            fs = [(new_ws[:,j] + 1j*new_ws[:,j+3]) for j in range(3)]
            naff = SuperFreq(t, p=c['hamming_p'])

            try:
                freqs,d,ixs = naff.find_fundamental_frequencies(fs, nintvec=c['nintvec'])
            except:
                allfreqs.append([np.nan,np.nan,np.nan])
                allamps.append([np.nan,np.nan,np.nan])
                continue

            allfreqs.append(freqs.tolist())
            allamps.append(d['|A|'][ixs].tolist())

        allfreqs = np.array(allfreqs)
        allamps = np.array(allamps)

        result['freqs'] = allfreqs
        result['amps'] = allamps
        result['dE_max'] = dEmax
        result['dt'] = float(dt)
        result['nsteps'] = nsteps
        result['success'] = True
        result['error_code'] = 0
        return result
