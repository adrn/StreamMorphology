# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy import log as logger
import gary.integrate as gi
import gary.coordinates as gc
import gary.dynamics as gd
from gary.util import rolling_window
from superfreq import SuperFreq

# Project
from .util import estimate_dt_nsteps
from .experimentrunner import OrbitGridExperiment

__all__ = ['FreqVariance']

class FreqVariance(OrbitGridExperiment):
    # failure error codes
    error_codes = {
        1: "Failed to integrate orbit or estimate dt, nsteps.",
        2: "Energy conservation criteria not met.",
        3: "SuperFreq failed on find_fundamental_frequencies()."
    }

    _run_kwargs = ['total_nperiods', 'window_width', 'window_stride',
                   'energy_tolerance', 'nsteps_per_period', 'hamming_p',
                   'force_cartesian']
    config_defaults = dict(
        total_nperiods=128+64, # total number of periods to integrate for
        window_width=128, # width of the window (in orbital periods) to compute freqs in
        window_stride=1, # how much to shift window after each computation
        energy_tolerance=1E-8, # Maximum allowed fractional energy difference
        nsteps_per_period=512, # Number of steps per integration period for integration stepsize
        hamming_p=1, # Exponent to use for Hamming filter
        force_cartesian=False, # Do frequency analysis on cartesian coordinates
        w0_filename='w0.npy', # Name of the initial conditions file
        cache_filename='freqvariance.npy', # Name of the cache file
        potential_filename='potential.yml' # Name of cached potential file
    )

    def __init__(self, cache_path, overwrite=False, **kwargs):
        super(FreqVariance, self).__init__(cache_path, overwrite=overwrite, **kwargs)
        self._nwindows = int((self.config.total_nperiods - self.config.window_width) / self.config.window_stride)

    @property
    def cache_dtype(self):
        dt = [
            ('freqs','f8',(self._nwindows,3)), # three fundamental frequencies computed in windows
            ('amps','f8',(self._nwindows,3)), # amplitudes of frequencies in time series
            ('dE_max','f8'), # maximum energy difference (compared to initial) during integration
            ('is_tube','b1'), # the orbit is a tube orbit
            ('dt','f8'), # timestep used for integration
            ('nsteps','i8'), # number of steps integrated
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

        # automatically estimate dt, nsteps
        try:
            dt, nsteps = estimate_dt_nsteps(w0.copy(), potential,
                                            c['total_nperiods'], c['nsteps_per_period'])
        except RuntimeError:
            logger.warning("Failed to integrate orbit when estimating dt,nsteps")
            result['freqs'] = np.nan
            result['success'] = False
            result['error_code'] = 1
            return result

        logger.debug("Integrating orbit with dt={0}, nsteps={1}".format(dt, nsteps))
        try:
            t,ws = potential.integrate_orbit(w0.copy(), dt=dt, nsteps=nsteps,
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

        # windowing properties - convert from period to steps
        window_width = int(c['window_width'] * c['nsteps_per_period'])
        window_stride = int(c['window_stride'] * c['nsteps_per_period'])

        # classify orbit full orbit
        circ = gd.classify_orbit(ws[:,0])
        is_tube = np.any(circ)

        logger.debug("Running SuperFreq on each window:")

        allfreqs = []
        allamps = []
        for (i1,i2),ww in rolling_window(ws[:,0], window_size=window_width, stride=window_stride, return_idx=True):
            if i2 >= nsteps:
                break

            logger.debug("Window: {0}:{1}".format(i1,i2))
            if is_tube and not c['force_cartesian']:
                # need to flip coordinates until circulation is around z axis
                new_ws = gd.align_circulation_with_z(ww, circ)
                new_ws = gc.cartesian_to_poincare_polar(new_ws)
            else:
                new_ws = ww

            fs = [(new_ws[:,j] + 1j*new_ws[:,j+3]) for j in range(3)]
            naff = SuperFreq(t[i1:i2], p=c['hamming_p'])

            try:
                freqs,d,ixs = naff.find_fundamental_frequencies(fs, nintvec=5)
            except:
                result['freqs'] = np.nan
                result['success'] = False
                result['error_code'] = 3
                return result

            allfreqs.append(freqs.tolist())
            allamps.append(d['|A|'][ixs].tolist())
        allfreqs = np.array(allfreqs)
        allamps = np.array(allamps)

        result['freqs'] = allfreqs
        result['amps'] = allamps
        result['dE_max'] = dEmax
        result['dt'] = float(dt)
        result['nsteps'] = nsteps
        result['is_tube'] = is_tube
        result['success'] = True
        result['error_code'] = 0
        return result
