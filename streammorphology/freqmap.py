# coding: utf-8

""" Class for running frequency mapping """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle

# Third-party
import numpy as np
from astropy import log as logger
import gary.potential as gp
import gary.integrate as gi
import gary.coordinates as gc
import gary.dynamics as gd
from superfreq import SuperFreq

# Project
from .config import ConfigNamespace, save, load
from .util import estimate_dt_nsteps

__all__ = ['Freqmap']

class OrbitGridExperiment(object):

    def __init__(self, cache_path, config_filename, config_defaults, overwrite=False, **kwargs):
        # validate cache path
        self.cache_path = os.path.abspath(cache_path)
        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)

        config_path = os.path.join(cache_path, config_filename)

        # if config file doesn't exist, create one with defaults
        if not os.path.exists(config_path):
            ns = ConfigNamespace()
            for k,v in config_defaults.items():
                setattr(ns, k, v)
            save(ns, config_path)

        # if config file exists, read in value from their or defaults
        else:
            ns = load(config_path)
            for k,v in config_defaults.items():
                if not hasattr(ns, k):
                    if k in kwargs: # config default overridden by class kwarg
                        setattr(ns, k, kwargs[k])
                    else: # set config item with default
                        setattr(ns, k, v)

        self.cache_file = os.path.join(self.cache_path, ns.cache_filename)
        if os.path.exists(self.cache_file) and overwrite:
            os.remove(self.cache_file)

        self.config = ns

        # load initial conditions
        self.w0 = np.load(os.path.join(self.cache_path, self.config.w0_filename))
        norbits = len(self.w0)
        logger.info("Number of orbits: {0}".format(norbits))

        # make sure memmap file exists
        if not os.path.exists(self.cache_file):
            # make sure memmap file exists
            d = np.memmap(self.cache_file, mode='w+',
                          dtype=self.cache_dtype, shape=(norbits,))
            d[:] = np.zeros(shape=(norbits,), dtype=self.cache_dtype)

        self.memmap = np.memmap(self.cache_file, mode='r+',
                                dtype=self.cache_dtype, shape=(norbits,))

    def read_cache(self):
        """
        Read the numpy memmap'd file containing cached results from running
        this experiment. This function returns a numpy structured array
        with named columns and proper data types.
        """

        # first get the memmap array
        return np.memmap(self.cache_file, mode='r', shape=(len(self.w0),),
                         dtype=self.cache_dtype)

    def __enter__(self):
        self._tmpdir = os.path.join(self.cache_path, "_tmp")
        logger.debug("Creating temp. directory {0}".format(self._tmpdir))
        os.mkdir(self._tmpdir)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if os.path.exists(self._tmpdir):
            logger.debug("Removing temp. directory {0}".format(self._tmpdir))
            import shutil
            shutil.rmtree(self._tmpdir)

        del self.memmap

    def callback(self, tmpfile):
        if tmpfile is None:
            logger.debug("Tempfile is None")
            return

        with open(tmpfile) as f:
            result = pickle.load(f)
        os.remove(tmpfile)

        logger.debug("Flushing {0} to output array...".format(result['index']))
        if result['error_code'] != 0.:
            # error happened
            for key in self.memmap.dtype.names:
                if key in result:
                    self.memmap[key][result['index']] = result[key]

        else:
            # all is well
            for key in self.memmap.dtype.names:
                self.memmap[key][result['index']] = result[key]

        # flush to output array
        self.memmap.flush()
        logger.debug("...flushed, washing hands.")

config_defaults = dict(
    energy_tolerance=1E-7, # Maximum allowed fractional energy difference
    nperiods=100, # Total number of orbital periods to integrate for
    nsteps_per_period=512, # Number of steps per integration period for integration stepsize
    hamming_p=2, # Exponent to use for Hamming filter
    w0_filename='w0.npy', # Name of the initial conditions file
    cache_filename='freqmap.npy', # Name of the cache file
    potential_filename='potential.yml' # Name of cached potential file
)

class Freqmap(OrbitGridExperiment):
    # failure error codes
    error_codes = {
        1: "Failed to integrate orbit or estimate dt, nsteps.",
        2: "Energy conservation criteria not met.",
        3: "SuperFreq failed on find_fundamental_frequencies()."
    }

    cache_dtype = [
        ('freqs','f8',(2,3)), # three fundamental frequencies computed in 2 windows
        ('amps','f8',(2,3)), # amplitudes of frequencies in time series
        ('dE_max','f8'), # maximum energy difference (compared to initial) during integration
        ('success','b1'), # whether computing the frequencies succeeded or not
        ('is_tube','b1'), # the orbit is a tube orbit
        ('dt','f8'), # timestep used for integration
        ('nsteps','i8'), # number of steps integrated
        ('error_code','i8') # if not successful, why did it fail? see below
    ]

    def __init__(self, cache_path, config_filename=None, overwrite=False, **kwargs):
        if config_filename is None:
            config_filename = 'freqmap.cfg'
        cache_path = os.path.abspath(cache_path)
        super(Freqmap, self).__init__(cache_path=cache_path,
                                      config_filename=config_filename,
                                      config_defaults=config_defaults,
                                      overwrite=overwrite,
                                      **kwargs)

    @classmethod
    def run(cls, w0, potential,
            nperiods=config_defaults['nperiods'],
            nsteps_per_period=config_defaults['nsteps_per_period'],
            hamming_p=config_defaults['hamming_p'],
            energy_tolerance=config_defaults['energy_tolerance']):
        # return dict
        result = dict()

        # get timestep and nsteps for integration
        try:
            dt, nsteps = estimate_dt_nsteps(potential, w0.copy(),
                                            nperiods,
                                            nsteps_per_period)
        except RuntimeError:
            logger.warning("Failed to integrate orbit when estimating dt,nsteps")
            result['freqs'] = np.ones((2,3))*np.nan
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

        if dEmax > energy_tolerance:
            logger.warning("Failed due to energy conservation check.")
            result['freqs'] = np.ones((2,3))*np.nan
            result['success'] = False
            result['error_code'] = 2
            return result

        # start finding the frequencies -- do first half then second half
        sf1 = SuperFreq(t[:nsteps//2+1], p=hamming_p)
        sf2 = SuperFreq(t[nsteps//2:], p=hamming_p)

        # classify orbit full orbit
        circ = gd.classify_orbit(ws)
        is_tube = np.any(circ)

        # define slices for first and second parts
        sl1 = slice(None,nsteps//2+1)
        sl2 = slice(nsteps//2,None)

        if is_tube:
            # first need to flip coordinates so that circulation is around z axis
            new_ws = gd.align_circulation_with_z(ws, circ)
            new_ws = gc.cartesian_to_poincare_polar(new_ws)
            fs1 = [(new_ws[sl1,j] + 1j*new_ws[sl1,j+3]) for j in range(3)]
            fs2 = [(new_ws[sl2,j] + 1j*new_ws[sl2,j+3]) for j in range(3)]

        else:  # box
            fs1 = [(ws[sl1,0,j] + 1j*ws[sl1,0,j+3]) for j in range(3)]
            fs2 = [(ws[sl2,0,j] + 1j*ws[sl2,0,j+3]) for j in range(3)]

        logger.debug("Running SuperFreq on the orbits")
        try:
            freqs1,d1,ixs1 = sf1.find_fundamental_frequencies(fs1, nintvec=8)
            freqs2,d2,ixs2 = sf2.find_fundamental_frequencies(fs2, nintvec=8)
        except:
            result['freqs'] = np.ones((2,3))*np.nan
            result['success'] = False
            result['error_code'] = 3
            return result

        result['freqs'] = np.vstack((freqs1, freqs2))
        result['dE_max'] = dEmax
        result['is_tube'] = float(is_tube)
        result['dt'] = float(dt)
        result['nsteps'] = nsteps
        result['amps'] = np.vstack((d1['|A|'][ixs1], d2['|A|'][ixs2]))
        result['success'] = True
        result['error_code'] = 0
        return result

    def _run_wrapper(self, index):
        logger.info("Orbit {0}".format(index))

        # unpack input argument dictionary
        potential = gp.load(os.path.join(self.cache_path, self.config.potential_filename))

        # read out just this initial condition
        norbits = len(self.w0)
        allfreqs = np.memmap(self.cache_file, mode='r',
                             shape=(norbits,), dtype=self.cache_dtype)

        # short-circuit if this orbit is already done
        if allfreqs['success'][index]:
            logger.debug("Orbit {0} already successfully completed.".format(index))
            return None

        res = self.run(w0=self.w0[index], potential=potential,
                       nperiods=self.config.nperiods,
                       nsteps_per_period=self.config.nsteps_per_period,
                       hamming_p=self.config.hamming_p,
                       energy_tolerance=self.config.energy_tolerance)
        res['index'] = index

        # cache res into a tempfile, return name of tempfile
        tmpfile = os.path.join(self._tmpdir, "{0}-{1}.pickle".format(self.__class__.__name__, index))
        with open(tmpfile, 'w') as f:
            pickle.dump(res, f)
        return tmpfile

    def __call__(self, index):
        return self._run_wrapper(index)
