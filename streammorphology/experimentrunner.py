# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from abc import ABCMeta, abstractproperty
try:
    from abc import abstractclassmethod
except ImportError: # only works in Python 3
    class abstractclassmethod(classmethod):

        __isabstractmethod__ = True

        def __init__(self, callable):
            callable.__isabstractmethod__ = True
            super(abstractclassmethod, self).__init__(callable)

from argparse import ArgumentParser
import logging
import os
import sys
try:
    import cPickle as pickle
except ImportError: # only works in Python 3
    import pickle

# Third-party
import numpy as np
from astropy import log as logger
from gary.util import get_pool

# Project
from .config import ConfigNamespace, save, load

__all__ = ['OrbitGridExperiment', 'ExperimentRunner']

class OrbitGridExperiment(object):

    __metaclass__ = ABCMeta

    def __init__(self, cache_path, config_filename=None, overwrite=False, **kwargs):
        if config_filename is None:
            config_filename = '{0}.cfg'.format(self.__class__.__name__)
            logger.debug("Config filename not specified - using default ({0})".format(config_filename))

        # validate cache path
        self.cache_path = os.path.abspath(cache_path)
        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)

        config_path = os.path.join(cache_path, config_filename)

        # if config file doesn't exist, create one with defaults
        if not os.path.exists(config_path):
            ns = ConfigNamespace()
            for k,v in self.config_defaults.items():
                setattr(ns, k, v)
            save(ns, config_path)

        # if config file exists, read in value from their or defaults
        else:
            ns = load(config_path)
            for k,v in self.config_defaults.items():
                if not hasattr(ns, k):
                    if k in kwargs: # config default overridden by class kwarg
                        setattr(ns, k, kwargs[k])
                    else: # set config item with default
                        setattr(ns, k, v)

        # make sure required stuff is in there:
        _required = ['cache_filename', 'w0_filename', 'potential_filename']
        for _req in _required:
            if not hasattr(ns, _req):
                raise ValueError("Config specification missing value for '{0}'".format(_req))

        self.cache_file = os.path.join(self.cache_path, ns.cache_filename)
        if os.path.exists(self.cache_file) and overwrite:
            os.remove(self.cache_file)

        self.config = ns

        # load initial conditions
        w0_path = os.path.join(self.cache_path, self.config.w0_filename)
        if not os.path.exists(w0_path):
            raise IOError("Initial conditions file '{0}' doesn't exist! You need"
                          "to generate this file first using make_grid.py".format(w0_path))
        self.w0 = np.load(w0_path)
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

    # Context management
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

    def read_cache(self):
        """
        Read the numpy memmap'd file containing cached results from running
        this experiment. This function returns a numpy structured array
        with named columns and proper data types.
        """

        # first get the memmap array
        return np.memmap(self.cache_file, mode='r', shape=(len(self.w0),),
                         dtype=self.cache_dtype)

    def callback(self, tmpfile):
        """
        TODO:
        """

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

    def __call__(self, index):
        return self._run_wrapper(index)

    def _run_wrapper(self, index):
        logger.info("Orbit {0}".format(index))

        # unpack input argument dictionary
        import gary.potential as gp
        potential = gp.load(os.path.join(self.cache_path, self.config.potential_filename))

        # read out just this initial condition
        norbits = len(self.w0)
        allfreqs = np.memmap(self.cache_file, mode='r',
                             shape=(norbits,), dtype=self.cache_dtype)

        # short-circuit if this orbit is already done
        if allfreqs['success'][index]:
            logger.debug("Orbit {0} already successfully completed.".format(index))
            return None

        # Only pass in things specified in _run_kwargs (w0 and potential required)
        kwargs = dict([(k,self.config[k]) for k in self.config.keys() if k in self._run_kwargs])
        res = self.run(w0=self.w0[index], potential=potential, **kwargs)
        res['index'] = index

        # cache res into a tempfile, return name of tempfile
        tmpfile = os.path.join(self._tmpdir, "{0}-{1}.pickle".format(self.__class__.__name__, index))
        with open(tmpfile, 'w') as f:
            pickle.dump(res, f)
        return tmpfile

    def status(self):
        """
        Prints out (to the logger) the status of the current run of the experiment.
        """

        d = self.read_cache()

        # numbers
        nsuccess = d['success'].sum()
        nfail = ((d['success'] is False) & (d['error_code'] > 0)).sum()

        # TODO: why don't logger.info() calls work here??
        # logger.info("------------- {0} Status -------------".format(self.__class__.__name__))
        # logger.info("Total number of orbits: {0}".format(len(d)))
        # logger.info("Successful: {0}".format(nsuccess))
        # logger.info("Failures: {0}".format(nfail))

        # for ecode in sorted(self.error_codes.keys()):
        #     nfail = (d['error_code'] == ecode).sum()
        #     logger.info("\t({0}) {1}: {2}".format(ecode, self.error_codes[ecode], nfail))

        print("------------- {0} Status -------------".format(self.__class__.__name__))
        print("Total number of orbits: {0}".format(len(d)))
        print("Successful: {0}".format(nsuccess))
        print("Failures: {0}".format(nfail))

        for ecode in sorted(self.error_codes.keys()):
            nfail = (d['error_code'] == ecode).sum()
            print("\t({0}) {1}: {2}".format(ecode, self.error_codes[ecode], nfail))

    # ------------------------------------------------------------------------
    # Subclasses must implement:

    @abstractproperty
    def error_codes(self):
        """ A dict mapping from integer error code to string describing the error """

    @abstractproperty
    def cache_dtype(self):
        """ The (numpy) dtype of the memmap'd cache file """

    @abstractproperty
    def _run_kwargs(self):
        """ A list of the names of the keyword arguments used in `run()` (below) """

    @abstractproperty
    def config_defaults(self):
        """ A dict of configuration defaults """

    @abstractclassmethod
    def run(cls, w0, potential, **kwargs):
        """ (classmethod) Run the experiment on a single orbit """

# ----------------------------------------------------------------------------

class ExperimentRunner(object):

    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite",
                        default=False, help="DESTROY. DESTROY. (default = False)")
    parser.add_argument("--seed", dest="seed", default=42, type=int,
                        help="Seed for random number generators.")

    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Use an MPI pool.")
    parser.add_argument("--path", dest="path", type=str, required=True,
                        help="Path to cache everything to (e.g., where to save the "
                             "initial conditions grid).")
    parser.add_argument("--config-filename", dest="config_filename", type=str, default=None,
                        help="Name of the config file (relative to the path).")

    parser.add_argument("--index", dest="index", type=str, default=None,
                        help="Specify a subset of orbits to run, e.g., "
                             "--index=20:40 to do only orbits 20-39.")

    def _parse_args(self):
        # Define parser object
        return self.parser.parse_args()

    def __init__(self, ExperimentClass):
        args = self._parse_args()

        np.random.seed(args.seed)

        # Set logger level based on verbose flags
        if args.verbose:
            logger.setLevel(logging.DEBUG)
        elif args.quiet:
            logger.setLevel(logging.ERROR)
        else:
            logger.setLevel(logging.INFO)

        # get a pool object for multiprocessing / MPI
        pool = get_pool(mpi=args.mpi)
        if args.mpi:
            logger.info("|----------- Using MPI -----------|")
        else:
            logger.info("|----------- Running in serial -----------|")

        if args.index is None:
            index = None
        else:
            try:
                index = slice(*map(int, args.index.split(":")))
            except:
                try:
                    index = np.array(map(int,args.index.split(",")))
                except:
                    index = None

        # Instantiate the experiment class
        with ExperimentClass(cache_path=args.path,
                             config_filename=args.config_filename,
                             overwrite=args.overwrite) as experiment:
            norbits = len(experiment.w0)

            if index is None:
                indices = np.arange(norbits,dtype=int)
            else:
                indices = np.arange(norbits,dtype=int)[index]

            try:
                pool.map(experiment, indices, callback=experiment.callback)
            except:
                pool.close()
                logger.error("Unexpected error!")
                raise

        sys.exit(0)
