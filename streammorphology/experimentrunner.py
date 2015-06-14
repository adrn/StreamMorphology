# coding: utf-8

""" ...explain... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from argparse import ArgumentParser
import sys
import logging

# Third-party
import numpy as np
from astropy import log as logger
from gary.util import get_pool

__all__ = ['ExperimentRunner']

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
                logger.error("Unexpected error: {0}".format(sys.exc_info()[0]))
            finally:
                pool.close()

        sys.exit(0)
