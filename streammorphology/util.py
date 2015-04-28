# coding: utf-8

""" General utilities. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import logging
import os
import sys

# Third-party
from astropy import log as logger
import numpy as np
from gary.util import get_pool

__all__ = ['main', 'get_parser']

def main(worker, path, cache_filename, cache_dtype, callback=None, index=None,
         mpi=False, overwrite=False, seed=42, verbose=False, quiet=False, str_index=None,
         **kwargs):
    np.random.seed(seed)

    # Set logger level based on verbose flags
    if verbose:
        logger.setLevel(logging.DEBUG)
    elif quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    # get a pool object for multiprocessing / MPI
    pool = get_pool(mpi=mpi)
    if mpi:
        logger.info("|----------- Using MPI -----------|")
    else:
        logger.info("|----------- Running in serial -----------|")

    if str_index is None:
        index = None
    else:
        try:
            index = slice(*map(int, str_index.split(":")))
        except:
            try:
                index = np.array(map(int,str_index.split(",")))
            except:
                # index is None
                index = None

    # file that will hold all results
    cache_path = os.path.join(path, cache_filename)

    # path to initial conditions cache
    w0_filename = os.path.join(path, 'w0.npy')
    w0 = np.load(w0_filename)

    # path to potential name file
    pot_filename = os.path.join(path, 'potential.yml')

    norbits = len(w0)
    logger.info("Number of orbits: {}".format(norbits))

    if os.path.exists(cache_path) and overwrite:
        os.remove(cache_path)

    logger.info("Cache dtype: {0}".format(cache_dtype))

    if not os.path.exists(cache_path):
        # make sure memmap file exists
        d = np.memmap(cache_path, mode='w+', dtype=cache_dtype, shape=(norbits,))
        d[:] = np.zeros(shape=(norbits,), dtype=cache_dtype)

    if index is None:
        tasks = [dict(index=i, w0_filename=w0_filename,
                      cache_filename=cache_path,
                      potential_filename=pot_filename, **kwargs)
                 for i in range(norbits)]
    else:
        tasks = [dict(index=i, w0_filename=w0_filename,
                      cache_filename=cache_path,
                      potential_filename=pot_filename, **kwargs)
                 for i in np.arange(norbits,dtype=int)[index]]

    pool.map(worker, tasks, callback=callback)
    pool.close()

    sys.exit(0)

def get_parser():
    from argparse import ArgumentParser

    # Define parser object
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

    parser.add_argument("--index", dest="index", type=str, default=None,
                        help="Specify a subset of orbits to run, e.g., "
                             "--index=20:40 to do only orbits 20-39.")

    return parser

def callback(result):
    if result is None:
        return

    memmap = np.memmap(result['mmap_filename'], mode='r+',
                       shape=(result['norbits'],), dtype=result['dtype'])

    logger.debug("Flushing {0} to output array...".format(result['index']))
    if result['error_code'] != 0.:
        # error happened
        for key in memmap.dtype.names:
            if key in result:
                memmap[key][result['index']] = result[key]

    else:
        # all is well
        for key in memmap.dtype.names:
            memmap[key][result['index']] = result[key]

    # flush to output array
    memmap.flush()
    logger.debug("...flushed, washing hands.")

    del memmap  # explicitly try to close?
