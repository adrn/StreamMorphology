# coding: utf-8

""" Map the KLD mixing time for a given set of initial conditions. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import logging
import sys

# Third-party
from astropy import log as logger

# project
from streammorphology.util import main, get_parser
from streammorphology.ensemble import worker, parser_arguments, memmap_shape

parser = get_parser()
for args,kwargs in parser_arguments:
    parser.add_argument(*args, **kwargs)

args = parser.parse_args()

# Set logger level based on verbose flags
if args.verbose:
    logger.setLevel(logging.DEBUG)
elif args.quiet:
    logger.setLevel(logging.ERROR)
else:
    logger.setLevel(logging.INFO)

try:
    main(worker=worker, path=args.path,
         cache_shape=memmap_shape, cache_filename='allkld.dat',
         mpi=args.mpi, overwrite=args.overwrite, seed=args.seed,
         **kwargs)
except:
    sys.exit(1)

sys.exit(0)
