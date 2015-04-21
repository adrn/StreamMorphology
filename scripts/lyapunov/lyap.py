# coding: utf-8

""" Map the Lyapunov exponent for a given set of initial conditions. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import logging
import sys

# Third-party
from astropy import log as logger

# project
from streammorphology.util import main, get_parser, callback
from streammorphology.lyapunov import worker, parser_arguments, dtype

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

dargs = dict(args._get_kwargs())
main(worker=worker, callback=callback, path=dargs.pop('path'),
     cache_filename='alllyap.dat', cache_dtype=dtype,
     mpi=dargs.pop('mpi'), overwrite=dargs.pop('overwrite'), seed=dargs.pop('seed'),
     **dargs)

sys.exit(0)
