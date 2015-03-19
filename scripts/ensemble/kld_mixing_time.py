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
from streammorphology.ensemble import worker, parser_arguments, get_dtype

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
main(worker=worker, path=dargs.pop('path'),
     cache_filename='allkld.dat', cache_dtype=get_dtype(args.nkld,args.ndensity_threshold),
     mpi=dargs.pop('mpi'), overwrite=dargs.pop('overwrite'), seed=dargs.pop('seed'),
     **dargs)

sys.exit(0)
