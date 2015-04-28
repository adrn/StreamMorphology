# coding: utf-8

""" Map the KLD mixing time for a given set of initial conditions. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import sys

# project
from streammorphology.util import main, get_parser, callback
from streammorphology.ensemble import worker, parser_arguments, dtype

parser = get_parser()
for args,kwargs in parser_arguments:
    parser.add_argument(*args, **kwargs)

args = parser.parse_args()

dargs = dict(args._get_kwargs())
main(worker=worker, callback=callback, path=dargs.pop('path'),
     cache_filename='allkld.dat', cache_dtype=dtype, str_index=dargs.pop('index'),
     mpi=dargs.pop('mpi'), overwrite=dargs.pop('overwrite'), seed=dargs.pop('seed'),
     verbose=dargs.pop('verbose'), quiet=dargs.pop('quiet'),
     **dargs)

sys.exit(0)
