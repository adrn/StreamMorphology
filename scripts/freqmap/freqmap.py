# coding: utf-8

from __future__ import division, print_function

"""
Frequency map a potential. Before calling this module, you'll need to generate
a grid of initial conditions or make sure the grid you have is in the correct
format. You also need to have a text file containing the name of the potential
that you used to generate the initial conditions (the name has to be one specified
in the ``potential_registry``).

For example, you might do::

    python scripts/make_grid.py -E -0.14 --potential=triaxial-NFW \
    --ic-func=tube_grid_xz --dx=5 --dz=5

and then run this module on::

    python scripts/freqmap/freqmap.py --path=output/freqmap/triaxial-NFW/E-0.140_tube_grid_xz/

"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import logging
import sys

# Third-party
from astropy import log as logger

# project
from streammorphology.util import main, get_parser
from streammorphology.freqmap import worker, callback, parser_arguments, dtype

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
     cache_filename='allfreqs.dat', cache_dtype=dtype,
     mpi=dargs.pop('mpi'), overwrite=dargs.pop('overwrite'), seed=dargs.pop('seed'),
     **dargs)

sys.exit(0)
