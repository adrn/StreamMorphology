# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Project
import gary.potential as gp
from gary.units import galactic

__all__ = ['potential_registry']

# built-in potentials
potential_registry = dict()

p = gp.LeeSutoTriaxialNFWPotential(v_c=0.239225, r_s=30.,
                                   a=1., b=0.8, c=0.6,
                                   units=galactic)
potential_registry['triaxial-NFW'] = p

p = gp.TriaxialMWPotential(units=galactic)
potential_registry['triaxial-NFW-disk-bulge'] = p
