# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Project
import gary.potential as gp
from gary.units import galactic

__all__ = ['potential_register']

# built-in potentials
potential_register = dict()

p = gp.LeeSutoTriaxialNFWPotential(v_c=0.239225, r_s=30.,
                                   a=1., b=0.8, c=0.6,
                                   units=galactic)
potential_register['triaxial-NFW'] = p

p = gp.TriaxialMWPotential(units=galactic)
potential_register['triaxial-NFW-disk-bulge'] = p
