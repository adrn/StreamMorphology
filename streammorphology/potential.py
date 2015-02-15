# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import numpy as np

# Project
import gary.potential as gp
from gary.units import galactic

__all__ = ['potential_registry']

# built-in potentials
potential_registry = dict()

# --------------------------------------------------------------
p1 = gp.LeeSutoTriaxialNFWPotential(v_c=(150*u.km/u.s).to(u.kpc/u.Myr).value,
                                    r_s=40., a=1., b=0.77, c=0.55,
                                    units=galactic)
potential_registry['triaxial-NFW'] = p1

# --------------------------------------------------------------
params = p1.parameters
params.pop('R')
p2 = gp.LeeSutoTriaxialNFWPotential(units=galactic, phi=np.pi/2.,
                                    **params)
potential_registry['triaxial-NFW-yz'] = p2

# --------------------------------------------------------------
p = gp.TriaxialMWPotential(units=galactic)
potential_registry['triaxial-NFW-disk-bulge'] = p

# --------------------------------------------------------------
p = gp.LogarithmicPotential(v_c=np.sqrt(2)*(140.372*u.km/u.s).decompose(galactic).value,
                            r_h=5.87963,
                            q1=0.872614, q2=1., q3=1.16395,
                            phi=1.58374, units=galactic)
potential_registry['via-lactea-log'] = p
