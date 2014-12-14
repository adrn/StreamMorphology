# coding: utf-8

""" Module for generating initial conditions for frequency mapping. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# Project
# ...

__all__ = ['']

def loop_grid(E, potential, Naxis=100):
    """ Generate a grid of points in the x-z plane (y=0), starting with initial
        velocities vx = vz = 0. vy is solved for as:

            vy = sqrt(2(E - potential(x,0,z)))

        Parameters
        ----------
        E : numeric
            Energy of the orbits - defines the zero-velocity curve within
            which the initial conditions are drawn.
        potential : gary.Potential
            A `gary.Potential` subclass instance.
        Naxis : int
            Number of grid points along each axis. Final grid of initial
            conditions will have < Naxis**2 points.
    """

    # find maximum x on z=0
    def func(x):
        return (E - potential.value(np.array([[x[0],0,0]])))**2
    res = minimize(func, x0=[10.], method='powell')
    if not res.success:
        raise ValueError("Failed to find boundary of ZVC on x-axis.")
    max_x = res.x

    xgrid = np.linspace(0.1, max_x, Naxis)

    # compute ZVC boundary for each x
    dz = None
    for xg in xgrid:
        # find maximum allowed z along x=xx
        def func(x):
            return (E - potential.value(np.array([[xg,0,x[0]]])))**2
        res = minimize(func, x0=[25.], method='powell')
        max_z = np.abs(res.x)
        if not res.success or max_z == 25.:
            vals = np.linspace(0.1,100)
            plt.clf()
            plt.plot(vals,[func([derp]) for derp in vals])
            plt.show()
            raise ValueError("Failed to find boundary of ZVC for x={}.".format(xx))

        logger.debug("Max. z: {}".format(max_z))
        if dz is None:
            zgrid = np.linspace(0.1, max_z, Naxis)
            dz = zgrid[1] - zgrid[0]

            xs = np.zeros_like(zgrid) + xg
            xz = np.vstack((xs,zgrid))
        else:
            zgrid = np.arange(0.1, max_z, dz)
            xs = np.zeros_like(zgrid) + xg
            xz = np.hstack((xz, np.vstack((xs,zgrid))))

    xyz = np.zeros((xz.shape[-1],3))
    xyz[:,0] = xz[0]
    xyz[:,2] = xz[1]

    # now, for each grid point, compute the y velocity
    vxyz = np.zeros_like(xyz)
    vxyz[:,1] = np.sqrt(2*(E - potential.value(xyz)))

    return np.hstack((xyz, vxyz))

def box_grid():
    pass
