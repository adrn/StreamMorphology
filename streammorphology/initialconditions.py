# coding: utf-8

""" Module for generating initial conditions for frequency mapping. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy import log as logger
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

__all__ = ['tube_grid', 'box_grid']

def tube_grid(E, potential, dx, dz):
    """
    Generate a grid of points in the :math:`x-z` plane (:math:`y=0`),
    starting with initial velocities :math:`v_x = v_z = 0`. :math`v_y`
    is solved for as:

    .. math::

        v_y = \sqrt(2(E - \Phi(x,0,z)))

    where :math:`E` is the energy and :math:`\Phi` is the potential.

    Parameters
    ----------
    E : numeric
        Energy of the orbits - defines the zero-velocity curve within
        which the initial conditions are determined.
    potential : :class:`~gary.potential.Potential`
        A :class:`~gary.potential.Potential` subclass instance.
    dx : numeric
        Step size in :math:`x`.
    dz : numeric
        Step size in :math:`z`.

    """

    # find maximum x on z=0
    def func(x):
        return (E - potential.value(np.array([[x[0],0,0]])))**2
    res = minimize(func, x0=[10.], method='powell')
    if not res.success:
        raise ValueError("Failed to find boundary of ZVC on x-axis.")
    max_x = res.x

    xgrid = np.arange(0.1, max_x+dx, dx)

    # compute ZVC boundary for each x
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
            raise ValueError("Failed to find boundary of ZVC for x={}.".format(xg))

        logger.debug("Max. z: {}".format(max_z))
        zgrid = np.arange(0.1, max_z, dz)
        xs = np.zeros_like(zgrid) + xg
        try:
            xz = np.hstack((xz, np.vstack((xs,zgrid))))
        except NameError:
            xz = np.vstack((xs,zgrid))

    xyz = np.zeros((xz.shape[-1],3))
    xyz[:,0] = xz[0]
    xyz[:,2] = xz[1]

    # now, for each grid point, compute the y velocity
    vxyz = np.zeros_like(xyz)
    vxyz[:,1] = np.sqrt(2*(E - potential.value(xyz)))

    return np.hstack((xyz, vxyz))

def box_grid(E, potential, approx_num=1000):
    """
    Generate a grid of points on an equipotential surface starting with
    zero initial velocity. The angular positions :math:`(\phi,\theta)` are
    sampled randomly (close to uniformly) over one octant of the equipotential
    surface, and the initial radius is determined by solving the equation
    below for :math`r`:

    .. math::

        \Phi(r,\phi,\theta) - E = 0

    Parameters
    ----------
    E : numeric
        Energy of the orbits - defines the zero-velocity curve within
        which the initial conditions are sampled.
    potential : :class:`~gary.potential.Potential`
        A :class:`~gary.potential.Potential` subclass instance.
    approx_num : int
        Approximate total number of grid points to generate. Final grid
        of initial conditions might have slightly less than this number
        of points.

    """

    # only want points in one octant, but algorithm generates points over whole sphere
    approx_num *= 8

    # generate points roughly evenly distributed on an octant using the golden
    #   ratio / spiral method
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(approx_num)
    z = np.linspace(1 - 1.0 / approx_num, 1.0 / approx_num - 1, approx_num)
    radius = np.sqrt(1 - z * z)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # convert to usual spherical angles
    phi = np.arctan2(y, x)
    theta = np.arccos(z)

    # only take one octant
    ix = (phi > 0) & (phi < np.pi/2.) & (theta < np.pi/2.)
    phi = phi[ix]
    theta = theta[ix]
    # phi,theta = map(np.ravel, np.meshgrid(phi,theta))

    def func(r,phi,theta):
        x = r[0]*np.cos(phi)*np.sin(theta)
        y = r[0]*np.sin(phi)*np.sin(theta)
        z = r[0]*np.cos(theta)
        return (E - potential.value(np.array([[x,y,z]])))**2

    r = np.zeros_like(phi)
    for i,p,t in zip(np.arange(len(phi)),phi,theta):
        res = minimize(func, x0=[25.], method='powell', args=(p,t))
        r[i] = res.x

    x = r*np.cos(phi)*np.sin(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(theta)
    v = np.zeros_like(x)

    return np.array([x,y,z,v,v,v]).T.copy()
