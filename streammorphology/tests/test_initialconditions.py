# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import numpy as np
import matplotlib.pyplot as plt

# Project
import gary.potential as gp
from gary.units import galactic
from ..initialconditions import tube_grid_xz, box_grid

plot_path = "output/tests/initialconditions"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

potential = gp.LeeSutoTriaxialNFWPotential(v_c=0.22, r_s=30.,
                                           a=1., b=1., c=0.8, units=galactic)

def test_tube():
    plt.figure(figsize=(10,10))

    for E in np.linspace(-0.12, -0.2, 5):
        w0 = tube_grid_xz(E=E, potential=potential, dx=1., dz=1.)
        Es = potential.total_energy(w0[:,:3], w0[:,3:])
        np.testing.assert_allclose(Es, E)

        plt.scatter(w0[:,0], w0[:,2], c='k', alpha=0.5)
        plt.savefig(os.path.join(plot_path, "tube_E{:.2f}.png".format(E)))
        plt.clf()

def test_box():
    from mpl_toolkits.mplot3d import Axes3D

    for E in np.linspace(-0.12, -0.2, 5):
        w0 = box_grid(E=E, potential=potential, approx_num=1024)
        print(w0.shape)
        Es = potential.total_energy(w0[:,:3], w0[:,3:])
        np.testing.assert_allclose(Es, E)

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111,projection='3d')
        ax.plot(w0[:,0], w0[:,1], w0[:,2], c='k',
                alpha=0.5, marker='o', linestyle='none')
        ax.elev = 45
        ax.azim = 45
        fig.savefig(os.path.join(plot_path, "box_E{:.2f}.png".format(E)))
