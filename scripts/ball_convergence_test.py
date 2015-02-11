# coding: utf-8

""" Convergence test to compute the KLD between configuration space density
    of particle balls.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
from astropy.coordinates.angles import rotation_matrix
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelmax, argrelmin

# Custom
import gary.dynamics as gd
import gary.integrate as gi

# project
from streammorphology.potential import potential_registry
potential = potential_registry['triaxial-NFW']

def create_ball(w0, potential, N=1000):
    menc = potential.mass_enclosed(w0)
    rscale = (2.5E4 / (3*menc))**(1/3.) * np.sqrt(np.sum(w0[:3]**2))
    vscale = (2.5E4 / (3*menc))**(1/3.) * np.sqrt(np.sum(w0[3:]**2))

    ball_w0 = np.zeros((N,6))

    ball_w0[:,:3] = np.random.normal(w0[:3], rscale, size=(N,3))
    ball_w0[:,3:] = np.random.normal(w0[3:], vscale, size=(N,3))

    return np.vstack((w0,ball_w0))

def align_particles(ws, ix=-1):
    # assumes ws[:,0] is the "progenitor" orbit
    new_cen_x = ws[:,0,:3].copy()
    new_cen_v = ws[:,0,3:].copy()

    end_w = ws[ix]

    # put endpoint on x axis in x-z plane

    # first about y
    theta = np.arctan2(new_cen_x[ix,2],new_cen_x[ix,0]) * u.radian
    R1 = rotation_matrix(-theta, 'y')
    new_cen_x = R1.dot(new_cen_x.T).T
    new_cen_v = R1.dot(new_cen_v.T).T

    # then about z
    theta = np.arctan2(new_cen_x[ix,1],new_cen_x[ix,0]) * u.radian
    R2 = rotation_matrix(theta, 'z')
    new_cen_x = R2.dot(new_cen_x.T).T
    new_cen_v = R2.dot(new_cen_v.T).T

    # now align L
    L = np.cross(new_cen_x[ix], new_cen_v[ix])[0]
    theta = np.arccos(L[2] / np.sqrt(np.sum(L**2))) * u.radian
    R3 = rotation_matrix(-theta, 'x')
    new_cen_x = np.array(R3.dot(new_cen_x.T).T)

    R = R3*R2*R1

    new_end_ptcl_x = np.array(R.dot(end_w[:,:3].T).T)
    return new_cen_x, new_end_ptcl_x

def make_aligned_figure(slow_ball, fast_ball, ix=-1):
    slow_cen, rotated_slow = align_particles(slow_ball, ix=ix)
    fast_cen, rotated_fast = align_particles(fast_ball, ix=ix)

    fig,all_axes = plt.subplots(2,2,sharex=True,sharey=True,figsize=(12,12))

    axes = all_axes[0]
    axes[0].plot(rotated_slow[:,0], rotated_slow[:,1], marker='.', linestyle='none', alpha=0.25)
    axes[0].plot(slow_cen[-1000:,0], slow_cen[-1000:,1], marker=None, linestyle='-')

    axes[1].plot(rotated_fast[:,0], rotated_fast[:,1], marker='.', linestyle='none', alpha=0.25)
    axes[1].plot(fast_cen[-1000:,0], fast_cen[-1000:,1], marker=None, linestyle='-')

    axes[0].set_xlabel("$x'$ [kpc]")
    axes[0].set_ylabel("$y'$ [kpc]")
    axes[1].set_xlabel("$x'$ [kpc]")
    axes[0].set_title("Slow", fontsize=18)
    axes[1].set_title("Fast", fontsize=18)

    axes = all_axes[1]
    axes[0].plot(rotated_slow[:,1], rotated_slow[:,2], marker='.', linestyle='none', alpha=0.25)
    axes[0].plot(slow_cen[-1000:,1], slow_cen[-1000:,2], marker=None, linestyle='-')

    axes[1].plot(rotated_fast[:,1], rotated_fast[:,2], marker='.', linestyle='none', alpha=0.25)
    axes[1].plot(fast_cen[-1000:,1], fast_cen[-1000:,2], marker=None, linestyle='-')

    axes[0].set_xlabel("$y'$ [kpc]")
    axes[0].set_ylabel("$z'$ [kpc]")
    axes[1].set_xlabel("$y'$ [kpc]")

    fig.tight_layout()
    return fig

def kld(potential, E0, slow_ball, fast_ball, slow_t, fast_t, nbins=25):
    bins = np.linspace(-50, 50, nbins)
    dx = bins[1]-bins[0]

    bincenters = (bins[1:]+bins[:-1])/2.
    derp = np.meshgrid(bincenters,bincenters,bincenters)
    xyz = np.vstack([np.ravel(xxx) for xxx in derp]).T

    # compute normalized, expected number of particles per cell
    p_dens = np.sqrt(E0 - potential(xyz)) * dx**3
    good_ix = np.isfinite(p_dens)
    A = 1/p_dens[good_ix].sum()
    p_dens = A * p_dens[good_ix]

    # compute KLD for slow
    slow_D = []
    t_ixes_slow = np.linspace(0, slow_ball.shape[0]-1, 512).astype(int)
    for i in t_ixes_slow:
        H, edges = np.histogramdd(slow_ball[i,:,:3], bins=(bins,bins,bins))
        Hravel = np.ravel(H) / H.sum()

        D = Hravel[good_ix] * np.log(Hravel[good_ix] / p_dens)
        slow_D.append(D[np.isfinite(D)].sum())

    fast_D = []
    t_ixes_fast = np.linspace(0, fast_ball.shape[0]-1, 512).astype(int)
    for i in t_ixes_fast:
        H, edges = np.histogramdd(fast_ball[i,:,:3], bins=(bins,bins,bins))
        Hravel = np.ravel(H) / H.sum()

        D = Hravel[good_ix] * np.log(Hravel[good_ix] / p_dens)
        fast_D.append(D[np.isfinite(D)].sum())

    fig,ax = plt.subplots(1,1,figsize=(12,8))
    ax.plot(slow_t[t_ixes_slow], slow_D)
    ax.plot(fast_t[t_ixes_fast], fast_D)
    return fig

def main():
    # =============================================================================
    # Things to set

    evln_time = 20000.  # time to integrate balls (20 Gyr)

    slow_fast_w0 = np.array([[27.85, 0.0, 22.1, 0.0, 0.16914188537254274, 0.0],
                             [27.85, 0.0, 23.6, 0.0, 0.16023972330323624, 0.0]])

    path = "/Users/adrian/projects/morphology/output/ball_convergence/"

    # =============================================================================

    if not os.path.exists(path):
        os.makedirs(path)

    # compute energy of orbits -- should be the same
    E0s = potential.total_energy(slow_fast_w0[:,:3], slow_fast_w0[:,3:])
    if np.allclose(E0s):
        raise ValueError("Orbits should have same energy")
    E0 = E0s[0]

    # integrate orbits for 1.2 times the evolution time
    sf_t,sf_w = potential.integrate_orbit(slow_fast_w0, dt=1., nsteps=int(1.2*evln_time),
                                          Integrator=gi.DOPRI853Integrator)

    # identify first pericenter of each orbit, then apocenter closest to the evolution
    #   time away
    Rs = np.sqrt(np.sum(sf_w[:,:,:3].T**2, axis=0))

    # estimate apocenters and pericenters with relative min/maxima
    slow_apos,slow_pers = argrelmax(Rs[0])[0], argrelmin(Rs[0])[0]
    fast_apos,fast_pers = argrelmax(Rs[1])[0], argrelmin(Rs[1])[0]

    # -----------------------------------------------------------------------------
    # figure out integration time and initial conditions for slow and fast orbit
    slow_dt = 1.
    fast_dt = 1.

    # slow
    t1 = sf_t[slow_pers[0]]
    t2 = sf_t[slow_apos]
    t2 = sf_t[slow_apos[np.abs((t2-t1) - evln_time).argmin()]]

    slow_nsteps = int((t2-t1) / slow_dt)
    slow_w0 = sf_w[slow_pers[0], 0]

    # fast
    t1 = sf_t[fast_pers[0]]
    t2 = sf_t[fast_apos]
    t2 = sf_t[fast_apos[np.abs((t2-t1) - evln_time).argmin()]]

    fast_nsteps = int((t2-t1) / fast_dt)
    fast_w0 = sf_w[fast_pers[0], 1]
    # -----------------------------------------------------------------------------

    max_nparticles = 10000

    # sample balls around slow,fast initial conditions
    slow_ball0 = create_ball(slow_w0, potential, max_nparticles)
    fast_ball0 = create_ball(fast_w0, potential, max_nparticles)

    # integrate all orbits
    slow_t,all_slow_ball = potential.integrate_orbit(slow_ball0, dt=slow_dt, nsteps=slow_nsteps,
                                                     Integrator=gi.DOPRI853Integrator)

    fast_t,all_fast_ball = potential.integrate_orbit(fast_ball0, dt=fast_dt, nsteps=fast_nsteps,
                                                     Integrator=gi.DOPRI853Integrator)

    for nparticles in [1000,10000]:
        slow_ball = all_slow_ball[:nparticles]
        fast_ball = all_fast_ball[:nparticles]

        fig = make_aligned_figure(slow_ball, fast_ball)
        fig.savefig(os.path.join(path, "aligned_{0}ptcl.png".format(nparticles)))

        for nbins in [11,19,27,35,43,51]:
            fig = kld(potential, E0, slow_ball, fast_ball, slow_t, fast_t, nbins=nbins)
            fig.savefig(os.path.join(path, "kld_{0}ptcl_{1}bins.png".format(nparticles,nbins)))
            plt.close('all')

if __name__ == '__main__':
    main()
