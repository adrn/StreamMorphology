# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
from astropy.coordinates.angles import rotation_matrix
import gary.integrate as gi
import numpy as np
from scipy.signal import argrelmax, argrelmin
from sklearn.neighbors import KernelDensity

from .fast_ensemble import ensemble_integrate

__all__ = ['create_ball', 'nearest_pericenter', 'align_ensemble']

def create_ball(w0, potential, N=1000, m_scale=1E4):
    menc = potential.mass_enclosed(w0)
    rscale = (m_scale / menc)**(1/3.) * np.sqrt(np.sum(w0[:3]**2))
    vscale = (m_scale / menc)**(1/3.) * np.sqrt(np.sum(w0[3:]**2))

    ball_w0 = np.zeros((N,6))
    ball_w0[:,:3] = np.random.normal(w0[:3], rscale, size=(N,3))
    ball_w0[:,3:] = np.random.normal(w0[3:], vscale, size=(N,3))

    return np.vstack((w0,ball_w0))

def nearest_pericenter(w0, potential, dt, T):
    """
    Find the nearest pericenter to w0
    """

    t,w = potential.integrate_orbit(w0, dt=dt, nsteps=T*10,
                                    Integrator=gi.DOPRI853Integrator)

    r = np.sqrt(np.sum(w[:,0,:3]**2, axis=-1))
    peris = argrelmin(r)[0]

    return w[peris[0], 0]

def compute_align_matrix(w):
    """
    Given a single phase-space position, compute the rotation matrix that
    orients the angular momentum with the z axis and places the point
    along the x axis.

    Parameters
    ----------
    w : array_like
    """

    if w.ndim > 1:
        raise ValueError("Input phase-space position should be 1D.")

    x = w[:3].copy()
    v = w[3:].copy()

    # first rotate about z to put on x-z plane
    theta = np.arctan2(x[1], x[0]) * u.radian
    R1 = rotation_matrix(theta, 'z')
    x = np.asarray(R1.dot(x))[0]
    v = np.asarray(R1.dot(v))[0]

    # now rotate about y to put on x axis
    theta = np.arctan2(x[2], x[0]) * u.radian
    R2 = rotation_matrix(-theta, 'y')
    x = np.asarray(R2.dot(x))[0]
    v = np.asarray(R2.dot(v))[0]

    # now align L with z axis
    # theta = np.arccos(L[2] / np.sqrt(np.sum(L**2))) * u.radian
    L = np.cross(x, v)
    theta = np.arctan2(L[2], L[1]) * u.radian
    R3 = rotation_matrix(theta - 90*u.deg, 'x')
    x = np.asarray(R3.dot(x))[0]
    v = np.asarray(R3.dot(v))[0]

    return R3*R2*R1

def align_ensemble(ws):
    R = compute_align_matrix(ws[-1,0])
    new_x = np.array(R.dot(ws[-1,:,:3].T).T)
    new_v = np.array(R.dot(ws[-1,:,3:].T).T)
    return new_x, new_v

def do_the_kld(nkld, ball_w0, potential, dt, nsteps, kde_bandwidth,
               density_thresholds):
    """
    Parameters
    ----------
    nkld : int
        Number of times to evaluate the KLD.
    ...TODO
    """

    ww = np.ascontiguousarray(ball_w0.copy())
    nensemble = ww.shape[0]

    # energy of parent orbit
    E0 = float(np.squeeze(potential.total_energy(ww[0,:3],ww[0,3:])))
    predicted_density = lambda x, E0: np.sqrt(E0 - potential(x))

    kld_idx = np.linspace(0, nsteps, nkld+1).astype(int)

    # container to store fraction of stars with density above each threshold
    frac_above_dens = np.zeros((nkld,len(density_thresholds)))

    t = np.empty(nkld+1)
    t[0] = 0.
    kld = np.empty(nkld)
    for i in range(nkld):
        # number of steps to advance the ensemble
        dstep = kld_idx[i+1] - kld_idx[i]
        www = ensemble_integrate(potential.c_instance, ww, dt, dstep, 0.)

        # store the time
        t[i+1] = t[i] + dt*dstep

        # build an estimate of the configuration space density of the ensemble
        kde = KernelDensity(kernel='epanechnikov', bandwidth=kde_bandwidth)
        kde.fit(www[:,:3])
        kde_densy = np.exp(kde.score_samples(www[:,:3]))

        # TODO: here, need to figure out # of stars with density > threshold
        for j,h in enumerate(density_thresholds):
            frac_above_dens[i,j] = (kde_densy > h).sum() / float(nensemble)

        p_densy = predicted_density(www[:,:3], E0)
        D = np.log(kde_densy / p_densy)
        KLD = D[np.isfinite(D)].sum() / float(nensemble)
        kld[i] = KLD

        ww = www.copy()

    return t[1:], kld
