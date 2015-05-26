# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy import log as logger
import astropy.units as u
from astropy.coordinates.angles import rotation_matrix
import gary.integrate as gi
import gary.dynamics as gd
import numpy as np
from scipy.signal import argrelmin, argrelmax
from scipy.stats import kurtosis, skew
from sklearn.neighbors import KernelDensity

from .fast_ensemble import ensemble_integrate

__all__ = ['create_ball', 'nearest_pericenter', 'align_ensemble', 'do_the_kld',
           'prepare_parent_orbit']

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

    t,w = potential.integrate_orbit(w0, dt=dt, nsteps=int(T*10),
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

default_metrics = dict(mean=np.mean,
                       median=np.median,
                       skewness_log=lambda x: skew(np.log10(x)),
                       kurtosis_log=lambda x: kurtosis(np.log10(x)),
                       nabove_mean=lambda dens: (dens >= np.mean(dens)).sum(),
                       nbelow_mean=lambda dens: (dens <= np.mean(dens)).sum())
def do_the_kld(ball_w0, potential, dt, nsteps, nkld, kde_bandwidth,
               metrics=default_metrics, return_all_density=False):
    ww = np.ascontiguousarray(ball_w0.copy())
    nensemble = ww.shape[0]

#     kld_idx = np.append(np.linspace(0, nsteps//4, nkld//2+1),
#                         np.linspace(nsteps//4, nsteps, nkld//2+1)[1:]).astype(int)
    kld_idx = np.linspace(0, nsteps, nkld+1).astype(int)

    # sort so I preserve some order around here
    metric_names = sorted(metrics.keys())

    # if set, store and return all of the density values
    if return_all_density:
        all_density = np.zeros((nkld, nensemble))

    # container to store fraction of stars with density above each threshold
    dtype = []
    for name in metric_names:
        dtype.append((name,'f8'))
    metric_data = np.zeros(nkld, dtype=dtype)

    # store energies
    Es = np.empty((nkld+1,nensemble))
    Es[0] = potential.total_energy(ball_w0[:,:3], ball_w0[:,3:])

    # time container
    t = np.empty(nkld)
    for i in range(nkld):
        logger.debug("KLD step: {0}/{1}".format(i+1, nkld))

        # number of steps to advance the ensemble -- not necessarily constant
        dstep = kld_idx[i+1] - kld_idx[i]
        www = ensemble_integrate(potential.c_instance, ww, dt, dstep, 0.)

        Es[i+1] = potential.total_energy(www[:,:3], www[:,3:])

        # store the time
        if i == 0:
            t[i] = dt*dstep
        else:
            t[i] = t[i-1] + dt*dstep

        # build an estimate of the configuration space density of the ensemble
        kde = KernelDensity(kernel='epanechnikov', bandwidth=kde_bandwidth)
        kde.fit(www[:,:3])

        # evaluate density at the position of the particles
        ln_densy = kde.score_samples(www[:,:3])
        density = np.exp(ln_densy)

        if return_all_density:
            all_density[i] = density

        # evaluate the metrics and save
        for name in metric_names:
            metric_data[name][i] = metrics[name](density)

        # reset initial conditions
        ww = www.copy()

    if return_all_density:
        return t, metric_data, Es, all_density
    else:
        return t, metric_data, Es

def prepare_parent_orbit(w0, potential, nperiods, nsteps_per_period):
    t,w = potential.integrate_orbit(w0, dt=0.5, nsteps=10000)
    r = np.sqrt(np.sum(w[:,0,:3]**2, axis=-1))
    T = gd.peak_to_peak_period(t, r)

    dt = T / nsteps_per_period
    nsteps = int(round(nperiods * T / dt))

    ix = argrelmin(r)[0]
#     ix = argrelmax(r)[0]
    new_w0 = w[ix[0],0]

    t,w = potential.integrate_orbit(new_w0, dt=dt, nsteps=nsteps + int(0.75*nsteps_per_period),
                                    Integrator=gi.DOPRI853Integrator)
    r = np.sqrt(np.sum(w[:,0,:3]**2, axis=-1))
    ix = argrelmax(r)[0]

    # if len(ix) < nperiods-1:
    #     raise ValueError("Dooooooood...")

    nsteps = ix[-1]

    return new_w0,dt,nsteps
