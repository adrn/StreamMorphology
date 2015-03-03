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

__all__ = ['create_ball', 'peri_to_apo', 'align_ensemble']

def create_ball(w0, potential, N=1000, m_scale=1E4):
    menc = potential.mass_enclosed(w0)
    rscale = (m_scale / menc)**(1/3.) * np.sqrt(np.sum(w0[:3]**2))
    vscale = (m_scale / menc)**(1/3.) * np.sqrt(np.sum(w0[3:]**2))

    ball_w0 = np.zeros((N,6))
    ball_w0[:,:3] = np.random.normal(w0[:3], rscale, size=(N,3))
    ball_w0[:,3:] = np.random.normal(w0[3:], vscale, size=(N,3))

    return np.vstack((w0,ball_w0))

def peri_to_apo(w0, potential, evolution_time, dt=1.):
    """
    Find the nearest pericenter to w0, then figure out the number of steps
    to integrate for to evolve for the specified evolution time and end
    at apocenter.
    """

    t,w = potential.integrate_orbit(w0, dt=dt, nsteps=int(evolution_time),
                                    Integrator=gi.DOPRI853Integrator)

    r = np.sqrt(np.sum(w[:,0,:3]**2, axis=-1))
    apos = argrelmax(r)[0]
    pers = argrelmin(r)[0]

    t1 = t[pers[0]]
    t2 = t[apos]
    t2 = t[apos[np.abs((t2-t1) - evolution_time).argmin()]]

    nsteps = int((t2-t1) / dt)
    w0 = w[pers[0], 0]

    t,w = potential.integrate_orbit(w0, dt=dt, nsteps=nsteps,
                                    Integrator=gi.DOPRI853Integrator)
    r = np.sqrt(np.sum(w[:,0,:3]**2, axis=-1))
    apo_ixes = argrelmax(r)[0]

    return w0, dt, nsteps, apo_ixes

def align_ensemble(ws):
    # progenitor position
    new_cen_x = ws[0,:3].copy()
    new_cen_v = ws[0,3:].copy()

    # put endpoint on x axis in x-z plane

    # first about y
    theta = np.arctan2(new_cen_x[2],new_cen_x[0]) * u.radian
    R1 = rotation_matrix(-theta, 'y')
    new_cen_x = np.asarray(R1.dot(new_cen_x))[0]
    new_cen_v = np.asarray(R1.dot(new_cen_v))[0]

    # then about z
    theta = np.arctan2(new_cen_x[1],new_cen_x[0]) * u.radian
    R2 = rotation_matrix(theta, 'z')
    new_cen_x = np.asarray(R2.dot(new_cen_x))[0]
    new_cen_v = np.asarray(R2.dot(new_cen_v))[0]

    # now align L with z axis
    L = np.cross(new_cen_x, new_cen_v)
    theta = np.arccos(L[2] / np.sqrt(np.sum(L**2))) * u.radian
    R3 = rotation_matrix(theta, 'x')
    new_cen_x = np.asarray(R3.dot(new_cen_x))[0]
    new_cen_v = np.asarray(R3.dot(new_cen_v))[0]

    R = R3*R2*R1

    new_ws = np.array(R.dot(ws[:,:3].T).T)

    return new_ws

def do_the_kld(ball_w0, potential, apo_ixes, dt=1., kde_bandwidth=10.):
    ww = np.ascontiguousarray(ball_w0.copy())
    nensemble = ball_w0.shape[0]

    # energy of parent orbit
    E0 = float(np.squeeze(potential.total_energy(ww[0,:3],ww[0,3:])))
    predicted_density = lambda x, E0: np.sqrt(E0 - potential(x))

    klds = []
    ts = []
    apo_ixes = np.append([0], apo_ixes)
    t = 0.
    for i,left_step,right_step in zip(range(len(apo_ixes)-1), apo_ixes[:-1], apo_ixes[1:]):
        dstep = right_step - left_step
        www = ensemble_integrate(potential.c_instance, ww, dt, dstep, 0.)

        t += dt*dstep
        ts.append(t)

        kde = KernelDensity(kernel='epanechnikov', bandwidth=kde_bandwidth)
        kde.fit(www[:,:3])
        kde_densy = np.exp(kde.score_samples(www[:,:3]))

        p_densy = predicted_density(www[:,:3], E0)
        D = np.log(kde_densy / p_densy)
        KLD = D[np.isfinite(D)].sum() / float(nensemble)
        klds.append(KLD)

        ww = www.copy()

    return ts, klds
