# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np

__all__ = ['create_ball']

def create_ball(w0, potential, N=1000, m_scale=1E4):
    menc = potential.mass_enclosed(w0)
    rscale = (m_scale / (3*menc))**(1/3.) * np.sqrt(np.sum(w0[:3]**2))
    vscale = (m_scale / (3*menc))**(1/3.) * np.sqrt(np.sum(w0[3:]**2)) / np.sqrt(3)

    ball_w0 = np.zeros((N,6))
    ball_w0[:,:3] = np.random.normal(w0[:3], rscale, size=(N,3))
    ball_w0[:,3:] = np.random.normal(w0[3:], vscale, size=(N,3))

    return np.vstack((w0,ball_w0))
