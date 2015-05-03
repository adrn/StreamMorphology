# coding: utf-8

""" Check how timestep and number of steps for an orbit effects the frequencies. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys
import glob

# Third-party
import numpy as np

# Project
import gary.dynamics as gd
import gary.integrate as gi
from gary.util import get_pool

# project
from streammorphology.potential import potential_registry

dt = 1.
nsteps = 100000
path = "/vega/astro/users/amp2217/projects/morphology/output/two_balls/"
nball = 1000

def create_ball(w0, potential, N=100):
    menc = potential.mass_enclosed(w0)
    rscale = (2.5E4 / (3*menc))**(1/3.) * np.sqrt(np.sum(w0[:3]**2))
    vscale = (2.5E4 / (3*menc))**(1/3.) * np.sqrt(np.sum(w0[3:]**2)) / np.sqrt(3)

    ball_w0 = np.zeros((N,6))

    r = rscale * np.random.uniform(0,1.,N)**(1/3.)
    theta = np.arccos(2*np.random.uniform(size=N) - 1)
    phi = np.random.uniform(0,2*np.pi,N)

    ball_w0[:,0] = w0[0] + r*np.cos(phi)*np.sin(theta)
    ball_w0[:,1] = w0[1] + r*np.sin(phi)*np.sin(theta)
    ball_w0[:,2] = w0[2] + r*np.cos(theta)

    # direction to shift velocity
    ball_w0[:,3:] = np.random.normal(w0[3:], vscale, size=(N,3))

    return np.vstack((w0,ball_w0))

def worker(args):
    potential,w0,i,name = args

    orbit_file = os.path.join(path, "{}_orbit_{}.npy".format(name,i))
    time_file = os.path.join(path, "time.npy")
    freq_file = os.path.join(path, "{}_freqs_{}.npy".format(name,i))

    if os.path.exists(freq_file):
        return np.load(freq_file)

    if not os.path.exists(orbit_file):
        t,w = potential.integrate_orbit(w0, dt=dt, nsteps=nsteps*2,
                                        Integrator=gi.DOPRI853Integrator)

        np.save(orbit_file, w)
        if not os.path.exists(time_file):
            np.save(time_file, t)

    w = np.load(orbit_file)
    t = np.load(time_file)

    naff = gd.NAFF(t[:nsteps+1])
    fs = gd.cartesian_to_poincare_polar(w[:nsteps+1,0])
    try:
        freqs1,d,nvec = naff.find_fundamental_frequencies(fs)
    except:
        print("Orbit {}(1) failed for dt={}, nsteps={}".format(w0,dt,nsteps))
        freqs1 = np.ones(3)*np.nan

    naff = gd.NAFF(t[nsteps:])
    fs = gd.cartesian_to_poincare_polar(w[nsteps:,0])
    try:
        freqs2,d,nvec = naff.find_fundamental_frequencies(fs)
    except:
        print("Orbit {}(2) failed for dt={}, nsteps={}".format(w0,dt,nsteps))
        freqs2 = np.ones(3)*np.nan

    np.save(freq_file, np.vstack((freqs1,freqs2)))

def main(mpi=False):
    pool = get_pool(mpi=mpi)

    potential = potential_registry['triaxial-NFW']

    # slow and fast diffusing orbits
    w0 = np.array([[-17.504234723,-17.2283745157,-9.07711397761,0.0721992194851,0.021293129758,0.199775306493],
                   [-1.69295332221,-13.78418595,15.6309115075,0.0580704842,0.228735516722,0.0307028904261]])

    slow_ball0 = create_ball(w0[0], potential, nball)
    fast_ball0 = create_ball(w0[1], potential, nball)

    tasks = [[potential,slow_ball0[i],i,'slow'] for i in range(nball)]
    pool.map(worker, tasks)

    tasks = [[potential,fast_ball0[i],i,'fast'] for i in range(nball)]
    pool.map(worker, tasks)

    pool.close()

    slow_fqz = np.zeros((nball,2,3))
    for i,filename in enumerate(glob.glob(os.path.join(path,"slow_freq*"))):
        slow_fqz[i] = np.load(filename)

    fast_fqz = np.zeros((nball,2,3))
    for i,filename in enumerate(glob.glob(os.path.join(path,"fast_freq*"))):
        fast_fqz[i] = np.load(filename)

    print("slow:", np.std(slow_fqz[:,0],axis=0), np.std(slow_fqz[:,1],axis=0))
    print("fast:", np.std(fast_fqz[:,0],axis=0), np.std(fast_fqz[:,1],axis=0))

    sys.exit(0)

if __name__ == '__main__':
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Use an MPI pool.")

    args = parser.parse_args()

    main(mpi=args.mpi)
