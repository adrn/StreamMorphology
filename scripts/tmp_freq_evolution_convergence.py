import os
import sys
import cPickle as pickle
from astropy import log as logger
import numpy as np
import gary.potential as gp
from gary.util import get_pool
from streammorphology.freqvar import FreqVariance

path = "/vega/astro/users/amp2217/projects/morphology/output/paper1-subgrid-17/"
cache_path = os.path.join(path, "convergence-test")
potential = gp.load(os.path.join(path, "potential.yml"))
_w0 = np.array([17.0, 0.0, 25.555555555555557, 0.0, 0.1331002672911241, 0.0])

def worker(args):
    p, window_width, nsteps_per_period = args
    cache_file = os.path.join(cache_path, "regular-p{0}_ww{1}_{2}perperiod.pickle".format(p, window_width, nsteps_per_period))
    if os.path.exists(cache_file):
        return

    result = FreqVariance.run(_w0, potential, nsteps_per_period=nsteps_per_period, 
                              window_stride=2., window_width=window_width,
                              total_nperiods=window_width+64, hamming_p=p)

    with open(cache_file, 'w') as f:
        pickle.dump(result, f)

    return

def main(mpi=False):
    pool = get_pool(mpi=mpi)

    # regular orbit, so variance should be small, but appears large due to aliasing...
    tasks = []
    for p in [1,2,3,4]:
        for window_width in 2**np.arange(5,8+1,1):
            for nsteps_per_period in 2**np.arange(8, 14+1, 2):
                tasks.append((p,window_width,nsteps_per_period))

    pool.map(worker, tasks)
    pool.close()
    sys.exit(0)


if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging
    parser = ArgumentParser()
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Use an MPI pool.")
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    main(args.mpi)
