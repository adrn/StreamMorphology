# coding: utf-8

""" Check the status of frequency mapping. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import numpy as np

def main(path):
    basepath = os.path.split(path)[0]
    w0_path = os.path.join(basepath, 'w0.npy')

    w0 = np.load(w0_path)
    d = np.memmap(path, mode='r', shape=(len(w0),2,8), dtype='float64')

    n_done = int(np.any(d[:,0] != 0., axis=1).sum())
    n_fail_some = int((np.any(np.isnan(d[:,0,:6]), axis=1) | np.any(np.isnan(d[:,1,:6]), axis=1)).sum())
    n_total_fail = int(np.all(np.isnan(d[:,0,:6]), axis=1).sum())

    print("Completed: {}".format(n_done))
    print("Some failures: {}".format(n_fail_some))
    print("Total failures: {}".format(n_total_fail))

if __name__ == '__main__':
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-p", "--path", dest="path", required=True,
                        help="Path to a Numpy memmap file containing the results "
                             "of frequency mapping.")

    args = parser.parse_args()

    main(args.path)
