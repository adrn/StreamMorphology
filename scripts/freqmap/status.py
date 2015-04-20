# coding: utf-8

""" Check the status of frequency mapping. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import numpy as np

from streammorphology.freqmap import read_allfreqs, error_codes

def main(path, index=None):
    # read allfreqs into structured array
    d = read_allfreqs(path)

    nsuccess = d['success'].sum()
    fail = np.any(np.any(np.isnan(d['freqs']), axis=-1), axis=-1)
    nfail = fail.sum()

    print("Number of orbits: {0}".format(len(d)))
    print("Successful: {0}".format(nsuccess))
    print("Total num. failures: {0}".format(nfail))

    count = 0
    for k,error_msg in error_codes.items():
        nnn = (d['error_code'] == k).sum()
        count += nnn
        print("\t - {0}: {1}".format(nnn, error_msg))

    if count != nfail:
        print("\t - Unaccounted for: {0}".format(nfail - count))

    if index is not None:
        w0 = np.load(os.path.join(os.path.split(path)[0], 'w0.npy'))

        row = d[index]

        print("-"*79)
        print("w0: {}".format(w0[index]))
        print("max(∆E): {}".format(row['dE_max']))
        print("dt, nsteps: {}, {}".format(row['dt'], row['nsteps']))
        if row['is_tube']:
            print("Tube orbit")
        else:
            print("Box orbit")

        fmt = "\t {0:.8e} {1:.8e} {2:.8e}"
        print("1st half freqs.:")
        print(fmt.format(*row['freqs'][0]))
        print("2nd half freqs.:")
        print(fmt.format(*row['freqs'][1]))

if __name__ == '__main__':
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-p", "--path", dest="path", required=True,
                        help="Path to a Numpy memmap file containing the results "
                             "of frequency mapping.")
    parser.add_argument("-i", "--index", dest="index", default=None,
                        help="Index of an orbit to check.")

    args = parser.parse_args()

    main(args.path, index=args.index)
