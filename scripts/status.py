# coding: utf-8

""" Check the status of frequency mapping. """

from __future__ import division, print_function

# Standard library
import os

__author__ = "adrn <adrn@astro.columbia.edu>"

def main(path, class_name):
    cmd = "from streammorphology import {0}".format(class_name)
    exec(cmd)

    base_path = os.path.split(path)[0]
    cmd = "experiment = {0}(cache_path=base_path)".format(class_name)
    exec(cmd)

    d = experiment.read_cache()

    # ndone = (d['dt'] != 0.).sum()
    nsuccess = d['success'].sum()
    nfail = ((d['success'] is False) & (d['error_code'] > 0)).sum()

    print("Number of orbits: {}".format(len(d)))
    # print("Done processing: {}".format(ndone))
    print("Successful: {}".format(nsuccess))
    print("Failures: {}".format(nfail))

    for ecode in sorted(experiment.error_codes.keys()):
        nfail = (d['error_code'] == ecode).sum()
        print("\t({0}) {1}: {2}".format(ecode, experiment.error_codes[ecode], nfail))

if __name__ == '__main__':
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-p", "--path", dest="path", required=True,
                        help="Path to a Numpy memmap file containing the results "
                             "of frequency mapping.")
    parser.add_argument("-c", "--class", dest="class_name", required=True,
                        help="Name of the experiment Class to get reader and error "
                             "codes from. e.g., 'Freqmap'")

    args = parser.parse_args()

    main(args.path, class_name=args.class_name)
