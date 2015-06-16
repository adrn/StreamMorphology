# coding: utf-8

""" Check the status of frequency mapping. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

def main(path, class_name):
    cmd = "from streammorphology import {0}".format(class_name)
    exec(cmd)

    cmd = "experiment = {0}(cache_path=path)".format(class_name)
    exec(cmd)

    experiment.status()

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
