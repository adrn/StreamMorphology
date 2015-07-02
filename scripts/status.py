# coding: utf-8

""" Check the status of frequency mapping. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

def main(path, class_name, config_name):
    cmd = "from streammorphology import {0}".format(class_name)
    exec(cmd)

    if config_name is None:
        config_name = "{0}.cfg".format(class_name)

    cmd = "experiment = {0}.from_config(cache_path=path, config_filename='{1}')"\
          .format(class_name, config_name)
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
    parser.add_argument("--cfg", dest="cfg_name", default=None, type=str,
                        help="Name of the config file.")

    args = parser.parse_args()

    main(args.path, class_name=args.class_name, config_name=args.cfg_name)
