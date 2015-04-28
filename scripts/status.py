# coding: utf-8

""" Check the status of frequency mapping. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

def main(path, subproject):
    cmd = "from streammorphology.{0} import read, error_codes".format(subproject)
    exec(cmd)

    # read cache file
    d = read(path)

    ndone = (d['dt'] != 0.).sum()
    nsuccess = d['success'].sum()

    print("Number of orbits: {}".format(len(d)))
    print("Done processing: {}".format(ndone))
    print("Successful: {}".format(nsuccess))
    print("Failures: {}".format(ndone-nsuccess))

    for ecode in sorted(error_codes.keys()):
        nfail = (d['error_code'] == ecode).sum()
        print("\t({0}) {1}: {2}".format(ecode, error_codes[ecode], nfail))

if __name__ == '__main__':
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-p", "--path", dest="path", required=True,
                        help="Path to a Numpy memmap file containing the results "
                             "of frequency mapping.")
    parser.add_argument("-s", "--subproject", dest="subproject", required=True,
                        help="Name of the subproject to get reader and error "
                             "codes from. e.g., 'freqmap'")

    args = parser.parse_args()

    main(args.path, subproject=args.subproject)
