# coding: utf-8

""" Check the status of Lyapunov mapping. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

from streammorphology.lyapunov import read_alllyap

def main(path):
    # read allfreqs into structured array
    d = read_alllyap(path)

    ndone = (d['lyap_exp'] != 0).sum()
    nsuccess = d['success'].sum()
    nfail_energy = (d['error_code'] == 1).sum()
    nfail_integrate = (d['error_code'] == 2).sum()

    print("Number of orbits: {}".format(len(d)))
    print("Done processing: {}".format(ndone))
    print("Successful: {}".format(nsuccess))
    print("Failures: {}".format(ndone-nsuccess))
    print("\t Energy failure: {}".format(nfail_energy))
    print("\t Integration failure: {}".format(nfail_integrate))

if __name__ == '__main__':
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-p", "--path", dest="path", required=True,
                        help="Path to a Numpy memmap file containing the results "
                             "of frequency mapping.")

    args = parser.parse_args()

    main(args.path)
