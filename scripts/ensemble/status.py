# coding: utf-8

""" Check the status of frequency mapping. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

from streammorphology.ensemble import read_allkld

def main(path, nkld):
    # read allfreqs into structured array
    d = read_allkld(path, nkld=nkld)

    ndone = (d['dt'] != 0.).sum()
    nsuccess = d['success'].sum()
    nfail_energy = (d['error_code'] == 1).sum()
    nfail_integrate = (d['error_code'] == 2).sum()
    nfail_kld = (d['error_code'] == 3).sum()

    print("Number of orbits: {}".format(len(d)))
    print("Done processing: {}".format(ndone))
    print("Successful: {}".format(nsuccess))
    print("Failures: {}".format(ndone-nsuccess))
    print("\t Energy failure: {}".format(nfail_energy))
    print("\t Integration failure: {}".format(nfail_integrate))
    print("\t KLD failure: {}".format(nfail_kld))

if __name__ == '__main__':
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-p", "--path", dest="path", required=True,
                        help="Path to a Numpy memmap file containing the results "
                             "of frequency mapping.")
    parser.add_argument("--nkld", dest="nkld", required=True, type=int,
                        help="Number of times the KLD was computed and saved.")

    args = parser.parse_args()

    main(args.path, nkld=args.nkld)
