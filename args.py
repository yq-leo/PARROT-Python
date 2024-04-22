from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='ACM-DBLP-A',
                        choices=['ACM-DBLP-A', 'ACM-DBLP-P', 'cora', 'foursquare-twitter', 'phone-email'],
                        help='datasets: ACM-DBLP-A; ACM-DBLP-P; cora; foursquare-twitter; phone-email')

    return parser.parse_args()
