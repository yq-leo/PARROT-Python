from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='ACM-DBLP-A',
                        choices=['ACM-DBLP-A', 'ACM-DBLP-P', 'cora', 'foursquare-twitter', 'phone-email', 'Douban', 'flickr-lastfm', 'ogbl-biokg'],
                        help='datasets: ACM-DBLP-A; ACM-DBLP-P; cora; foursquare-twitter; phone-email; Douban; flickr-lastfm; ogbl-biokg')
    parser.add_argument('--runs', dest='runs', type=int, default=1, help='number of runs')
    parser.add_argument('--record', dest='record', action='store_true', help='record results')

    return parser.parse_args()
