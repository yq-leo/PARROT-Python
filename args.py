from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='ACM-DBLP-A',
                        choices=['ACM-DBLP-A', 'ACM-DBLP-P', 'cora', 'foursquare-twitter', 'phone-email', 'Douban', 'flickr-lastfm'],
                        help='datasets: ACM-DBLP-A; ACM-DBLP-P; cora; foursquare-twitter; phone-email; Douban; flickr-lastfm')

    return parser.parse_args()
