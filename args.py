from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='ACM-DBLP-A',
                        choices=['ACM-DBLP-A', 'ACM-DBLP-P', 'cora', 'foursquare-twitter', 'phone-email', 'Douban', 'flickr-lastfm'],
                        help='datasets: ACM-DBLP-A; ACM-DBLP-P; cora; foursquare-twitter; phone-email; Douban; flickr-lastfm')
    parser.add_argument('--shuffle', dest='shuffle', type=str, default='off',
                        choices=['off', 'imbalanced', 'balanced'], help='shuffle options: off; imbalanced; balanced')
    parser.add_argument('--no_edge_reg', dest='no_edge_reg', action='store_true',
                        help='no edge consistency regularization')
    parser.add_argument('--no_neigh_reg', dest='no_neigh_reg', action='store_true',
                        help='no neighbor consistency regularization')
    parser.add_argument('--no_pref_reg', dest='no_pref_reg', action='store_true',
                        help='no preference consistency regularization')
    parser.add_argument('--no_joint_rwr', dest='no_joint_rwr', action='store_true',
                        help='no joint random walk with restart')
    parser.add_argument('--use_pgna', dest='use_pgna', action='store_true',
                        help='use PGNA embeddings to calculate transport cost')
    parser.add_argument('--use_num', dest='use_num', action='store_true', help='use non-uniform marginals')
    parser.add_argument('--use_pgna_num', dest='use_pgna_num', action='store_true', help='use non-uniform marginals with PGNA')
    parser.add_argument('--record', dest='record', action='store_true', help='record results')
    parser.add_argument('--inIter', dest='inIter', type=int, default=-1, help='number of inner iterations')
    parser.add_argument('--outIter', dest='outIter', type=int, default=-1, help='number of outer iterations')
    parser.add_argument('--self_train', dest='self_train', type=str, default='off', choices=['off', 'god', 'hit'])

    # Experiment settings
    parser.add_argument('--edge_noise', dest='edge_noise', type=float, default=0.0, help='edge noise')
    parser.add_argument('--attr_noise', dest='attr_noise', type=float, default=0.0, help='attribute noise')
    parser.add_argument('--use_attr', dest='use_attr', action='store_true', help='use input node attributes')
    parser.add_argument('--runs', dest='runs', type=int, default=1, help='number of runs')

    return parser.parse_args()
