import argparse

from TrackEval.scripts.run_mot_challenge import evaluate_mot17

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--trackers-path', type=str, default='trackers',
                        help='Path to the folder containing all trackers.')

    parser.add_argument('-s', '--seqmap', type=str,
                        help='The name of the seqmap containing the list of tracks to evaluate.\n'
                             'All seqmaps for MOTX are assumed to be in MOTX/seqmaps.')

    parser.add_argument('--datapath', type=str, default='data',  # TODO remove default
                        help='Path to the folder containing all datasets.')

    parser.add_argument('--output-path', type=str, default='output',
                        help='Path to the folder where the results will be saved.')

    # Not really sure of its role
    parser.add_argument('-e', '--experiment', type=str,
                        help='Subfolder of tracker_path where the tracker files are located.')

    args = parser.parse_args()

    evaluate_mot17(tracker_path=args.trackers_path,
                   split=args.seqmap,
                   data_path=args.datapath,
                   tracker_sub_folder=args.experiment,
                   output_sub_folder=args.output_path)
